"""
---
title: 用于稳定扩散的 U-Net
summary: >
 稳定扩散中 U-Net 的带注释的 PyTorch 实现/教程
---

# 用于[稳定扩散](../index.html)的 U-Net

这实现了给出$\epsilon_\text{cond}(x_t, c)$的 U-Net

我们保留了与[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)相同的模型定义和命名，
以便我们可以直接加载检查点。
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet_attention import SpatialTransformer


class UNetModel(nn.Module):
    """
    ## U-Net 模型
    """

    def __init__(
            self, *,
            in_channels: int,
            out_channels: int,
            channels: int,
            n_res_blocks: int,
            attention_levels: List[int],
            channel_multipliers: List[int],
            n_heads: int,
            tf_layers: int = 1,
            d_cond: int = 768):
        """
        :param in_channels: 输入特征图的通道数
        :param out_channels: 输出特征图的通道数
        :param channels: 模型的基础通道数
        :param n_res_blocks: 每个级别中的残差块数量
        :param attention_levels: 需要进行注意力操作的级别
        :param channel_multipliers: 每个级别的通道乘数
        :param n_heads: 注意力头的数量
        :param tf_layers: 变换器层的数量
        :param d_cond: 变换器中的条件嵌入大小
        """
        super().__init__()
        self.channels = channels

        # 级别数量
        levels = len(channel_multipliers)
        # 时间嵌入的大小
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # U-Net 的输入部分
        self.input_blocks = nn.ModuleList()
        # 初始的 $3 \times 3$ 卷积，将输入映射到 `channels`
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels, channels, 3, padding=1)))
        # U-Net 输入部分每个级别中的通道数
        input_block_channels = [channels]
        # U-Net 每个级别的通道数
        channels_list = [channels * m for m in channel_multipliers]
        # 准备级别
        for i in range(levels):
            # 添加残差块和注意力
            for _ in range(n_res_blocks):
                # 残差块从先前的通道数映射到当前级别的通道数
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                # 添加变换器
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                # 添加到 U-Net 的输入部分，并跟踪其输出的通道数
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            # 除最后一级外，在所有级别下降采样
            if i!= levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # U-Net 的中间部分
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, d_time_emb),
        )

        # U-Net 的输出部分
        self.output_blocks = nn.ModuleList([])
        # 以相反的顺序准备级别
        for i in reversed(range(levels)):
            # 添加残差块和注意力
            for j in range(n_res_blocks + 1):
                # 残差块从输入半 U-Net 的先前通道数加上跳过连接映射到当前级别的通道数
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                # 添加变换器
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                # 在除最后一个残差块后的所有级别上进行上采样，除了最后一个级别
                if i!= 0 and j == n_res_blocks:
                    layers.append(UpSample(channels))
                # 添加到 U-Net 的输出部分
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # 最终的归一化和 $3 \times 3$ 卷积
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## 创建正弦时间步嵌入

        :param time_steps: 时间步的张量，形状为 `[batch_size]`
        :param max_period: 控制嵌入的最小频率
        """
        # 通道数的一半，sin 和 cos 各占一半
        half = self.channels // 2
        # $10000^{-\frac{2i}{c}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $t10000^{-\frac{2i}{c}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ 和 $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        """
        :param x: 输入特征图的张量，形状为 `[batch_size, channels, width, height]`
        :param time_steps: 时间步的张量，形状为 `[batch_size]`
        :param cond: 条件的张量，形状为 `[batch_size, n_cond, d_cond]`
        """
        # 用于存储输入半部分的输出以进行跳过连接
        x_input_block = []

        # 获取时间步嵌入
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # U-Net 的输入半部分
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        # U-Net 的中间部分
        x = self.middle_block(x, t_emb, cond)
        # U-Net 的输出半部分
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # 最终的归一化和 $3 \times 3$ 卷积
        return self.out(x)

class TimestepEmbedSequential(nn.Sequential):
    """
    ### 具有不同输入的顺序模块

    这个顺序模块可以由诸如 `ResBlock`、`nn.Conv` 和 `SpatialTransformer` 等不同模块组成，并使用匹配的签名调用它们
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """
    ### 上采样层
    """

    def __init__(self, channels: int):
        """
        :param channels: 通道数
        """
        super().__init__()
        # $3 \times 3$ 卷积映射
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: 形状为 `[batch_size, channels, height, width]` 的输入特征图
        """
        # 上采样因子为 $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 应用卷积
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## 下采样层
    """

    def __init__(self, channels: int):
        """
        :param channels: 通道数
        """
        super().__init__()
        # $3 \times 3$ 卷积，步长为 $2$，下采样因子为 $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: 形状为 `[batch_size, channels, height, width]` 的输入特征图
        """
        # 应用卷积
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet 块
    """

    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """
        :param channels: 输入通道数
        :param d_t_emb: 时间步嵌入的大小
        :param out_channels: 输出通道数。默认为 `channels`。
        """
        super().__init__()
        # `out_channels` 未指定
        if out_channels is None:
            out_channels = channels

        # 第一次归一化和卷积
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # 时间步嵌入
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # 最终卷积层
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # 从 `channels` 到 `out_channels` 的映射层，用于残差连接
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: 形状为 `[batch_size, channels, height, width]` 的输入特征图
        :param t_emb: 形状为 `[batch_size, d_t_emb]` 的时间步嵌入
        """
        # 初始卷积
        h = self.in_layers(x)
        # 时间步嵌入
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # 添加时间步嵌入
        h = h + t_emb[:, :, None, None]
        # 最终卷积
        h = self.out_layers(h)
        # 添加跳过连接
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### 浮点 32 位的组归一化

    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### 组归一化

    这是一个辅助函数，具有固定数量的组..
    """
    return GroupNorm32(32, channels)


def _test_time_embeddings():
    """
    测试正弦时间步嵌入
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    m = UNetModel(in_channels=1, out_channels=1, channels=320, n_res_blocks=1, attention_levels=[],
                  channel_multipliers=[],
                  n_heads=1, tf_layers=1, d_cond=1)
    te = m.time_step_embedding(torch.arange(0, 1000))
    plt.plot(np.arange(1000), te[:, [50, 100, 190, 260]].numpy())
    plt.legend(["dim %d" % p for p in [50, 100, 190, 260]])
    plt.title("Time embeddings")
    plt.show()


#
if __name__ == '__main__':
    _test_time_embeddings()