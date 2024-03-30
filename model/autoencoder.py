"""
---
title: 用于 Stable Diffusion 的自动编码器
summary: >
  自动编码器的带注释的 PyTorch 实现/教程
---

# 用于[Stable Diffusion](../index.html)的自动编码器

此代码实现了用于在图像空间和潜在空间之间进行映射的自动编码器。

我们保留了从
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
的模型定义和命名不变，以便直接加载检查点。
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    """
    ## 自动编码器

    它由编码器和解码器模块组成。
    """

    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', emb_channels: int, z_channels: int):
        """
        :param encoder: 编码器
        :param decoder: 解码器
        :param emb_channels: 量化嵌入空间的维度数
        :param z_channels: 嵌入空间的通道数
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 从嵌入空间到
        # 量化嵌入空间时刻（均值和对数方差）的卷积
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # 从量化嵌入空间回到
        # 嵌入空间的卷积
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        """
        ### 将图像编码为潜在表示

        :param img: 形状为 `[batch_size, img_channels, img_height, img_width]` 的图像张量
        """
        # 获取具有形状 `[batch_size, z_channels * 2, z_height, z_height]` 的嵌入
        z = self.encoder(img)
        # 获取量化嵌入空间中的时刻
        moments = self.quant_conv(z)
        # 返回分布
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        """
        ### 从潜在表示解码图像

        :param z: 形状为 `[batch_size, emb_channels, z_height, z_height]` 的潜在表示
        """
        # 从量化表示映射到嵌入空间
        z = self.post_quant_conv(z)
        # 解码形状为 `[batch_size, channels, height, width]` 的图像
        return self.decoder(z)


class Encoder(nn.Module):
    """
    ## 编码器模块
    """

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 in_channels: int, z_channels: int):
        """
        :param channels: 第一层卷积层中的通道数
        :param channel_multipliers: 后续块中通道数的乘法因子
        :param n_resnet_blocks: 每个分辨率的 ResNet 层数量
        :param in_channels: 图像中的通道数
        :param z_channels: 嵌入空间中的通道数
        """
        super().__init__()

        # 不同分辨率的块数。
        # 每次在最后一个顶级块之后分辨率减半
        n_resolutions = len(channel_multipliers)

        # 初始 $3 \times 3$ 卷积层，将图像映射到 `channels`
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)

        # 每个顶级块中的通道数列表
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # 顶级块列表
        self.down = nn.ModuleList()
        # 创建顶级块
        for i in range(n_resolutions):
            # 每个顶级块由多个 ResNet Blocks 和下采样组成
            resnet_blocks = nn.ModuleList()
            # 添加 ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
            # 顶级块
            down = nn.Module()
            down.block = resnet_blocks
            # 除最后一个块外，每个顶级块末尾的下采样
            if i!= n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            #
            self.down.append(down)

        # 具有注意力的最终 ResNet 块
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # 映射到嵌入空间的 $3 \times 3$ 卷积
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)

    def forward(self, img: torch.Tensor):
        """
        :param img: 形状为 `[batch_size, img_channels, img_height, img_width]` 的图像张量
        """

        # 用初始卷积映射到 `channels`
        x = self.conv_in(img)

        # 顶级块
        for down in self.down:
            # ResNet Blocks
            for block in down.block:
                x = block(x)
            # 下采样
            x = down.downsample(x)

        # 具有注意力的最终 ResNet 块
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # 归一化并映射到嵌入空间
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        #
        return x

class Decoder(nn.Module):
    """
    ## Decoder 模块
    """

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 out_channels: int, z_channels: int):
        """
        :param channels: 是最终卷积层中的通道数
        :param channel_multipliers: 是前几个块中通道数的乘法因子，按反向顺序
        :param n_resnet_blocks: 是每个分辨率的 ResNet 层数量
        :param out_channels: 是图像中的通道数
        :param z_channels: 是嵌入空间中的通道数
        """
        # 继承 nn.Module
        super().__init__()

        # 不同分辨率的块数
        # 每个顶级块的分辨率在最后一个顶级块之后减半
        num_resolutions = len(channel_multipliers)

        # 按反向顺序的每个顶级块中的通道数列表
        channels_list = [m * channels for m in channel_multipliers]

        # 顶级块中的通道数
        channels = channels_list[-1]

        # 将嵌入空间映射到 `channels` 的初始 $3 \times 3$ 卷积层
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # 具有注意力的 ResNet 块
        self.mid = nn.Module()
        # 第一个 ResNet 块
        self.mid.block_1 = ResnetBlock(channels, channels)
        # 第一个注意力块
        self.mid.attn_1 = AttnBlock(channels)
        # 第二个 ResNet 块
        self.mid.block_2 = ResnetBlock(channels, channels)

        # 顶级块列表
        self.up = nn.ModuleList()
        # 创建顶级块
        for i in reversed(range(num_resolutions)):
            # 每个顶级块由多个 ResNet 块和上采样组成
            resnet_blocks = nn.ModuleList()
            # 添加 ResNet 块
            for _ in range(n_resnet_blocks + 1):
                # 每个顶级块中除了第一个块之外的 ResNet 块数量
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                # 更新通道数
                channels = channels_list[i]
            # 顶级块
            up = nn.Module()
            up.block = resnet_blocks
            # 除了第一个顶级块之外的每个顶级块末尾的上采样
            if i!= 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            # 为了与检查点保持一致，将其前置
            self.up.insert(0, up)

        # 映射到图像空间的 $3 \times 3$ 卷积
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        """
        :param z: 是形状为 `[batch_size, z_channels, z_height, z_height]` 的嵌入张量
        """

        # 用初始卷积将其映射到 `channels`
        h = self.conv_in(z)

        # 具有注意力的 ResNet 块
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # 顶级块
        for up in reversed(self.up):
            # ResNet 块
            for block in up.block:
                h = block(h)
            # 上采样
            h = up.upsample(h)

        # 归一化并映射到图像空间
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        #
        return img

class GaussianDistribution:
    """
    ## 高斯分布
    """

    def __init__(self, parameters: torch.Tensor):
        """
        :param parameters: 是形状为 `[batch_size, z_channels * 2, z_height, z_height]` 的嵌入的均值和对数方差张量
        """
        # 拆分均值和对数方差
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # 对数方差的夹紧
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # 计算标准差
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # 从分布中采样
        return self.mean + self.std * torch.randn_like(self.std)


class AttnBlock(nn.Module):
    """
    ## 注意力块
    """

    def __init__(self, channels: int):
        """
        :param channels: 是通道数
        """
        super().__init__()
        # 组归一化
        self.norm = normalization(channels)
        # 查询、键和值的映射
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        # 最终 $1 \times 1$ 卷积层
        self.proj_out = nn.Conv2d(channels, channels, 1)
        # 注意力缩放因子
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        """
        :param x: 是形状为 `[batch_size, channels, height, width]` 的张量
        """
        # 规范化 `x`
        x_norm = self.norm(x)
        # 获取查询、键和向量嵌入
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # 将查询、键和向量嵌入从
        # `[batch_size, channels, height, width]` 重塑为
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # 计算 $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # 计算 $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum('bij,bcj->bci', attn, v)

        # 重新塑形为 `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w)
        # 最终 $1 \times 1$ 卷积层
        out = self.proj_out(out)

        # 添加残差连接
        return x + out


class UpSample(nn.Module):
    """
    ## 上采样层
    """
    def __init__(self, channels: int):
        """
        :param channels: 是通道数
        """
        super().__init__()
        # $3 \times 3$ 卷积映射
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: 是形状为 `[batch_size, channels, height, width]` 的输入特征图
        """
        # 以因子 $2$ 上采样
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # 应用卷积
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## 下采样层
    """
    def __init__(self, channels: int):
        """
        :param channels: 是通道数
        """
        super().__init__()
        # $3 \times 3$ 卷积，步长为 $2$，以因子 $2$ 下采样
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        """
        :param x: 是形状为 `[batch_size, channels, height, width]` 的输入特征图
        """
        # 添加填充
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        # 应用卷积
        return self.conv(x)


class ResnetBlock(nn.Module):
    """
    ## ResNet 块
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super().__init__()
        # 第一次归一化和卷积层
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        # 第二次归一化和卷积层
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        # `in_channels` 到 `out_channels` 的映射层，用于残差连接
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        :param x: 是形状为 `[batch_size, channels, height, width]` 的输入特征图
        """

        h = x

        # 第一次归一化和卷积层
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # 第二次归一化和卷积层
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        # 映射并添加残差
        return self.nin_shortcut(x) + h


def swish(x: torch.Tensor):
    """
    ### Swish 激活

    $$x \cdot \sigma(x)$$
    """
    return x * torch.sigmoid(x)


def normalization(channels: int):
    """
    ### 组归一化

    这是一个辅助函数，具有固定数量的组和 `eps`
    """
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)