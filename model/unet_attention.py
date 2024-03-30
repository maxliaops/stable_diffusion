"""
---
title: 用于稳定扩散 U-Net 的 Transformer
summary: >
 用于稳定扩散中 U-Net 的 Transformer 的带注释的 PyTorch 实现/教程
---

# 用于稳定扩散 U-Net 的 Transformer

这实现了在[U-Net](unet.html)中使用的 Transformer 模块，它给出了$\epsilon_\text{cond}(x_t, c)$

我们保持了与[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)相同的模型定义和命名，以便可以直接加载检查点。
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """
    ## 空间 Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: 是特征图中的通道数
        :param n_heads: 是注意力头的数量
        :param n_layers: 是 Transformer 层的数量
        :param d_cond: 是条件嵌入的大小
        """
        # 继承 nn.Module
        super().__init__()
        # 初始组归一化
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        # 初始$1 \times 1$卷积
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer 层
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )

        # 最终$1 \times 1$卷积
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: 是形状为[batch_size, channels, height, width]的特征图
        :param cond: 是形状为[batch_size,  n_cond, d_cond]的条件嵌入
        """
        # 获取形状[batch_size, channels, height, width]
        b, c, h, w = x.shape
        # 用于残差连接
        x_in = x
        # 归一化
        x = self.norm(x)
        # 初始$1 \times 1$卷积
        x = self.proj_in(x)
        # 从[batch_size, channels, height, width]
        # 转置并重塑为[batch_size, height * width, channels]
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # 应用 Transformer 层
        for block in self.transformer_blocks:
            x = block(x, cond)
        # 从[batch_size, height * width, channels]
        # 重塑并转置为[batch_size, channels, height, width]
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # 最终$1 \times 1$卷积
        x = self.proj_out(x)
        # 添加残差
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer 层
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: 是输入嵌入的大小
        :param n_heads: 是注意力头的数量
        :param d_head: 是注意力头的大小
        :param d_cond: 是条件嵌入的大小
        """
        # 继承 nn.Module
        super().__init__()
        # 自注意力层和前置归一化层
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # 交叉注意力层和前置归一化层
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        # 前馈网络和前置归一化层
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: 是形状为[batch_size, height * width, d_model]的输入嵌入
        :param cond: 是形状为[batch_size,  n_cond, d_cond]的条件嵌入
        """
        # 自注意力
        x = self.attn1(self.norm1(x)) + x
        # 与条件的交叉注意力
        x = self.attn2(self.norm2(x), cond=cond) + x
        # 前馈网络
        x = self.ff(self.norm3(x)) + x
        #
        return x

class CrossAttention(nn.Module):
    """
    ### 交叉注意力层

    当没有指定条件嵌入时，会退回到自注意力
    """

    use_flash_attention: bool = False

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: 是输入嵌入的大小
        :param n_heads: 是注意力头的数量
        :param d_head: 是一个注意力头的大小
        :param d_cond: 是条件嵌入的大小
        :param is_inplace: 指定是否在原地执行注意力 softmax 计算以节省内存
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # 注意力缩放因子
        self.scale = d_head ** -0.5

        # 查询、键和值的映射
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # 最终线性层
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # 设置 [flash attention](https://github.com/HazyResearch/flash-attention)。
        # Flash attention 仅在安装并`CrossAttention.use_flash_attention`设置为`True`时使用
        try:
            # 您可以通过克隆他们的 Github 仓库，
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # 然后运行`python setup.py install`来安装 flash attention
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()
            # 设置用于缩放点积注意力的缩放因子
            self.flash.softmax_scale = self.scale
        # 如果未安装，则设置为`None`
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: 是形状为`[batch_size, height * width, d_model]`的输入嵌入
        :param cond: 是形状为`[batch_size, n_cond, d_cond]`的条件嵌入
        """

        # 如果`cond`为`None`，我们执行自注意力
        has_cond = cond is not None
        if not has_cond:
            cond = x

        # 获取查询、键和值向量
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # 如果可用并且头大小小于或等于`128`，则使用 flash 注意力
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        # 否则，回退到正常注意力
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash 注意力

        :param q: 是在拆分头之前的查询向量，形状为`[batch_size, seq, d_attn]`
        :param k: 是在拆分头之前的键向量，形状为`[batch_size, seq, d_attn]`
        :param v: 是在拆分头之前的值向量，形状为`[batch_size, seq, d_attn]`
        """

        # 获取批大小和序列轴上的元素数量（`width * height`）
        batch_size, seq_len, _ = q.shape

        # 将`q`、`k`、`v`向量堆叠起来，以得到一个单一的张量
        # 形状为`[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # 拆分头
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash 注意力适用于头大小`32`、`64`和`128`，因此我们必须填充头以适应此大小
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'头大小{self.d_head}对于 Flash 注意力太大')

        # 填充头
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1)

        # 计算注意力
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # 这给出了一个形状为`[batch_size, seq_len, n_heads, d_padded]`的张量
        out, _ = self.flash(qkv)
        # 截断额外的头大小
        out = out[:, :, :, :self.d_head]
        # 重塑为`[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # 用线性层映射到`[batch_size, height * width, d_model]` 
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### 正常注意力

        :param q: 是在拆分头之前的查询向量，形状为`[batch_size, seq, d_attn]`
        :param k: 是在拆分头之前的键向量，形状为`[batch_size, seq, d_attn]`
        :param v: 是在拆分头之前的值向量，形状为`[batch_size, seq, d_attn]`
        """

        # 将它们拆分为头的形状`[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # 计算注意力$\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # 计算 softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # 计算注意力输出
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # 重塑为`[batch_size, height * width, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # 用线性层映射到`[batch_size, height * width, d_model]` 
        return self.to_out(out)

class FeedForward(nn.Module):
    """
    ### Feed-Forward Network（前馈网络）
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        """
        :param d_model: 输入嵌入的大小
        :param d_mult: 隐藏层大小的乘法因子
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),  # 构建 GeGLU 模块
            nn.Dropout(0.),  # 应用 dropout 操作
            nn.Linear(d_model * d_mult, d_model)  # 构建线性层
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)  # 前向传播


class GeGLU(nn.Module):
    """
    ### GeGLU Activation（GeGLU 激活函数）

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # 组合线性投影 $xW + b$ 和 $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # 获取 $xW + b$ 和 $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)