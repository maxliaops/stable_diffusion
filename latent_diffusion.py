"""
---
标题：潜在扩散模型
摘要：>
  带注释的 PyTorch 实现/教程的潜在扩散模型来自论文
  《使用潜在扩散模型的高分辨率图像合成》
---

# 潜在扩散模型

潜在扩散模型使用自动编码器在图像空间和潜在空间之间进行映射。扩散模型在潜在空间上工作，这使得训练变得容易得多。
它基于论文
[使用潜在扩散模型的高分辨率图像合成](https://arxiv.org/abs/2112.10752)。

它们使用预训练的自动编码器，并在预训练的自动编码器的潜在空间上训练扩散 U 型网络。

对于更简单的扩散实现，请参考我们的[DDPM 实现](../ddpm/index.html)。
我们对$\alpha_t$，$\beta_t$调度等使用相同的符号。

"""

from typing import List

import torch
import torch.nn as nn

from model.autoencoder import Autoencoder
from model.clip_embedder import CLIPTextEmbedder
from model.unet import UNetModel


class DiffusionWrapper(nn.Module):
    """
    *这是[U 型网络](model/unet.html)的空包装类。
    我们保留这个，以使其具有与
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)相同的模型结构
    ，这样我们就不必明确映射检查点权重*。
    """

    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    """
    ## 潜在扩散模型

    这包含以下组件：

    * [自动编码器](model/autoencoder.html)
    * 带有[注意力](model/unet_attention.html)的[U 型网络](model/unet.html)
    * [CLIP 嵌入生成器](model/clip_embedder.html)
    """
    model: DiffusionWrapper
    first_stage_model: Autoencoder
    cond_stage_model: CLIPTextEmbedder

    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 ):
        """
        :param unet_model: 是预测噪声$\epsilon_\text{cond}(x_t, c)$的[U 型网络](model/unet.html)，在潜在空间中
         $x_t$。
        :param autoencoder: 是[自动编码器](model/autoencoder.html)
        :param clip_embedder: 是[CLIP 嵌入生成器](model/clip_embedder.html)
        :param latent_scaling_factor: 是潜在空间的缩放因子。自动编码器的编码通过此进行缩放，然后再输入到 U 型网络中。
        :param n_steps: 是扩散步骤$T$的数量。
        :param linear_start: 是$\beta$调度的开始。
        :param linear_end: 是$\beta$调度的结束。
        """
        super().__init__()
        # 将[U 型网络](model/unet.html)包装起来，以保持与
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)相同的模型结构。
        self.model = DiffusionWrapper(unet_model)
        # 自动编码器和缩放因子
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        # [CLIP 嵌入生成器](model/clip_embedder.html)
        self.cond_stage_model = clip_embedder

        # 步骤数$T$
        self.n_steps = n_steps

        # $\beta$调度
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        ### 获取模型设备
        """
        return next(iter(self.model.parameters())).device

    def get_text_conditioning(self, prompts: List[str]):
        """
        ### 获取文本提示列表的[CLIP 嵌入](model/clip_embedder.html)
        """
        return self.cond_stage_model(prompts)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### 获取图像的缩放潜在空间表示

        编码器输出是一个分布。
        我们从该分布中进行采样，并乘以缩放因子。
        """
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### 从潜在表示中获取图像

        我们按缩放因子缩放下来，然后进行解码。
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### 预测噪声

        根据潜在表示$x_t$，时间步$t$和条件上下文$c$预测噪声。

        $\epsilon_\text{cond}(x_t, c)$
        """
        return self.model(x, t, context)