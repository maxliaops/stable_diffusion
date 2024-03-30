"""
---
title: 去噪扩散隐式模型（DDIM）采样
summary: >
 稳定扩散模型的带注释的 PyTorch 实现/教程
 去噪扩散隐式模型（DDIM）采样
---

# 去噪扩散隐式模型（DDIM）采样

这实现了论文《去噪扩散隐式模型》中的 DDMI 采样
[Denosing Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
"""

from typing import Optional, List
import numpy as np
import torch
from labml import monit
from latent_diffusion import LatentDiffusion
from sampler import DiffusionSampler


class DDIMSampler(DiffusionSampler):
    """
    ## DDIM Sampler

    这扩展了[`DiffusionSampler`基础类](index.html)。

    DDIM 通过反复去除噪声并按步采样来采样图像，使用的公式为：

    \begin{align*}
    x_{\tau_{i-1}}&=\sqrt{\alpha_{\tau_{i-1}}}\Bigg(
            \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
            \Bigg) \\
            &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
            &+ \sigma_{\tau_i} \epsilon_{\tau_i}
    \end{align*}

    其中$\epsilon_{\tau_i}$是随机噪声，
    $\tau$是长度为$S$的$[1,2,\dots,T]$的子序列，
    并且
    $$\sigma_{\tau_i} =
    \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
    \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$

    请注意，DDIM 论文中的$\alpha_t$指的是[DDPM](ddpm.html)中的${\color{lightgreen}\bar\alpha_t}$。

    """

    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        """
        :param model: 是预测噪声$\epsilon_\text{cond}(x_t, c)$的模型
        :param n_steps: 是 DDIM 采样步骤的数量，$S$
        :param ddim_discretize: 指定如何从$[1,2,\dots,T]$中提取$\tau$。
            它可以是“uniform”或“quad”。
        :param ddim_eta: 是用于计算$\sigma_{\tau_i}$的$\eta$。$\eta=0$使采样过程确定。
        """
        super().__init__(model)
        # 采样步骤数量，$T$
        self.n_steps = model.n_steps

        # 计算$\tau$在$[1,2,\dots,T]$上均匀分布
        if ddim_discretize == 'uniform':
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1
        # 计算$\tau$在$[1,2,\dots,T]$上呈二次分布
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps *.8), n_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # 获取${\color{lightgreen}\bar\alpha_t}$
            alpha_bar = self.model.alpha_bar

            # $\alpha_{\tau_i}$
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            # $\sqrt{\alpha_{\tau_i}}$
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            # $\alpha_{\tau_{i-1}}$
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]])

            # $$\sigma_{\tau_i} =
            # \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
            # \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) **.5)

            # $\sqrt{1 - \alpha_{\tau_i}}$
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) **.5

    @torch.no_grad()
    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """
        ### 采样循环

        :param shape: 是生成图像的形状，形式为[batch_size, channels, height, width]
        :param cond: 是条件嵌入$c$
        :param temperature: 是噪声温度（随机噪声乘以这个）
        :param x_last: 是$x_{\tau_S}$。如果未提供，则使用随机噪声。
        :param uncond_scale: 是无条件引导比例$s$。这用于
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: 是空提示的条件嵌入$c_u$
        :param skip_steps: 是要跳过的时间步长$i'$。我们从$S - i'$开始采样。然后$x_last$就是$x_{\tau_{S - i'}}$。
        """

        # 获取设备和批次大小
        device = self.model.device
        bs = shape[0]

        # 获取$x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # 要在$\tau_{S - i'}$，$\tau_{S - i' - 1}$，...，$\tau_1$时间步长上采样
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps):
            # 在列表$[\tau_1, \tau_2, \dots, \tau_S]$中的索引$i$
            index = len(time_steps) - i - 1
            # 时间步长$\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # 采样$x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step, index=index,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # 返回$x_0$
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        """
        ### 采样$x_{\tau_{i-1}}$

        :param x: 是形状为[batch_size, channels, height, width]的$x_{\tau_i}$
        :param c: 是形状为[batch_size, emb_size]的条件嵌入$c$
        :param t: 是形状为[batch_size]的$\tau_i$
        :param step: 是作为整数的步骤$\tau_i$
        :param index: 是列表$[\tau_1, \tau_2, \dots, \tau_S]$中的索引$i$
        :param repeat_noise: 指定批处理中所有样本的噪声是否应相同
        :param temperature: 是噪声温度（随机噪声乘以这个）
        :param uncond_scale: 是无条件引导比例$s$。这用于
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: 是空提示$c_u$的条件嵌入
        """

        # 获取$\epsilon_\theta(x_{\tau_i})$
        e_t = self.get_eps(x, t, c,
                           uncond_scale=uncond_scale,
                           uncond_cond=uncond_cond)

        # 计算$x_{\tau_{i - 1}}$和预测的$x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        #
        return x_prev, pred_x0, e_t


    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        """
        ### 在给定$\epsilon_\theta(x_{\tau_i})$的情况下采样$x_{\tau_{i-1}}$
        """

        # $\alpha_{\tau_i}$
        alpha = self.ddim_alpha[index]
        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.ddim_alpha_prev[index]
        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]
        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # 当前对$x_0$的预测，
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        # 指向$x_t$的方向
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # 当$\eta = 0$时不添加噪声
        if sigma == 0.:
            noise = 0.
        # 如果为批处理中的所有样本使用相同的噪声
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
            # 为每个样本添加不同的噪声
        else:
            noise = torch.randn(x.shape, device=x.device)

        # 乘以温度
        noise = noise * temperature

        #  \begin{align}
        #     x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
        #             \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
        #             \Bigg) \\
        #             &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
        #             &+ \sigma_{\tau_i} \epsilon_{\tau_i}
        #  \end{align}
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        #
        return x_prev, pred_x0


    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        ### 从$q_{\sigma,\tau}(x_{\tau_i}|x_0)$中采样

        $$q_{\sigma,\tau}(x_t|x_0) =
         \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$

        :param x0: 是形状为[batch_size, channels, height, width]的$x_0$
        :param index: 是时间步长$\tau_i$的索引$i$
        :param noise: 是噪声，$\epsilon$
        """

        # 如果未指定噪声，则生成随机噪声
        if noise is None:
            noise = torch.randn_like(x0)

        # 从
        #  $$q_{\sigma,\tau}(x_t|x_0) =
        #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise

    @torch.no_grad()
    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        ### 绘画循环

        :param x: 是形状为[batch_size, channels, height, width]的$x_{S'}$ 
        :param cond: 是条件嵌入$c$
        :param t_start: 是开始采样的步骤，$S'$
        :param orig: 是我们正在绘画的原始图像的潜在页面。如果未提供，则这将是图像到图像的转换。
        :param mask: 是保留原始图像的掩码。
        :param orig_noise: 是添加到原始图像的固定噪声。
        :param uncond_scale: 是无条件引导尺度$s$。这用于
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: 是空提示$c_u$的条件嵌入
        """
        # 获取批次大小
        bs = x.shape[0]

        # 采样时间步长为$\tau_{S`},\tau_{S' - 1},\ldots,\tau_1$
        time_steps = np.flip(self.time_steps[:t_start])

        for i, step in monit.enum('Paint', time_steps):
            # 列表中$\tau_1,\tau_2,\ldots,\tau_S$的索引$i$
            index = len(time_steps) - i - 1
            # 时间步长$\tau_i$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # 对$x_{\tau_{i-1}}$进行采样
            x, _, _ = self.p_sample(x, cond, ts, step, index=index,
                                    uncond_scale=uncond_scale,
                                    uncond_cond=uncond_cond)

            # 用原始图像替换掩码区域
            if orig is not None:
                # 获取原始图像在潜在空间中的$q_{\sigma,\tau}(x_{\tau_i}|x_0)$
                orig_t = self.q_sample(orig, index, noise=orig_noise)
                # 替换掩码区域
                x = orig_t * mask + x * (1 - mask)

        #
        return x