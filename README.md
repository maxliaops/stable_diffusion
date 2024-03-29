# stable_diffusion
基于扩散的文本到图像生成模型


稳定扩散是一种潜在的文本到图像扩散模型，能够根据任何文本输入生成逼真的照片。

**Stable-Diffusion-v-1-4** 检查点是使用[Stable-Diffusion-v-1-2](https://steps/huggingface.co/CompVis/stable-diffusion-v-1-2-original) 
检查点的权重进行初始化的，并在“laion-aesthetics v2 5+”上以 512x512 的分辨率进行了 225k 步的微调，并且丢弃了10%的文本条件，以改进[无分类器引导采样](https://arxiv.org/abs/2207.12598)。

#### 下载权重
- [sd-v1-4.ckpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt)
- [sd-v1-4-full-ema.ckpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt)

这些权重旨在与原始的[CompVis 稳定扩散代码库](https://github.com/CompVis/stable-diffusion)一起使用。如果您正在寻找与 Diffusers 库一起使用的模型，请[点击这里](https://huggingface.co/CompVis/stable-diffusion-v1-4)。

## 模型详情
- **开发者**：Robin Rombach, Patrick Esser
- **模型类型**：基于扩散的文本到图像生成模型
- **语言**：英语
- **许可证**：[创意 ml OpenRAIL M 许可证](https://huggingface.co/spaces/CompVis/stable-diffusion-license)是一种[开放 RAIL M 许可证](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses)，改编自[BigScience](https://bigscience.huggingface.co/)和[RAIL 倡议](https://www.licenses.ai/)在负责任的人工智能许可领域共同开展的工作。另见关于我们许可证所基于的[BLOOM 开放 RAIL 许可证的文章](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)。
- **模型描述**：这是一个可以根据文本提示生成和修改图像的模型。它是一种[潜在扩散模型](https://arxiv.org/abs/2112.10752)，使用固定的、预训练的文本编码器([CLIP ViT-L/14](https://arxiv.org/abs/2103.00020))，如[Imagen 论文](https://arxiv.org/abs/2205.11487)中所建议的。
- **更多信息资源**：[GitHub 存储库](https://github.com/CompVis/stable-diffusion)，[论文](https://arxiv.org/abs/2112.10752)。
- **引用方式**：

      @InProceedings{Rombach_2022_CVPR,
          author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
          title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
          month     = {June},
          year      = {2022},
          pages     = {10684-10695}
      }

