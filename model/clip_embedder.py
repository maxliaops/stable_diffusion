"""
---
title: CLIP 文本嵌入器
summary: >
 CLIP 嵌入器，用于获取稳定扩散的提示嵌入
---

# CLIP 文本嵌入器

这用于为[稳定扩散](../index.html)获取提示嵌入。
它使用 HuggingFace Transformers CLIP 模型。
"""

from typing import List

from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP 文本嵌入器
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
        """
        :param version: 是模型版本
        :param device: 是设备
        :param max_length: 是标记化提示的最大长度
        """
        # 继承 nn.Module
        super().__init__()
        # 加载 tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # 加载 CLIP 转换器
        self.transformer = CLIPTextModel.from_pretrained(version).eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: 是要嵌入的提示列表
        """
        # 标记化提示
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # 获取令牌 id
        tokens = batch_encoding["input_ids"].to(self.device)
        # 获取 CLIP 嵌入
        return self.transformer(input_ids=tokens).last_hidden_state