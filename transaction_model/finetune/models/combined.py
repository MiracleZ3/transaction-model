"""Route C CombinedModel：Llama (per-txn) → GPT2 (cross-txn) → 分类头。

与 risk_control_2::CombinedModel 接口一致，去掉 MLM 路径（Route A 的预训 token
流内部不含 [MASK]，下游训练只需分类信号）。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .llama_encoder import LlamaEncoder
from .gpt2_seq_encoder import GPT2SeqEncoder
from .classifier import ClassifierHead


class CombinedModel(nn.Module):
    """Llama + GPT2 + Classifier 三段式下游模型。

    前向输入 `info` 是 collate 产出的 dict，至少含：
      - input_ids        : [B, T, L]    每笔交易内部的 token id 序列
      - attention_mask   : [B, T, L]    每笔交易内部的有效 mask（1=有效）
      - gpt2_attention_mask : [B, T]    序列级 mask（每用户真实步数）
      - delta_time_stap  : [B, T]       相邻交易小时差
      - lens_in          : [B]          每用户真实步数
    """

    def __init__(
        self,
        llama: LlamaEncoder,
        gpt2: GPT2SeqEncoder,
        classifier: ClassifierHead,
    ):
        super().__init__()
        self.llama = llama
        self.gpt2 = gpt2
        self.classifier = classifier

    def forward(self, info: Dict) -> torch.Tensor:
        input_ids = info["input_ids"]
        attention_mask = info["attention_mask"]
        gpt2_attention_mask = info.get("gpt2_attention_mask")
        delta_time = info.get("delta_time_stap")
        lens_in = info.get("lens_in")

        # 1. per-txn Llama
        per_txn = self.llama(input_ids, attention_mask)  # [B, T, H]
        B, T, _ = per_txn.shape

        # dtype 对齐（Llama + LoRA 可能 fp16）
        mydtype = next(self.gpt2.parameters()).dtype
        per_txn = per_txn.to(dtype=mydtype)
        if delta_time is not None:
            delta_time = delta_time.to(dtype=mydtype)
        if gpt2_attention_mask is not None:
            gpt2_attention_mask = gpt2_attention_mask.to(dtype=mydtype)

        # 2. GPT2 跨交易
        gpt2_enc = self.gpt2(per_txn, delta_time, gpt2_attention_mask)

        # 3. 分类头
        if lens_in is None:
            lens_in = torch.full(
                (B,), T, device=gpt2_enc.device, dtype=torch.long
            )
        logits = self.classifier(gpt2_enc, lens_in)
        return logits
