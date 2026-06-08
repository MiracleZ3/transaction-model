"""下游分类头（Route C）。

直接移植 risk_control_2/models/decode.py::ClassifierDXZP_1，去掉 core.registry。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """二分类 head，吃 GPT2 输出 [B, T, H]，输出 [B, num_class]。

    cls : "cls" 取最后一笔有效交易的 hidden；
          "pool" 平均所有有效交易的 hidden。
    lens_in : [B] 每个用户真实序列长度。
    """

    def __init__(
        self,
        gpt2_hidden_size: int,
        num_class: int = 2,
        mlp_hidden_size: Optional[int] = None,
        cls: str = "cls",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = (
            mlp_hidden_size if mlp_hidden_size is not None else 2 * gpt2_hidden_size
        )
        self.out = nn.Sequential(
            nn.LayerNorm(gpt2_hidden_size),
            nn.Linear(gpt2_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_class),
        )
        self.cls = cls

    def forward(
        self,
        enc: torch.Tensor,            # [B, T, H]
        lens_in: torch.Tensor,         # [B]
        **kwargs,
    ) -> torch.Tensor:
        B, T, _ = enc.shape
        device = enc.device
        lens_in = lens_in.to(device).long()

        if self.cls == "cls":
            last_step = (lens_in - 1).clamp(0, T - 1)
            batch_idx = torch.arange(B, device=device)
            last_enc = enc[batch_idx, last_step]
        elif self.cls == "pool":
            mask = torch.arange(T, device=device).unsqueeze(0) < lens_in.unsqueeze(1)
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            enc = enc * mask
            eps = 1e-7
            last_enc = enc.sum(dim=1) / (lens_in.unsqueeze(1) + eps)
        else:
            raise ValueError(f"Unknown cls mode: {self.cls}")
        return self.out(last_enc)
