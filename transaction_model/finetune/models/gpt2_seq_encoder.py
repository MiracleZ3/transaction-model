"""GPT2 跨交易时序编码器（Route C）。

移植自 risk_control_2/models/gpt2.py 的 GPT2LikeEncode + FrequencyEncode，
默认维度对齐 Route A 的 Llama hidden=512。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import GPT2Model


class FrequencyEncode(nn.Module):
    """delta_time 的频率编码（与 risk_control_2 一致）。

    把相邻交易的小时差通过 2^k·π·t 的 sin/cos 多频展开，再线性投影到 out_dim。
    """

    def __init__(self, L: int = 8, out_dim: int = 16, padding: int = -1):
        super().__init__()
        self.L = L
        self.out_dim = out_dim
        self.in_dim = 2 * L
        self.padding = padding
        self.lin = nn.Identity() if out_dim == self.in_dim else nn.Linear(self.in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 1]
        x = x.unsqueeze(-1)
        mask = (x != self.padding).float()
        pi_x = x * math.pi
        freq = 2 ** torch.arange(self.L, device=x.device).float() * pi_x
        sin = torch.sin(freq)
        cos = torch.cos(freq)
        feat = torch.cat([sin, cos], dim=-1)
        out = self.lin(feat)
        return mask * out


class GPT2SeqEncoder(nn.Module):
    """GPT2 跨交易序列编码器。

    与 risk_control_2 的 GPT2LikeEncode 等价，只是默认维度对齐 Route A（512d）。

    前向：
        cls_emb : [B, T, H_in]     per-txn 表示（来自 Llama）
        delta_t : [B, T]           相邻交易时间差（小时）
        attn_mask : [B, T]         gpt2 的序列级 mask
        ↓
        last_hidden : [B, T, H_gpt2]
    """

    def __init__(
        self,
        bert_hidden_size: int = 512,
        gpt2_hidden_size: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 512,
        dt_dim: int = 16,
        pad_token_id: int = 0,
    ):
        super().__init__()
        from transformers import GPT2Config

        self.bert_hidden_size = bert_hidden_size
        self.gpt2_hidden_size = gpt2_hidden_size
        self.dt_dim = dt_dim
        self.pad_token_id = pad_token_id

        gpt2_cfg = GPT2Config(
            n_embd=gpt2_hidden_size,
            n_layer=n_layers,
            n_head=n_heads,
            n_positions=max_len,
            n_ctx=max_len,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.gpt2 = GPT2Model(gpt2_cfg)
        self.gpt2.wte = nn.Identity()  # 不用 token embedding，直接吃外部 embed

        self.pos_emb = nn.Embedding(max_len, bert_hidden_size)
        self.input_dim = bert_hidden_size + dt_dim if dt_dim is not None else bert_hidden_size
        self.freq = (
            nn.Identity()
            if dt_dim is None
            else FrequencyEncode(int(dt_dim // 2), dt_dim, padding=pad_token_id)
        )
        self.proj = nn.Linear(self.input_dim, gpt2_hidden_size)

    def forward(
        self,
        cls_emb: torch.Tensor,
        delta_t: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = cls_emb.shape
        device = cls_emb.device
        dtype = next(self.parameters()).dtype

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_emb(pos_ids).to(device, dtype=dtype)
        x = cls_emb + pos_emb

        if delta_t is not None:
            delta_t = delta_t.to(device)
            dt = self.freq(delta_t.float()).to(device, dtype=dtype)  # [B, T, dt_dim]
            x_expanded = torch.cat([x, dt], dim=-1)
        else:
            x_expanded = x

        x_expanded = x_expanded.to(dtype=dtype)
        x_proj = self.proj(x_expanded)

        # GPT2Model 的 attention_mask 语义：1=valid，0=pad。需先把 bool→int。
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.long()

        last_hidden = self.gpt2(
            inputs_embeds=x_proj,
            attention_mask=attention_mask,
        ).last_hidden_state
        return last_hidden
