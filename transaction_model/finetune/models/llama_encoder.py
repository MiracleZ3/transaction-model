"""Llama per-transaction encoder (Route C).

替代 risk_control_2 的 BertLikeEncode：
  - 输入是 [B, T, L] 的 token id（每笔交易 L 个 token）
  - reshape 成 [B*T, L] 送进 Llama transformer 体（取 .model，不要 lm_head）
  - 对每笔交易池化（last-token / mean / cls）成 [B*T, hidden]
  - reshape 回 [B, T, hidden] 返回

支持 LoRA 微调（在 query/value 上注入 peft adapter）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LlamaEncoder(nn.Module):
    """Per-transaction Llama encoder.

    Parameters
    ----------
    model_path : str | Path
        route A 训练保存的 HF Llama 目录（含 config.json + safetensors）。若为 None，
        必须传 llama_config 用于从零构造（仅 toy test 用）。
    hidden_size : int
        Llama 的 hidden_size。route A 默认 512。
    pool_mode : {"last_token", "mean", "cls"}
        per-txn 池化策略：
          - "last_token": 取非 PAD 的最后一位 hidden（NV-Embed 风格）
          - "mean":       非 PAD token 的 hidden 平均
          - "cls":        取 sequence 起始一个 token（route A 没有 [CLS]，会把 <bos> 当 cls）
    freeze : bool
        True → 冻结 Llama 全部参数（仍可被 LoRA 改写，由 lora_cfg 控制）
    lora_cfg : dict, optional
        peft LoraConfig 参数（r/alpha/target_modules/dropout/bias）。None 表示不注入 LoRA。
        默认 target_modules=["q_proj","v_proj"]。
    pad_token_id : int
        用于 attention_mask 推断（Llama 不需要 pad，但下游 [B,T,L] collate 可能.pad）
    add_bos_pool_token : bool
        当前默认 False：route A 的前置 token 是 <bos>，但每笔交易内也可能含 <bos>，
        所以 "last_token" 是更稳的选择；若你的 collate 在每笔交易前加了 [CLS] token，
        可改成 "cls" + add_bos_pool_token=True。
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        hidden_size: int = 512,
        pool_mode: str = "last_token",
        freeze: bool = True,
        lora_cfg: Optional[dict] = None,
        pad_token_id: int = 0,
        llama_config: Optional[object] = None,
    ):
        super().__init__()
        from transformers import LlamaModel

        self.hidden_size = hidden_size
        self.pool_mode = pool_mode
        self.pad_token_id = pad_token_id

        if model_path is not None:
            from transaction_model.checkpoints import resolve_hf_model_dir
            try:
                hf_dir = resolve_hf_model_dir(model_path)
                if hf_dir != Path(model_path):
                    logger.info(
                        f"model_path {model_path} 不是 HF 目录，自动定位到 {hf_dir}"
                    )
                model_path = hf_dir
            except FileNotFoundError as e:
                # 让 transformers 的 from_pretrained 报更清晰的错——但leep the 消息。
                logger.error(str(e))
                raise
            logger.info(f"Loading Llama from {model_path}")
            self.llama = LlamaModel.from_pretrained(str(model_path))
            # 校验维度
            actual_h = self.llama.config.hidden_size
            if actual_h != hidden_size:
                logger.warning(
                    f"Llama hidden_size={actual_h} != configured {hidden_size}; "
                    f"using actual={actual_h}"
                )
                self.hidden_size = actual_h
        elif llama_config is not None:
            logger.info(f"Building Llama from config (toy/test mode)")
            self.llama = LlamaModel(llama_config)
        else:
            raise ValueError("Either model_path or llama_config must be provided")

        # 可选：注入 LoRA
        self.lora_applied = False
        if lora_cfg is not None:
            self._inject_lora(lora_cfg)
        elif freeze:
            logger.info("Freezing Llama (no LoRA). requires_grad=False for all params.")
            for p in self.llama.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def _inject_lora(self, lora_cfg: dict) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "Route C 的 LoRA 微调需要安装 peft: pip install peft"
            ) from e

        cfg = {
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj"],
            **lora_cfg,
        }
        logger.info(f"Injecting LoRA into Llama: {cfg}")
        # 关键：LlamaModel 不是 ForCausalLM，所以 can't use TaskType.SEQ_CLS
        # target_modules 直接打到 LlamaModel 的 attn 上即可
        try:
            self.llama = get_peft_model(self.llama, LoraConfig(**cfg))
            self.lora_applied = True
        except Exception as e:
            # 失败回退：尝试 all-linear
            logger.warning(
                f"LoRA injection with target_modules={cfg['target_modules']} "
                f"failed: {e}. Falling back to all-linear."
            )
            cfg["target_modules"] = "all-linear"
            self.llama = get_peft_model(self.llama, LoraConfig(**cfg))
            self.lora_applied = True

        # 打印可训参数占比
        trainable, total = self._count_trainable()
        logger.info(
            f"LoRA injected. Trainable params: {trainable:,} / {total:,} "
            f"({100*trainable/max(total,1):.3f}%)"
        )

    def _count_trainable(self) -> Tuple[int, int]:
        trainable, total = 0, 0
        for p in self.llama.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return trainable, total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T, L] 或 [B', L]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回 [B, T, hidden] 的 per-txn 表示。

        若输入是 3D，自动展开 B*T 维度，调用一维前向，再 reshape 回 [B, T, hidden]。
        """
        is_3d = input_ids.dim() == 3
        if is_3d:
            B, T, L = input_ids.shape
            input_ids = input_ids.view(B * T, L)
            if attention_mask is not None:
                attention_mask = attention_mask.view(B * T, L)

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = out.last_hidden_state  # [N, L, H]

        per_txn = self._pool(last_hidden, attention_mask)  # [N, H]

        if is_3d:
            per_txn = per_txn.view(B, T, -1)
        return per_txn

    def _pool(
        self,
        last_hidden: torch.Tensor,    # [N, L, H]
        attention_mask: torch.Tensor, # [N, L]
    ) -> torch.Tensor:
        H = last_hidden.size(-1)
        if self.pool_mode == "last_token":
            # 取每个样本非 PAD 最后一位
            seq_lens = attention_mask.sum(dim=1).clamp(min=1) - 1  # [N]
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            return last_hidden[batch_idx, seq_lens, :]
        if self.pool_mode == "mean":
            mask = attention_mask.unsqueeze(-1).float()  # [N, L, 1]
            summed = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-5)
            return summed / count
        if self.pool_mode == "cls":
            return last_hidden[:, 0, :]
        raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
