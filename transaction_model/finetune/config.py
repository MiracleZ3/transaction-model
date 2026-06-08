"""Route C 配置加载器。

直接读 JSON（与 risk_control_2/configs/default.json 风格兼容，不复用 Hydra）。

Config 字段：
    train_name, task_type,
    llama : { model_path, hidden_size, pool_mode, freeze }
    lora  : { r, alpha, dropout, bias, target_modules, task_type }
    gpt2  : { bert_hidden_size, gpt2_hidden_size, n_layers, n_heads, max_len, dt_dim }
    decode: { num_class, cls, dropout }
    data_config : { folder, val_folder, batch_size, max_txn_len, minleng, hiswindow }
    loss_fn : { name, params }
    step_scheduler : { max_steps, grad_accum_steps, val_every_steps, ckpt_every_steps,
                       warmup_steps, init_lr, min_lr, weight_decay, betas, use_amp }
    paths : { save_ckpt, logging_path }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RouteCConfig:
    """Parsed Route C config."""
    raw: Dict[str, Any] = field(default_factory=dict)
    train_name: str = "routec"
    task_type: str = "lora"             # lora | freeze | all_params
    llama: Dict[str, Any] = field(default_factory=dict)
    lora: Optional[Dict[str, Any]] = None
    gpt2: Dict[str, Any] = field(default_factory=dict)
    decode: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    loss_fn: Dict[str, Any] = field(default_factory=dict)
    step_scheduler: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> RouteCConfig:
    """加载 JSON config。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Route C config not found: {path}")
    with open(path) as f:
        raw = json.load(f)

    return RouteCConfig(
        raw=raw,
        train_name=raw.get("train_name", "routec"),
        task_type=raw.get("task_type", "lora"),
        llama=raw.get("llama", {}),
        lora=raw.get("lora"),
        gpt2=raw.get("gpt2", {}),
        decode=raw.get("decode", {}),
        data_config=raw.get("data_config", {}),
        loss_fn=raw.get("loss_fn", {}),
        step_scheduler=raw.get("step_scheduler", {}),
        paths=raw.get("paths", {}),
    )
