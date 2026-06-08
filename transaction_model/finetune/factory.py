"""Route C Trainer 工厂：从 RouteCConfig 装配模型 + 数据 + 训练器。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from transaction_model.finetune.config import RouteCConfig
from transaction_model.finetune.models import (
    ClassifierHead, CombinedModel, GPT2SeqEncoder, LlamaEncoder,
)
from transaction_model.finetune.data import SftNDJsonDataset, prepare_collate
from transaction_model.finetune.trainer import Trainer, TrainConfig
from transaction_model.tokenizer import YLTabularTokenizer

logger = logging.getLogger(__name__)


def build_tokenizer(cfg) -> YLTabularTokenizer:
    """从 cfg.paths.tokenizer_state 加载已 fit 的 YLTabularTokenizer。"""
    state_path = cfg.paths.get("tokenizer_state")
    if not state_path:
        raise ValueError(
            "configs/routec/*.json 缺 paths.tokenizer_state；"
            "请先跑 step_02_tokenize_ndjson.py 生成 yl_tokenizer.json"
        )
    return YLTabularTokenizer.from_file(state_path)


def build_model(cfg: RouteCConfig) -> CombinedModel:
    """按 cfg 装配 Llama + GPT2 + Classifier。

    task_type 决定 LoRA 注入：
      - "lora"         注入 LoRA，原 Llama 冻结
      - "freeze"       不注入 LoRA，Llama 完全冻结
      - "all_params"   不注入 LoRA，Llama 全训
    """
    llama_cfg = cfg.llama or {}
    lora_cfg = cfg.lora if cfg.task_type == "lora" else None
    freeze = cfg.task_type != "all_params"

    llama = LlamaEncoder(
        model_path=llama_cfg.get("model_path"),
        hidden_size=llama_cfg.get("hidden_size", 512),
        pool_mode=llama_cfg.get("pool_mode", "last_token"),
        freeze=freeze,
        lora_cfg=lora_cfg,
        pad_token_id=llama_cfg.get("pad_token_id", 0),
    )

    gpt2_cfg = cfg.gpt2 or {}
    gpt2 = GPT2SeqEncoder(
        bert_hidden_size=gpt2_cfg.get("bert_hidden_size", llama.hidden_size),
        gpt2_hidden_size=gpt2_cfg.get("gpt2_hidden_size", 512),
        n_layers=gpt2_cfg.get("n_layers", 6),
        n_heads=gpt2_cfg.get("n_heads", 8),
        max_len=gpt2_cfg.get("max_len", 512),
        dt_dim=gpt2_cfg.get("dt_dim", 16),
        pad_token_id=gpt2_cfg.get("pad_token_id", 0),
    )

    dec_cfg = cfg.decode or {}
    classifier = ClassifierHead(
        gpt2_hidden_size=gpt2_cfg.get("gpt2_hidden_size", 512),
        num_class=dec_cfg.get("num_class", 2),
        cls=dec_cfg.get("cls", "cls"),
        dropout=dec_cfg.get("dropout", 0.2),
    )

    return CombinedModel(llama=llama, gpt2=gpt2, classifier=classifier)


def build_datasets(
    cfg: RouteCConfig,
    tokenizer: YLTabularTokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """构造 train/val DataLoader。

    `data_config.folder` 既可以是单个 NDJSON 也可以是目录（取所有 .jsonl）。
    val 暂时复用同一份数据（冒烟测试用）；正式训练时分离 val_folder。
    """
    dc = cfg.data_config or {}
    pipeline = tokenizer._pipeline
    vocab = tokenizer.vocab
    pad = tokenizer.pad_token_id
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    sep = tokenizer.sep_token_id

    train_folder = Path(dc.get("folder", "data/yl/raw"))
    val_folder = Path(dc.get("val_folder", train_folder))

    def find_jsonls(d: Path):
        if d.is_file():
            return [d]
        return sorted(d.glob("*.jsonl"))

    def build_loader(folder: Path, name: str) -> DataLoader:
        files = find_jsonls(folder)
        if not files:
            raise FileNotFoundError(f"No .jsonl under {folder} for {name}")
        # 合并多个 NDJSON 文件到一个 Dataset
        ds = None
        for p in files:
            sub = SftNDJsonDataset(
                ndjson_path=p,
                pipeline=pipeline,
                vocab=vocab,
                pad_token_id=pad, bos_token_id=bos,
                eos_token_id=eos, sep_token_id=sep,
                max_txn_len=dc.get("max_txn_len", 32),
                minleng=dc.get("minleng", 1),
                max_hiswindow=dc.get("hiswindow", 512),
            )
            ds = sub if ds is None else _ConcatDatasetLike([ds, sub])
        return DataLoader(
            ds,
            batch_size=dc.get("batch_size", 12),
            shuffle=(name == "train"),
            collate_fn=prepare_collate(pad, dc.get("hiswindow", 512)),
            num_workers=0,
        )

    train_loader = build_loader(train_folder, "train")
    val_loader = build_loader(val_folder, "val")
    return train_loader, val_loader


class _ConcatDatasetLike(torch.utils.data.Dataset):
    """极简 concat，避免运维 import torch.utils.data.ConcatDataset 时的 side-effect。"""

    def __init__(self, datasets):
        from torch.utils.data import ConcatDataset
        self._inner = ConcatDataset(datasets)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, i):
        return self._inner[i]


def build_trainer(
    cfg: RouteCConfig, model: CombinedModel, device: str = "cuda",
) -> Trainer:
    """从 cfg 装配 Trainer。loss_fn/optimizer/scheduler 都在 Trainer 内部初始化。"""
    ss = cfg.step_scheduler or {}
    paths = cfg.paths or {}
    tc = TrainConfig(
        max_steps=ss.get("max_steps", 5000),
        grad_accum_steps=ss.get("grad_accum_steps", 1),
        val_every_steps=ss.get("val_every_steps", 200),
        ckpt_every_steps=ss.get("ckpt_every_steps", 500),
        keep_last_ckpts=ss.get("keep_last_ckpts", 5),
        warmup_steps=ss.get("warmup_steps", 200),
        min_lr=ss.get("min_lr", 1e-6),
        init_lr=ss.get("init_lr", 1e-4),
        weight_decay=ss.get("weight_decay", 0.01),
        betas=tuple(ss.get("betas", [0.9, 0.999])),
        use_amp=ss.get("use_amp", True),
        log_every_steps=ss.get("log_every_steps", 20),
        seed=ss.get("seed", 42),
        save_dir=paths.get("save_ckpt", "models/routec"),
        log_dir=paths.get("logging_path", "log/routec"),
    )
    loss_cfg = cfg.loss_fn or {"name": "sft_focal_loss_with_amount"}
    return Trainer(
        model=model,
        cfg=tc,
        loss_name=loss_cfg["name"],
        loss_params=loss_cfg.get("params", {}),
        device=device,
    )
