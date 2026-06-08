"""Route C 轻量 Trainer（pure PyTorch + AdamW + cosine + LoRA）。

不做 DeepSpeed / FSDP（先单机 DDP 起步，DS 留作后续）。

调用约定：
    trainer = Trainer(model, loss_fn, optimizer, scheduler, scaler, cfg, args)
    trainer.train(train_loader, val_loader)

Checkpoint 格式与 risk_control_2 一致：torch.save 一个 dict 含 step/model/opt/...
LoRA adapter 单独用 peft.save_pretrained 保存到一个子目录。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import build_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Route C 训练参数（从 JSON config 派生）。

    字段对应 configs/routec/default.json 的 step_scheduler / strategy / optimizer。
    """
    max_steps: int = 10000
    grad_accum_steps: int = 1
    val_every_steps: int = 50
    ckpt_every_steps: int = 200
    keep_last_ckpts: int = 5
    warmup_steps: int = 200
    min_lr: float = 1e-6
    init_lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    use_amp: bool = True
    log_every_steps: int = 20
    seed: int = 42
    save_dir: str = "models/routec"
    log_dir: str = "log/routec"


class Trainer:
    """单机 DDP 友好的下游微调训练器。

    Parameters
    ----------
    model : CombinedModel
    cfg : TrainConfig
    loss_name : str
        loss registry 名（见 finetune.losses.LOSS_REGISTRY）
    loss_params : dict
        透传给 build_loss 的 kwargs
    device : str | torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        loss_name: str,
        loss_params: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.model = model.to(self.device)
        self.loss_fn = build_loss(loss_name, **(loss_params or {}))
        self.loss_name = loss_name

        # 只训 requires_grad=True 的参数
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Trainable params: {n_train:,} / {n_total:,} "
            f"({100*n_train/max(n_total,1):.3f}%)"
        )

        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.init_lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = self._build_scheduler(cfg)
        # torch 2.x 推荐 torch.amp.GradScaler('cuda')；老式 torch.cuda.amp.GradScaler 兼容回退
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=cfg.use_amp and self.device.type == "cuda"
            )
        else:  # pragma: no cover - very old torch
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=cfg.use_amp and self.device.type == "cuda"
            )

        self.global_step = 0
        self._save_dir = Path(cfg.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = Path(cfg.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _build_scheduler(self, cfg: TrainConfig):
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            min_ratio = cfg.min_lr / max(cfg.init_lr, 1e-12)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Train / Eval
    # ------------------------------------------------------------------

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        stats = {"loss": [], "lr": []}

        while self.global_step < cfg.max_steps:
            for batch in train_loader:
                if self.global_step >= cfg.max_steps:
                    break

                batch = self._to_device(batch)
                # torch 2.x: torch.amp.autocast('cuda') 兼容老版本 torch.cuda.amp.autocast
                if hasattr(torch.amp, "autocast"):
                    ctx = torch.amp.autocast("cuda", enabled=self.scaler.is_enabled())
                else:  # pragma: no cover
                    ctx = torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())
                with ctx:
                    logits = self.model(batch)
                    loss = self.loss_fn(
                        logits, batch["label"], batch.get("amount")
                    ) / cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                if (self.global_step + 1) % cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad), 1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                self.global_step += 1
                stats["loss"].append(float(loss.detach()) * cfg.grad_accum_steps)
                stats["lr"].append(self.scheduler.get_last_lr()[0])

                if self.global_step % cfg.log_every_steps == 0:
                    avg_loss = np.mean(stats["loss"][-cfg.log_every_steps:])
                    lr = stats["lr"][-1]
                    elapsed = time.time() - t0
                    logger.info(
                        f"[step {self.global_step:>6d}] loss={avg_loss:.4f} "
                        f"lr={lr:.2e} elapsed={elapsed:.0f}s"
                    )
                    self._dump_log(stats, prefix="train")

                if cfg.val_every_steps > 0 and self.global_step % cfg.val_every_steps == 0:
                    if val_loader is not None:
                        self.evaluate(val_loader)

                if cfg.ckpt_every_steps > 0 and self.global_step % cfg.ckpt_every_steps == 0:
                    self.save_ckpt()

        # 训练结束兜底
        self.save_ckpt(tag="final")
        logger.info(f"Training done. global_step={self.global_step}")

    def evaluate(self, val_loader: DataLoader) -> dict:
        cfg = self.cfg
        self.model.eval()
        all_probs, all_labels, all_users, all_amounts = [], [], [], []
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                if hasattr(torch.amp, "autocast"):
                    ctx = torch.amp.autocast("cuda", enabled=self.scaler.is_enabled())
                else:  # pragma: no cover
                    ctx = torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())
                with ctx:
                    logits = self.model(batch)
                    loss = self.loss_fn(logits, batch["label"], batch.get("amount"))
                total_loss += float(loss.detach())
                n_batches += 1

                probs = torch.softmax(logits.float(), dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch["label"].cpu().numpy())
                all_amounts.append(batch["amount"].cpu().numpy())
                all_users.extend(batch["user"])

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        preds = (probs[:, 1] >= 0.5).astype(int)
        amounts = np.concatenate(all_amounts)

        # 基础指标
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, average_precision_score,
        )
        metrics = {
            "val_step": self.global_step,
            "avg_loss": total_loss / max(1, n_batches),
            "acc": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
        except ValueError:
            metrics["auc"] = 0.5
        try:
            metrics["ap"] = float(average_precision_score(labels, probs[:, 1]))
        except ValueError:
            metrics["ap"] = 0.0

        logger.info(
            f"[val {self.global_step}] loss={metrics['avg_loss']:.4f} "
            f"acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f} ap={metrics['ap']:.4f}"
        )

        # val_prob.json
        prob_path = self._log_dir / f"{self.global_step}_val_prob.json"
        prob_records = {
            u: {
                "pred": int(p),
                "p0": float(probs[i, 0]),
                "p1": float(probs[i, 1]),
                "pred_prob": float(probs[i, 1]),
                "label": int(labels[i]),
                "amount": float(amounts[i]),
            }
            for i, (u, p) in enumerate(zip(all_users, preds))
        }
        with open(prob_path, "w") as f:
            json.dump(prob_records, f, indent=2)
        with open(self._log_dir / f"{self.global_step}_val.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_ckpt(self, tag: Optional[str] = None) -> None:
        step = self.global_step
        suffix = tag or f"step{step}"
        ckpt_path = self._save_dir / f"ckpt_{suffix}.pt"
        state = {
            "step": step,
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "opt": self.optimizer.state_dict() if not tag else None,
            "scheduler": self.scheduler.state_dict() if not tag else None,
            "scaler": self.scaler.state_dict() if not tag else None,
            "loss_name": self.loss_name,
        }
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint → {ckpt_path}")

        # LoRA adapter 单独存（便于部署期把 adapter 加载到不同 base 模型）
        try:
            from peft import PeftModel
            if hasattr(self.model, "llama") and isinstance(self.model.llama.llama, PeftModel):
                adapter_dir = self._save_dir / f"adapter_{suffix}"
                self.model.llama.llama.save_pretrained(str(adapter_dir))
                logger.info(f"Saved LoRA adapter → {adapter_dir}")
        except Exception as e:
            logger.warning(f"Failed to save LoRA adapter: {e}")

        # 保留最近 N 份
        self._prune_old_ckpts()

    def _prune_old_ckpts(self) -> None:
        ckpts = sorted(self._save_dir.glob("ckpt_step*.pt"))
        n = len(ckpts) - self.cfg.keep_last_ckpts
        if n <= 0:
            return
        for path in ckpts[:n]:
            try:
                path.unlink()
                logger.info(f"Pruned old checkpoint: {path.name}")
            except OSError:
                pass

    def load_ckpt(self, ckpt_path: Path, load_opt: bool = True) -> None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=False)
        self.global_step = state.get("step", 0)
        if load_opt and state.get("opt") is not None:
            try:
                self.optimizer.load_state_dict(state["opt"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.scaler.load_state_dict(state["scaler"])
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to restore optimizer state: {e}")
        logger.info(f"Loaded checkpoint: {ckpt_path} (step={self.global_step})")

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _dump_log(self, stats: dict, prefix: str = "train") -> None:
        path = self._log_dir / f"{prefix}_stats.jsonl"
        with open(path, "a") as f:
            for i in range(max(0, len(stats["loss"]) - 1), len(stats["loss"])):
                f.write(json.dumps({
                    "step": self.global_step - len(stats["loss"]) + i + 1,
                    "loss": stats["loss"][i],
                    "lr": stats["lr"][i],
                }) + "\n")
