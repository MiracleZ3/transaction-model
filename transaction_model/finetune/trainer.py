"""Route C Trainer（pure PyTorch + AdamW + cosine + LoRA + DDP）。

调用约定：
    trainer = Trainer(model, cfg, loss_name, ..., rank, world_size, local_rank)
    trainer.train(train_loader, val_loader)

DDP 行为（向后兼容单卡）：
  - 构造时若 world_size > 1，用 torch.nn.parallel.DistributedDataParallel 包装 self.model
  - 梯度累积：routec grad_accum-1 个 micro-step 用 self.model.no_sync() 包 backward，
    第 grad_accum 个 micro-step 才触发 allreduce（与 risk_control_2 trainer 462-470 一致）
  - checkpoint / log 文件：仅 rank 0 写
  - load_ckpt：所有 rank 同时 load（同 ckpt）→ 各自走 DDP 同步

Checkpoint 格式与 risk_control_2 一致：torch.save dict {step, model, opt, scheduler, scaler, ...}
LoRA adapter 单独用 peft.save_pretrained 保存到子目录。
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .distributed import (
    barrier,
    gather_predictions,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_tensor,
)
from .losses import build_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Route C 训练参数（从 JSON config 派生）。

    字段对应 configs/routec/default.json 的 step_scheduler / strategy / optimizer。
    多机多卡新增字段：
      find_unused_parameters   DDP 是否容忍 forward 没用到的参数。
                               LoRA 模式建议 True（adapter 没覆盖所有模块）。
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
    amp_dtype: str = "bf16"   # bf16（A800/H100 推荐，无需 scaler） | fp16（需 GradScaler）
    log_every_steps: int = 20
    seed: int = 42
    save_dir: str = "models/routec"
    log_dir: str = "log/routec"
    find_unused_parameters: bool = True   # DDP 兼容 LoRA / freeze


class Trainer:
    """下游微调训练器，支持单卡 + DDP。

    Parameters
    ----------
    model : CombinedModel
    cfg : TrainConfig
    loss_name : str
        loss registry 名（见 finetune.losses.LOSS_REGISTRY）
    loss_params : dict, optional
        透传给 build_loss 的 kwargs
    device : str | torch.device
        训练设备。单卡模式直接用；DDP 模式只用于 "_to_device"。device_ids 由
        local_rank 决定。
    rank, world_size, local_rank : int
        分布式参数。单卡时传 (0, 1, 0)。setup_distributed() 的返回值即可。
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        loss_name: str,
        loss_params: Optional[dict] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.loss_name = loss_name
        self.loss_fn = build_loss(loss_name, **(loss_params or {}))

        # 设备定位：DDP 时每个 rank 用自己的 GPU；单卡退化 device='cuda' / 'cpu'
        if torch.cuda.is_available() and world_size > 1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)

        # 只训 requires_grad=True 的参数（LoRA 注入后大部分 Llama 已 frozen）
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in self.model.parameters())
        if is_main_process():
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
        # 混合精度：bf16 不需要 GradScaler（A800/H100 原生支持），fp16 才需要。
        # 旧实现无条件建 scaler + autocast 默认 fp16，在 focal/pAUC loss 上易溢出 NaN。
        amp_enabled = cfg.use_amp and self.device.type == "cuda"
        use_fp16 = amp_enabled and cfg.amp_dtype == "fp16"
        self._amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
        if use_fp16:
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:  # pragma: no cover - very old torch
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            # bf16 或未开 AMP：scer disabled，scale/update/unscale_ 均为 no-op
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=False)
            else:  # pragma: no cover
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # DDP 包装：必须在 optimizer 构造之后做，否则会对未经 forward 的参数报错。
        # 调用方传 world_size>1 时才 wrap。
        if world_size > 1 and torch.cuda.is_available():
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=cfg.find_unused_parameters,
            )
            if is_main_process():
                logger.info(
                    f"DDP wrapped: rank={rank}/{world_size} local_rank={local_rank} "
                    f"find_unused_parameters={cfg.find_unused_parameters}"
                )

        self.global_step = 0
        # 仅主 rank 创建输出目录，避免并发 mkdir 撞车
        if is_main_process():
            self._save_dir = Path(cfg.save_dir)
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir = Path(cfg.log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = Path(cfg.save_dir)
            self._log_dir = Path(cfg.log_dir)
        barrier()  # 确保 rank 0 目录建好其它 rank 再继续

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _model_module(self) -> nn.Module:
        """解包 DDP，返回底层 CombinedModel。"""
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _amp_ctx(self):
        # autocast 是否启用跟 use_amp 绑定；dtype 用 self._amp_dtype（默认 bf16）。
        # 不能用 scaler.is_enabled() 当开关：bf16 时 scaler disabled 但 autocast 仍要开。
        enabled = self.cfg.use_amp and self.device.type == "cuda"
        if hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=self._amp_dtype, enabled=enabled)
        return torch.cuda.amp.autocast(enabled=enabled)  # pragma: no cover

    def _build_scheduler(self, cfg: TrainConfig):
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            denom = max(1, cfg.max_steps - cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / denom
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
                with self._amp_ctx():
                    logits = self.model(batch)
                    loss = self.loss_fn(
                        logits, batch["label"], batch.get("amount")
                    ) / cfg.grad_accum_steps

                # 梯度累积：accum-1 步走 no_sync，第 accum 步才真正 allreduce
                # 参考 risk_control_2/trainer.py 第 462-470 行逻辑
                is_accum = (self.global_step + 1) % cfg.grad_accum_steps != 0
                no_sync_ctx = (
                    self.model.no_sync()
                    if is_accum and hasattr(self.model, "no_sync")
                    else nullcontext()
                )
                with no_sync_ctx:
                    self.scaler.scale(loss).backward()

                if not is_accum:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad), 1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                self.global_step += 1
                # log loss 同步所有 rank 平均（DDP 模式下各 rank 看到不同 batch，
                # 自己的 loss 数值会偏差）；单卡时 reduce_tensor 返回原值。
                reduced = float(reduce_tensor(loss.detach()).item())
                stats["loss"].append(reduced * cfg.grad_accum_steps)
                stats["lr"].append(self.scheduler.get_last_lr()[0])

                if self.global_step % cfg.log_every_steps == 0 and is_main_process():
                    avg_loss = np.mean(stats["loss"][-cfg.log_every_steps:])
                    lr = stats["lr"][-1]
                    elapsed = time.time() - t0
                    logger.info(
                        f"[rank {self.rank} step {self.global_step:>6d}] "
                        f"loss={avg_loss:.4f} lr={lr:.2e} elapsed={elapsed:.0f}s"
                    )
                    self._dump_log(stats, prefix="train")

                if cfg.val_every_steps > 0 and self.global_step % cfg.val_every_steps == 0:
                    if val_loader is not None:
                        self.evaluate(val_loader)
                        self.model.train()

                if cfg.ckpt_every_steps > 0 and self.global_step % cfg.ckpt_every_steps == 0:
                    self.save_ckpt()
                    barrier()  # 等 rank 0 写完再继续，防止 prune 误删

        # 训练结束兜底：rank 0 存 final ckpt
        if is_main_process():
            self.save_ckpt(tag="final")
        barrier()
        if is_main_process():
            logger.info(f"Training done. global_step={self.global_step}")

    def evaluate(self, val_loader: DataLoader) -> dict:
        """分布式评估：所有 rank 都跑 val_loader，gather 拼到 rank 0 算指标。

        策略：
          - 若 val_loader.dataset 有 DistributedSampler，每 rank 各跑自己那份
          - 否则每 rank 跑全量（慢但稳，适合小验证集）
          - 用 gather_predictions 拼所有 rank 的预测到 rank 0
          - rank 0 算 AUC/f1/acc 并写 val.json + val_prob.json
          - 非 rank 0 返回空 dict（避免无谓计算）
        """
        self.model.eval()
        all_logits, all_labels, all_users, all_amounts = [], [], [], []
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                with self._amp_ctx():
                    logits = self.model(batch)
                    loss = self.loss_fn(logits, batch["label"], batch.get("amount"))
                total_loss += float(reduce_tensor(loss.detach()).item())
                n_batches += 1

                # probs 在 device 上算完再 gather（减少 host-GPU 来回）
                probs = torch.softmax(logits.float(), dim=-1)
                all_logits.append(probs)
                all_labels.append(batch["label"])
                all_amounts.append(batch["amount"])
                all_users.extend(batch["user"])

        # 拼本 rank 内的所有 batch
        if all_logits:
            rank_logits = torch.cat(all_logits, dim=0)
            rank_labels = torch.cat(all_labels, dim=0)
            rank_amounts = torch.cat(all_amounts, dim=0)
        else:
            num_class = 2  # 二分类；若扩展到多类需从 self.model 取
            rank_logits = torch.zeros((0, num_class), device=self.device)
            rank_labels = torch.zeros(0, dtype=torch.long, device=self.device)
            rank_amounts = torch.zeros(0, device=self.device)

        global_logits, global_labels, global_users, global_amounts = gather_predictions(
            rank_logits, rank_labels, all_users, rank_amounts, to_rank=0,
        )

        metrics: dict = {"val_step": self.global_step, "rank": self.rank}
        if not is_main_process():
            barrier()
            self.model.train()
            return metrics

        # rank 0：算指标 + 写文件
        probs_np = global_logits.cpu().numpy()
        labels_np = global_labels.cpu().numpy()
        amounts_np = global_amounts.cpu().numpy()
        preds = (probs_np[:, 1] >= 0.5).astype(int)

        from sklearn.metrics import (
            accuracy_score, average_precision_score, f1_score,
            precision_score, recall_score, roc_auc_score,
        )
        metrics.update({
            "avg_loss": total_loss / max(1, n_batches) / get_world_size(),
            "global_n_samples": int(len(labels_np)),
            "acc": float(accuracy_score(labels_np, preds)),
            "f1": float(f1_score(labels_np, preds, zero_division=0)),
            "precision": float(precision_score(labels_np, preds, zero_division=0)),
            "recall": float(recall_score(labels_np, preds, zero_division=0)),
        })
        try:
            metrics["auc"] = float(roc_auc_score(labels_np, probs_np[:, 1]))
        except ValueError:
            metrics["auc"] = 0.5
        try:
            metrics["ap"] = float(average_precision_score(labels_np, probs_np[:, 1]))
        except ValueError:
            metrics["ap"] = 0.0

        logger.info(
            f"[val rank {self.rank} step {self.global_step}] "
            f"loss={metrics['avg_loss']:.4f} acc={metrics['acc']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
            f"n={metrics['global_n_samples']}"
        )

        prob_path = self._log_dir / f"{self.global_step}_val_prob.json"
        prob_records = {
            u: {
                "pred": int(preds[i]),
                "p0": float(probs_np[i, 0]),
                "p1": float(probs_np[i, 1]),
                "pred_prob": float(probs_np[i, 1]),
                "label": int(labels_np[i]),
                "amount": float(amounts_np[i]),
            }
            for i, u in enumerate(global_users)
        }
        with open(prob_path, "w", encoding="utf-8") as f:
            json.dump(prob_records, f, indent=2, ensure_ascii=False)
        with open(self._log_dir / f"{self.global_step}_val.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        barrier()
        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_ckpt(self, tag: Optional[str] = None) -> None:
        """仅主 rank 写 ckpt；其它 rank no-op。

        ckpt 内容用底层 module.state_dict()（去掉 DDP 包装前缀 'module.'）。
        """
        if not is_main_process():
            return

        step = self.global_step
        suffix = tag or f"step{step}"
        ckpt_path = self._save_dir / f"ckpt_{suffix}.pt"
        module = self._model_module()
        state = {
            "step": step,
            "model": {k: v.cpu() for k, v in module.state_dict().items()},
            "opt": self.optimizer.state_dict() if not tag else None,
            "scheduler": self.scheduler.state_dict() if not tag else None,
            "scaler": self.scaler.state_dict() if not tag else None,
            "loss_name": self.loss_name,
        }
        torch.save(state, ckpt_path)
        logger.info(f"[rank {self.rank}] Saved checkpoint → {ckpt_path}")

        # LoRA adapter 单独存（便于部署期把 adapter 加载到不同 base 模型）
        try:
            from peft import PeftModel
            llama = getattr(module, "llama", None)
            inner = getattr(llama, "llama", None) if llama is not None else None
            if isinstance(inner, PeftModel):
                adapter_dir = self._save_dir / f"adapter_{suffix}"
                inner.save_pretrained(str(adapter_dir))
                logger.info(f"Saved LoRA adapter → {adapter_dir}")
        except Exception as e:
            logger.warning(f"Failed to save LoRA adapter: {e}")

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
        """所有 rank 同时 load（同 ckpt path）；load_opt=False 时不恢复优化器状态。

        若是 DDP 模式，state dict 直接喂给底层 module（不喂包装后的 DDP）。
        """
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        module = self._model_module()
        module.load_state_dict(state["model"], strict=False)
        self.global_step = state.get("step", 0)
        if load_opt and state.get("opt") is not None:
            try:
                self.optimizer.load_state_dict(state["opt"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.scaler.load_state_dict(state["scaler"])
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"[rank {self.rank}] Failed to restore optimizer state: {e}"
                )
        logger.info(
            f"[rank {self.rank}] Loaded checkpoint: {ckpt_path} (step={self.global_step})"
        )

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
        if not is_main_process():
            return
        path = self._log_dir / f"{prefix}_stats.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for i in range(max(0, len(stats["loss"]) - 1), len(stats["loss"])):
                f.write(json.dumps({
                    "step": self.global_step - len(stats["loss"]) + i + 1,
                    "loss": stats["loss"][i],
                    "lr": stats["lr"][i],
                    "rank": self.rank,
                }) + "\n")
