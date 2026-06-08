"""Step 06: Route C — Llama+GPT2 混合下游微调。

前置：
  - Route A 的预训 Llama checkpoint 已保存到 models/decoder-yl/
  - Route C 词表 state 已 fit 并保存到 data/yl/yl_tokenizer.json
    （由 step_02_tokenize_ndjson.py 产出）

用法（单卡）：
    python scripts/step_06_finetune_routec.py \\
        --config configs/routec/default.json \\
        --demo                       # 仅跑 30 步（冒烟）

用法（单机多卡 DDP）：
    torchrun --nproc-per-node=8 --master-port=29500 \\
        scripts/step_06_finetune_routec.py \\
        --config configs/routec/default_multinode.json

用法（多机多卡）：
    # 每节点跑一份（仅 NODE_RANK 不同），详见 scripts/routec_ddp_multi_node.sh
    torchrun --nproc-per-node=8 --nnodes=2 --node-rank=$NODE_RANK \\
        --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \\
        scripts/step_06_finetune_routec.py \\
        --config configs/routec/default_multinode.json

torchrun 自动注入 LOCAL_RANK/RANK/WORLD_SIZE；单卡直接 `python ...` 时
未设环境变量分布式自动退化。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from transaction_model.finetune.config import load_config
from transaction_model.finetune.distributed import (
    cleanup_distributed, is_main_process, setup_distributed,
)
from transaction_model.finetune.factory import (
    build_datasets, build_model, build_tokenizer, build_trainer,
)


def main():
    parser = argparse.ArgumentParser(description="Step 06: Route C finetune")
    parser.add_argument(
        "--config", default="configs/routec/default.json",
        help="Route C JSON config 路径",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="覆盖 max_steps=30, val_every_steps=10 (冒烟)",
    )
    parser.add_argument(
        "--auto-load", action="store_true",
        help="从 save_dir 里最新的 ckpt_*.pt 自动续训",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="训练设备（如 cuda / cuda:0 / cpu）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="覆盖 config 的 max_steps",
    )
    args = parser.parse_args()

    # 1. 初始化分布式（单卡时自动退化为 (0, 0, 1)）
    local_rank, rank, world_size = setup_distributed(timeout_minutes=60)

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(name)s - %(message)s",
    )

    cfg = load_config(args.config)
    if args.demo:
        cfg.step_scheduler["max_steps"] = 30
        cfg.step_scheduler["val_every_steps"] = 10
        cfg.step_scheduler["ckpt_every_steps"] = 30
    if args.max_steps is not None:
        cfg.step_scheduler["max_steps"] = args.max_steps

    # 2. 词表
    tokenizer = build_tokenizer(cfg)

    # 3. 模型（含 LoRA 注入）
    model = build_model(cfg)

    # 4. 数据（按 shard_mode 自适应：DistributedSampler / IterableDataset 文件分片）
    train_loader, val_loader = build_datasets(cfg, tokenizer)

    # 5. Trainer（DDP 包装在 Trainer.__init__ 内部完成）
    trainer = build_trainer(
        cfg, model,
        device=args.device,
        rank=rank, world_size=world_size, local_rank=local_rank,
    )

    # 6. 续训：所有 rank 同时 load（不含进程同步，由 load_ckpt 内 strict=False 兜底）
    if args.auto_load:
        save_dir = Path(cfg.paths.get("save_ckpt", "models/routec"))
        if save_dir.exists():
            ckpts = sorted(save_dir.glob("ckpt_step*.pt"))
            if ckpts:
                trainer.load_ckpt(ckpts[-1], load_opt=True)

    # 7. 跑
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Route C 微调：task_type={cfg.task_type} loss={trainer.loss_name}")
        print(f"  rank/world_size: {rank}/{world_size} (local_rank={local_rank})")
        print(f"  train samples: {len(train_loader.dataset):,}")
        print(f"  val samples:   {len(val_loader.dataset):,}")
        print(f"  max_steps:     {cfg.step_scheduler.get('max_steps')}")
        print(f"  strategy:      {cfg.strategy.get('name', 'single')}")
        print(f"{'='*60}\n")

    try:
        trainer.train(train_loader, val_loader)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
