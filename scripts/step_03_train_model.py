"""Step 03: 训练 decoder 基础模型"""
from __future__ import annotations

import argparse

from transaction_model.training.train import launch_training


def main():
    parser = argparse.ArgumentParser(description="Step 03: Train Model")
    parser.add_argument("--demo", action="store_true", help="Demo 模式 (30步)")
    parser.add_argument("--max-steps", type=int, default=None, help="最大训练步数")
    parser.add_argument("--nproc", type=int, default=1, help="GPU 数量")
    parser.add_argument(
        "--variant", choices=["tabformer", "yl"], default="tabformer",
        help="数据路线：tabformer（默认，走 configs/training.yaml）/"
             "yl（走 configs/training_yl.yaml，需先跑 step_02_tokenize_ndjson）",
    )
    args = parser.parse_args()

    config_name = "training_yl" if args.variant == "yl" else "training"
    result = launch_training(
        config_name=config_name,
        demo=args.demo,
        max_steps=args.max_steps,
        nproc=args.nproc,
    )
    print(f"\nTraining exit code: {result.returncode}")


if __name__ == "__main__":
    main()
