"""全流程 pipeline 执行器"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = {
    "data":     "scripts/step_01_dataset_baseline.py",
    "tokenize": "scripts/step_02_tokenize_corpus.py",
    "train":    "scripts/step_03_train_model.py",
    "extract":  "scripts/step_04_extract_embeddings.py",
    "detect":   "scripts/step_05_fraud_detection.py",
}

STEP_ORDER = ["data", "tokenize", "train", "extract", "detect"]


def main():
    parser = argparse.ArgumentParser(description="Transaction Model Pipeline")
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help=f"要执行的步骤，逗号分隔: {','.join(STEP_ORDER)}, all"
    )
    parser.add_argument("--demo", action="store_true", help="训练步骤使用 demo 模式")
    parser.add_argument("--force", action="store_true", help="强制重新生成已有产出")
    args = parser.parse_args()

    if args.steps == "all":
        steps = STEP_ORDER
    else:
        steps = [s.strip() for s in args.steps.split(",")]
        for s in steps:
            if s not in STEPS:
                print(f"Unknown step: {s}")
                print(f"Available: {','.join(STEP_ORDER)}")
                sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    python = sys.executable

    for i, step in enumerate(steps):
        script = project_root / STEPS[step]
        cmd = [python, str(script)]
        if args.force and step in ("tokenize", "extract"):
            cmd.append("--force")
        if args.demo and step == "train":
            cmd.append("--demo")

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(steps)}] Running: {step}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*70}")

        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            print(f"\nStep '{step}' failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'='*70}")
    print("  Pipeline complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
