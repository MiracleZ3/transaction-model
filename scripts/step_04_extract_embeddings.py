"""Step 04: 提取嵌入向量"""
from __future__ import annotations

import argparse

from transaction_model.inference.extract import extract_all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Step 04: Extract Embeddings")
    parser.add_argument("--force", action="store_true", help="强制重新提取")
    parser.add_argument(
        "--dataset-config", default="dataset",
        help="数据集配置名（dataset 走 TabFormer；dataset_yl 走银联 NDJSON）",
    )
    args = parser.parse_args()

    results = extract_all_embeddings(
        force=args.force,
        dataset_config_name=args.dataset_config,
    )
    print(f"\nTotal embeddings: {len(results['embeddings']):,}")


if __name__ == "__main__":
    main()
