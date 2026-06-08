"""Step 01b: 加载银联 NDJSON → 每行一笔交易的 parquet（按时间切分）

路线 A 的 Step 1：把 risk_control_2 风格的「用户级聚合 NDJSON」展开成
「每行一笔交易」的 DataFrame，按 unix_timestap 做 80/10/10 时间切分，保存为
train/val/test.parquet，供 Step 02b tokenize。

用法:
    python scripts/step_01b_load_ndjson.py [--ndjson-dir DIR] [--config dataset_yl]
"""
from __future__ import annotations

import argparse
import time

from transaction_model.config import load_config, resolve_path
from transaction_model.data.ndjson_loader import (
    load_ndjson,
    print_ndjson_summary,
    temporal_split_ndjson,
)


def main():
    parser = argparse.ArgumentParser(description="Step 01b: Load YL NDJSON → parquet")
    parser.add_argument(
        "--config", default="dataset_yl", help="数据集配置名（不含 .yaml）"
    )
    parser.add_argument(
        "--ndjson-dir", default=None,
        help="NDJSON 目录或文件（覆盖 config）",
    )
    parser.add_argument(
        "--drop-time-cols", action="store_true",
        help="丢弃分离的 year/month/day/hour/minutes/seconds 列（派生后冗余）",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="强制用 pandas（不开 cuDF）"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    split_cfg = cfg["split"]

    ndjson_dir = args.ndjson_dir or ds_cfg["ndjson_dir"]
    ndjson_dir = resolve_path(ndjson_dir)
    output_dir = resolve_path(ds_cfg["temporal_split_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 + 展开 NDJSON → 行级 DataFrame
    print(f"Loading NDJSON from {ndjson_dir}")
    t0 = time.time()
    df = load_ndjson(
        ndjson_dir, use_gpu=not args.no_gpu, drop_time_cols=args.drop_time_cols
    )
    print_ndjson_summary(df)
    print(f"  Load+expand time: {time.time()-t0:.1f}s")

    # 2. 时间切分
    train_df, val_df, test_df = temporal_split_ndjson(
        df,
        time_col="unix_timestap",
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
    )

    # 3. 保存 parquet
    t0 = time.time()
    for name, gdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{name}.parquet"
        if hasattr(gdf, "to_parquet") and not callable(getattr(gdf, "to_pandas", None)):
            # cuDF
            gdf.to_parquet(str(path), index=False)
        else:
            gdf.to_parquet(str(path), index=False)
        print(f"Saved: {path} ({len(gdf):,} rows)")
    print(f"Parquet write time: {time.time()-t0:.2f}s")

    print("\nStep 01b complete!")


if __name__ == "__main__":
    main()
