"""Step 01: 数据下载、时间分割、XGBoost 基线"""
from __future__ import annotations

import argparse
import random
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch

from transaction_model.config import load_config, resolve_path
from transaction_model.data.download import download_tabformer
from transaction_model.data.feature import engineer_features, encode_categorical
from transaction_model.data.loader import load_raw_csv, print_dataset_summary
from transaction_model.data.sampling import (
    create_balanced_sample,
    save_eval_subsets,
    stratified_subsample,
)
from transaction_model.data.split import (
    add_date_column,
    print_split_stats,
    save_splits,
    temporal_split,
)
from transaction_model.detection.xgboost import train_xgb_model


def main():
    parser = argparse.ArgumentParser(description="Step 01: Dataset Baseline")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过 XGBoost 基线训练")
    args = parser.parse_args()

    cfg = load_config("dataset")

    # 0. 设置随机种子
    RANDOM_STATE = cfg["sampling"]["random_state"]
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # 1. 下载数据
    if not args.skip_download:
        csv_path = download_tabformer()

    # 2. 加载
    csv_path = resolve_path(cfg["dataset"]["raw_csv"])
    gdf = load_raw_csv(csv_path)
    print_dataset_summary(gdf)

    # 3. 时间分割
    gdf = add_date_column(gdf)
    train_gdf, val_gdf, test_gdf, train_cutoff, test_cutoff = temporal_split(
        gdf, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"]
    )
    print_split_stats(train_gdf, val_gdf, test_gdf)
    split_dir = save_splits(train_gdf, val_gdf, test_gdf, cfg["dataset"]["temporal_split_dir"])

    # 4. 特征工程
    for gdf_split in [train_gdf, val_gdf, test_gdf]:
        engineer_features(gdf_split)

    # 5. 保存评估子集
    feature_cols = cfg["feature_cols"]
    save_eval_subsets(
        val_gdf, test_gdf,
        feature_cols=feature_cols,
        target_col="_target",
        temporal_split_dir=cfg["dataset"]["temporal_split_dir"],
        n_samples=cfg["sampling"]["eval_samples"],
        random_state=RANDOM_STATE,
    )

    # 6. XGBoost 基线
    if not args.skip_baseline:
        # 转换为 pandas
        if hasattr(train_gdf, 'to_pandas'):
            train_df = train_gdf.to_pandas()
            val_df = val_gdf.to_pandas()
            test_df = test_gdf.to_pandas()
        else:
            train_df = train_gdf
            val_df = val_gdf
            test_df = test_gdf

        X_train, y_train, _ = create_balanced_sample(
            train_df, feature_cols, "_target",
            total_samples=cfg["sampling"]["balanced_train_size"],
            random_state=RANDOM_STATE,
        )
        X_val, y_val = stratified_subsample(
            val_df, feature_cols, "_target",
            n_samples=cfg["sampling"]["eval_samples"],
            random_state=RANDOM_STATE,
        )
        X_test, y_test = stratified_subsample(
            test_df, feature_cols, "_target",
            n_samples=cfg["sampling"]["eval_samples"],
            random_state=RANDOM_STATE,
        )

        X_train_enc, X_val_enc, X_test_enc, _ = encode_categorical(X_train, X_val, X_test)

        xgb_cfg = load_config("xgboost")
        xgb_train_cfg = xgb_cfg.get("xgboost", {}).get("train", {})
        clf, metrics = train_xgb_model(
            X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test,
            params=xgb_cfg["xgboost"]["params_raw"],
            name="XGBoost Baseline",
            early_stopping_rounds=xgb_train_cfg.get("early_stopping_rounds"),
            eval_metric=xgb_train_cfg.get("eval_metric", "auc"),
        )
        print("\nStep 01 complete!")


if __name__ == "__main__":
    main()
