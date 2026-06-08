"""数据采样工具"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_balanced_sample(
    df,
    feature_cols: list[str],
    target_col: str = "_target",
    total_samples: int = 1_000_000,
    fraud_ratio: float = 0.1,
    random_state: int = 42,
):
    """创建平衡训练样本（全欺诈 + 随机正常）

    Args:
        df: 含 target 列的 DataFrame
        feature_cols: 特征列名
        target_col: 目标列名
        total_samples: 总采样数
        fraud_ratio: 欺诈样本占比
        random_state: 随机种子

    Returns:
        (X, y, sampled_idx) 特征矩阵、标签数组和索引
    """
    np.random.seed(random_state)
    fraud_idx = df.index[df[target_col] == 1].tolist()
    normal_idx = df.index[df[target_col] == 0].tolist()

    n_fraud = min(len(fraud_idx), int(total_samples * fraud_ratio))
    n_normal = min(len(normal_idx), total_samples - n_fraud)

    sampled_fraud = np.random.choice(fraud_idx, n_fraud, replace=False)
    sampled_normal = np.random.choice(normal_idx, n_normal, replace=False)
    sampled_idx = np.concatenate([sampled_fraud, sampled_normal])
    np.random.shuffle(sampled_idx)

    sampled_df = df.loc[sampled_idx]
    X = sampled_df[feature_cols].reset_index(drop=True)
    y = sampled_df[target_col].values
    return X, y, sampled_idx


def stratified_subsample(
    df,
    feature_cols: list[str],
    target_col: str,
    n_samples: int,
    random_state: int = 42,
):
    """分层子采样（保持原始类别分布）

    Args:
        df: 含 target 列的 DataFrame
        feature_cols: 特征列名
        target_col: 目标列名
        n_samples: 目标样本数
        random_state: 随机种子

    Returns:
        (X, y) 特征矩阵和标签数组
    """
    if n_samples >= len(df):
        X_sub = df[feature_cols]
        y_sub = df[target_col]
    else:
        _, X_sub, _, y_sub = train_test_split(
            df[feature_cols], df[target_col],
            test_size=n_samples, stratify=df[target_col],
            random_state=random_state
        )
    return X_sub, y_sub.values


def balanced_subsample_by_index(
    labels: np.ndarray,
    total_samples: int = 1_000_000,
    fraud_ratio: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """基于标签数组进行平衡采样（返回采样索引）

    用于 NB04/NB05 中对 embedding 的 train split 做平衡采样

    Args:
        labels: 标签数组 (0/1)
        total_samples: 总采样数
        fraud_ratio: 欺诈占比
        random_state: 随机种子

    Returns:
        采样后的索引数组
    """
    fraud_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]

    np.random.seed(random_state)
    n_fraud = min(len(fraud_idx), int(total_samples * fraud_ratio))
    n_normal = min(len(normal_idx), total_samples - n_fraud)

    sampled = np.concatenate([
        np.random.choice(fraud_idx, n_fraud, replace=False),
        np.random.choice(normal_idx, n_normal, replace=False),
    ])
    np.random.shuffle(sampled)
    return sampled


def save_eval_subsets(
    val_df, test_df,
    feature_cols: list[str],
    target_col: str,
    temporal_split_dir: str,
    n_samples: int = 100_000,
    random_state: int = 42,
) -> None:
    """创建并保存评估子集 (val_eval / test_eval parquet)

    Args:
        val_df: 验证集 DataFrame
        test_df: 测试集 DataFrame
        feature_cols: 特征列
        target_col: 目标列
        temporal_split_dir: 时间分割数据目录
        n_samples: 评估子采样数
        random_state: 随机种子
    """
    split_dir = Path(temporal_split_dir)

    for split_name, df in [("val", val_df), ("test", test_df)]:
        _, X_sub, _, y_sub = train_test_split(
            df[feature_cols], df[target_col],
            test_size=n_samples, stratify=df[target_col],
            random_state=random_state
        )
        # 从原始 parquet 提取对应行
        raw_full = pd.read_parquet(split_dir / f"{split_name}.parquet")
        subset_raw = raw_full.iloc[X_sub.index].reset_index(drop=True)
        out_path = split_dir / f"{split_name}_eval.parquet"
        subset_raw.to_parquet(out_path, index=False)
        fraud_vals = subset_raw['Is Fraud?'].astype(str).str.lower().eq('yes')
        print(f"Saved {split_name}_eval: {len(subset_raw):,} rows -> {out_path.name} "
              f"(fraud {fraud_vals.sum():,}, {fraud_vals.mean():.4%})")
