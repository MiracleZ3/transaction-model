"""数据加载工具（cuDF GPU 加速 + pandas 回退）"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw_csv(
    csv_path: str | Path,
    use_gpu: bool = True,
) -> "pd.DataFrame":
    """加载原始 CSV，优先使用 cuDF GPU 加速

    Args:
        csv_path: CSV 文件路径
        use_gpu: 是否使用 cuDF GPU 加速

    Returns:
        DataFrame (cuDF 或 pandas)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    if use_gpu:
        try:
            import cudf
            print("Loading raw data with cuDF (GPU)...")
            t0 = time.time()
            gdf = cudf.read_csv(str(csv_path))
            print(f"cuDF load time: {time.time()-t0:.2f}s")
            print(f"Shape: {gdf.shape[0]:,} rows x {gdf.shape[1]} columns")
            return gdf
        except ImportError:
            print("cuDF not available, falling back to pandas")

    print("Loading raw data with pandas...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"pandas load time: {time.time()-t0:.2f}s")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def load_parquet(
    path: str | Path,
    columns: Optional[list[str]] = None,
    use_gpu: bool = True,
) -> "pd.DataFrame":
    """加载 parquet 文件

    Args:
        path: parquet 文件路径
        columns: 只加载指定列
        use_gpu: 是否使用 cuDF

    Returns:
        DataFrame (cuDF 或 pandas)
    """
    path = Path(path)

    if use_gpu:
        try:
            import cudf
            return cudf.read_parquet(str(path), columns=columns)
        except ImportError:
            pass

    return pd.read_parquet(path, columns=columns)


def print_dataset_summary(df) -> None:
    """打印数据集概要统计"""
    print("Dataset Summary:")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    try:
        n_users = df['User'].nunique()
        print(f"  Users:   {n_users:,}")
    except Exception:
        pass

    fraud_col = "Is Fraud?"
    if fraud_col in df.columns:
        if hasattr(df[fraud_col], 'to_pandas'):
            fraud_counts = df[fraud_col].value_counts().to_pandas()
        else:
            fraud_counts = df[fraud_col].value_counts()
        total_fraud = int(fraud_counts.get('Yes', 0))
        fraud_rate = total_fraud / len(df)
        print(f"  Fraud:   {total_fraud:,} / {len(df):,} ({fraud_rate:.4%})")
