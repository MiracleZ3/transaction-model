"""时间分割工具"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from transaction_model.config import resolve_path


def add_date_column(gdf):
    """为 DataFrame 添加 date 列（用于时间分割）

    Args:
        gdf: DataFrame，需包含 Year, Month, Day 列

    Returns:
        添加了 date 列的 DataFrame
    """
    gdf.columns = [c.strip() for c in gdf.columns]

    year_str = gdf['Year'].astype(str)
    month_str = gdf['Month'].astype(str).str.zfill(2)
    day_str = gdf['Day'].astype(str).str.zfill(2)
    date_str = year_str + '-' + month_str + '-' + day_str

    # cuDF to_datetime（带 format=）对异常日期（越界 year、空串、NaN）非常严格，
    # 会抛 NotImplementedError / OverflowError / KeyError / ValueError；统一兜底到
    # pandas（errors='coerce' 把坏行变 NaT，下游按时间分割自然落到最早段）。
    try:
        import cudf
        try:
            gdf['date'] = cudf.to_datetime(date_str, format='%Y-%m-%d')
        except Exception:
            import pandas as _pd
            col = date_str.to_pandas() if hasattr(date_str, 'to_pandas') else date_str
            gdf['date'] = _pd.to_datetime(col, format='%Y-%m-%d', errors='coerce')
    except ImportError:
        import pandas as pd
        gdf['date'] = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')

    return gdf


def find_cutoff_date(gdf, target_ratio: float):
    """找到累计行数达到 target_ratio 比例的日期

    Args:
        gdf: 含 date 列的 DataFrame
        target_ratio: 目标累计比例 (0~1)

    Returns:
        截断日期
    """
    daily_counts = gdf.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')
    daily_counts['cumulative'] = daily_counts['count'].cumsum()
    total = int(daily_counts['cumulative'].iloc[-1])
    target = total * target_ratio
    filtered = daily_counts[daily_counts['cumulative'] >= target].head(1)
    if hasattr(filtered, 'to_pandas'):
        cutoff_pdf = filtered.to_pandas()
    else:
        cutoff_pdf = filtered
    return cutoff_pdf['date'].iloc[0]


def temporal_split(
    gdf,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """执行时间分割

    Args:
        gdf: 含 date 列的 DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    Returns:
        (train_gdf, val_gdf, test_gdf, train_cutoff, test_cutoff) 元组
    """
    train_cutoff = find_cutoff_date(gdf, train_ratio)
    test_cutoff = find_cutoff_date(gdf, train_ratio + val_ratio)

    print(f"Train/Val cutoff: {train_cutoff.strftime('%Y-%m-%d')}")
    print(f"Val/Test cutoff:  {test_cutoff.strftime('%Y-%m-%d')}")

    train_mask = gdf['date'] < np.datetime64(train_cutoff)
    val_mask = (gdf['date'] >= np.datetime64(train_cutoff)) & (gdf['date'] < np.datetime64(test_cutoff))
    test_mask = gdf['date'] >= np.datetime64(test_cutoff)

    train_gdf = gdf[train_mask].drop(columns=['date']).reset_index(drop=True)
    val_gdf = gdf[val_mask].drop(columns=['date']).reset_index(drop=True)
    test_gdf = gdf[test_mask].drop(columns=['date']).reset_index(drop=True)

    return train_gdf, val_gdf, test_gdf, train_cutoff, test_cutoff


def save_splits(
    train_gdf, val_gdf, test_gdf,
    output_dir: str | Path,
) -> Path:
    """保存时间分割结果为 parquet

    Args:
        train_gdf, val_gdf, test_gdf: 分割后的 DataFrame
        output_dir: 输出目录

    Returns:
        输出目录路径
    """
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for name, gdf in [("train", train_gdf), ("val", val_gdf), ("test", test_gdf)]:
        path = output_dir / f"{name}.parquet"
        gdf.to_parquet(str(path), index=False)
        print(f"Saved: {path} ({len(gdf):,} rows)")
    print(f"Parquet write time: {time.time()-t0:.2f}s")
    return output_dir


def print_split_stats(train_gdf, val_gdf, test_gdf) -> None:
    """打印分割统计"""
    def _stats(gdf):
        n = len(gdf)
        fraud = int((gdf['Is Fraud?'].str.lower() == 'yes').sum())
        return n, fraud, fraud / n * 100

    total = len(train_gdf) + len(val_gdf) + len(test_gdf)
    print(f"{'Split':<8} {'Rows':>12} {'%':>7} {'Fraud':>8} {'Fraud Rate':>12}")
    print("-" * 52)
    for name, gdf in [('Train', train_gdf), ('Val', val_gdf), ('Test', test_gdf)]:
        n, fraud, rate = _stats(gdf)
        print(f"{name:<8} {n:>12,} {n/total*100:>6.2f}% {fraud:>8,} {rate:>11.4f}%")
    print("-" * 52)
    print(f"{'Total':<8} {total:>12,}")
