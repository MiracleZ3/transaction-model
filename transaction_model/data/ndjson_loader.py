"""银联风控 NDJSON 数据加载（risk_control_2 / new_ylformer 格式）

把「用户级聚合」的 NDJSON（每行 = 一个用户的全部交易）展开成「每行一笔交易」的
DataFrame，使下游 tokenizer/corpus 管道可以像处理 TabFormer parquet 一样处理它。

NDJSON 行结构（见 ../risk_control_2/docs/data_schema.md §1）::

    {"cert_sm3": "u1", "label": 0|1, "trans": [[...20+字段...], ...]}

展开后每行 = 一笔交易，并保留 cert_sm3 / label / 用户内序号 seq_order / unix_timestap。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from transaction_model.constants import (
    YL_AMOUNT_IDX,
    YL_FIELDS_POS,
    YL_LABEL_KEY,
    YL_TRANS_FIELDS_LEN,
    YL_TRANS_KEY,
    YL_USER_KEY,
)

# cuDF 是可选依赖：没有它则回退 pandas。所有 GPU 操作必须 import 成功才用。
try:
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    cudf = None  # type: ignore


PathLike = Union[str, Path]


def _iter_ndjson_records(
    paths: Iterable[Path],
    encoding: str,
    errors: str,
) -> Iterable[dict]:
    """流式生成 NDJSON 记录（一个 record = 一个用户的所有交易）。"""
    for path in paths:
        with open(path, "r", encoding=encoding, errors=errors) as f:
            for line_no, raw in enumerate(f):
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_no}: {e}"
                    ) from e


def load_ndjson_to_records(
    path_or_dir: PathLike,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> list[dict]:
    """加载 NDJSON 文件（或目录）为原始 record 列表。

    一个 record 对应一行 NDJSON：「一个用户的全量历史」。
    配套 `expand_records_to_dataframe` 把 record 展开成行级 DataFrame。

    Args:
        path_or_dir: 单个 .jsonl 文件或包含 .jsonl 的目录
        encoding: 文件编码
        errors: 解码错误处理（"ignore" 贴合 risk_control_2 的默认）

    Returns:
        record 列表（每项是 NDJSON 的一行）
    """
    path_or_dir = Path(path_or_dir)
    if path_or_dir.is_dir():
        files = sorted(p for p in path_or_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found under: {path_or_dir}"
            )
    elif path_or_dir.is_file():
        files = [path_or_dir]
    else:
        raise FileNotFoundError(f"NDJSON path not found: {path_or_dir}")

    t0 = time.time()
    records = list(_iter_ndjson_records(files, encoding, errors))
    print(
        f"Loaded {len(records):,} user records from {len(files)} file(s) "
        f"in {time.time()-t0:.2f}s"
    )
    return records


def expand_records_to_dataframe(
    records: list[dict],
    use_gpu: bool = True,
    drop_time_cols: bool = False,
) -> "pd.DataFrame":
    """把 record 列表展开为「每行一笔交易」的 DataFrame。

    派生字段（与 risk_control_2 `__mapizer` 一致）：
        - cups_星期  ← datetime.date(year, month, day).weekday()  (0..6)
        - cups_时间段 ← hour/min/sec 分 4 段 (0..3)
        - cert_sm3   ← 顶层 user_id（广播到每笔交易）
        - label      ← 顶层 label（广播到每笔交易，缺失则不创建）
        - seq_order  ← 该用户内交易序号（按文件顺序，0-based）
        - unix_timestap ← trans[9]（用于算 delta_time 与时间切分）

    跳过 trans 数组里下标 >= YL_TRANS_FIELDS_LEN 的元素（样本里有 22 字段，
    loader 只消费 0..19）。

    Args:
        records: load_ndjson_to_records 的输出
        use_gpu: 优先用 cuDF 拼装
        drop_time_cols: 是否丢弃分离的 year/month/day/hour/minutes/seconds 列
            （派生 cups_星期/cups_时间段 后这些原始 time 列可丢）

    Returns:
        DataFrame（cuDF 或 pandas）。列名用 cups_ 前缀，与旧 feature_config 对齐。
    """
    rows: list[dict] = []
    n_users_with_no_trans = 0
    for rec in records:
        trans_list = rec.get(YL_TRANS_KEY) or []
        if not trans_list:
            n_users_with_no_trans += 1
            continue
        user_id = rec.get(YL_USER_KEY)
        label = rec.get(YL_LABEL_KEY)

        for seq_idx, txn in enumerate(trans_list):
            row = _expand_one_txn(txn, user_id, label, seq_idx)
            rows.append(row)

    if not rows:
        raise ValueError(
            "No transactions found after expanding records "
            "(all records had empty 'trans')"
        )

    if n_users_with_no_trans:
        print(
            f"  {n_users_with_no_trans} user(s) had empty 'trans', skipped"
        )

    if use_gpu and cudf is not None:
        # cuDF 24.x 起移除了 DataFrame.from_pandas，标准入口是构造器 cudf.DataFrame(pdf)。
        # 老版本（<23）只有 from_pandas，这里两种都兜住。
        pdf = pd.DataFrame(rows)
        _from_pandas = getattr(cudf.DataFrame, "from_pandas", None)
        df = _from_pandas(pdf) if _from_pandas else cudf.DataFrame(pdf)
    else:
        df = pd.DataFrame(rows)

    if drop_time_cols:
        time_cols = ["year", "month", "day", "hour", "minutes", "seconds"]
        keep = [c for c in time_cols if c in df.columns]
        if keep:
            df = df.drop(columns=keep)

    n_users = df[YL_USER_KEY].nunique() if YL_USER_KEY in df.columns else 0
    print(
        f"  Expanded to {len(df):,} transaction rows "
        f"across {int(n_users):,} users"
    )
    return df


def _expand_one_txn(
    txn: list,
    user_id,
    label,
    seq_idx: int,
) -> dict:
    """展开一笔交易为一个行字典。"""
    if len(txn) < YL_TRANS_FIELDS_LEN:
        raise ValueError(
            f"trans[{seq_idx}] has only {len(txn)} fields; "
            f"need >= {YL_TRANS_FIELDS_LEN} (consuming indices 0..19)"
        )

    row: dict = {}
    # 只消费下标 0..19
    for idx in range(YL_TRANS_FIELDS_LEN):
        col = YL_FIELDS_POS[idx]
        row[col] = txn[idx]

    # 派生字段
    try:
        row["cups_星期"] = _weekday_from_ymd(
            row["year"], row["month"], row["day"]
        )
    except (ValueError, TypeError, KeyError):
        row["cups_星期"] = 0
    row["cups_时间段"] = _time_period(
        row["hour"], row["minutes"], row["seconds"]
    )

    # meta
    row[YL_USER_KEY] = user_id
    if label is not None:
        row[YL_LABEL_KEY] = int(label)
    row["seq_order"] = seq_idx

    return row


def _weekday_from_ymd(year, month, day) -> int:
    import datetime as _dt

    return _dt.date(int(year), int(month), int(day)).weekday()


def _time_period(hour, minute, second) -> int:
    """与 risk_control_2 __get_time_period 一致：0(0-6h)/1(6-12h)/2(12-18h)/3(>=18h)。"""
    hour = min(int(hour), 24)
    minute = min(int(minute), 60)
    second = min(int(second), 60)
    total_seconds = hour * 3600 + minute * 60 + second
    if total_seconds < 6 * 3600:
        return 0
    if total_seconds < 12 * 3600:
        return 1
    if total_seconds < 18 * 3600:
        return 2
    return 3


def load_ndjson(
    path_or_dir: PathLike,
    use_gpu: bool = True,
    drop_time_cols: bool = False,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> "pd.DataFrame":
    """一行接口：加载 NDJSON 并展开为交易行级 DataFrame。

    等价于 load_ndjson_to_records + expand_records_to_dataframe。
    """
    records = load_ndjson_to_records(path_or_dir, encoding=encoding, errors=errors)
    return expand_records_to_dataframe(
        records, use_gpu=use_gpu, drop_time_cols=drop_time_cols
    )


def temporal_split_ndjson(
    df: "pd.DataFrame",
    time_col: str = "unix_timestap",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple:
    """按 unix_timestap 分位数做 train/val/test 时间切分。

    银联旧数据每条交易的 unix_timestap（trans[9]）是单调递增的全局时间戳，
    直接按分位数切即可（与 data/split.py 的「按日期累计行数」思路等价）。

    Returns:
        (train_df, val_df, test_df)
    """
    ts = df[time_col].astype("float64")
    if hasattr(ts, "to_pandas"):
        ts_host = ts.to_pandas()
    else:
        ts_host = ts

    q_train = ts_host.quantile(train_ratio)
    q_val = ts_host.quantile(train_ratio + val_ratio)

    train_mask = df[time_col] <= q_train
    val_mask = (df[time_col] > q_train) & (df[time_col] <= q_val)
    test_mask = df[time_col] > q_val

    if hasattr(df, "to_pandas"):
        masks = [m() if callable(m) else m for m in (train_mask, val_mask, test_mask)]
    else:
        masks = (train_mask, val_mask, test_mask)

    train_df = df[masks[0]].reset_index(drop=True)
    val_df = df[masks[1]].reset_index(drop=True)
    test_df = df[masks[2]].reset_index(drop=True)

    print(
        f"Temporal split on {time_col}: "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def get_amount_series(df: "pd.DataFrame") -> "pd.Series":
    """取每笔交易的金额（trans[19] → cups_交易金额 列）。"""
    from transaction_model.constants import YL_FIELDS_POS

    amount_col = YL_FIELDS_POS[YL_AMOUNT_IDX]
    if amount_col not in df.columns:
        raise KeyError(f"Amount column '{amount_col}' not in DataFrame")
    return df[amount_col]


def print_ndjson_summary(df: "pd.DataFrame") -> None:
    """打印 NDJSON 展开后的概要（用户数、交易数、标签分布）。"""
    print("NDJSON Summary:")
    print(f"  Rows:    {len(df):,}")
    if YL_USER_KEY in df.columns:
        print(f"  Users:   {df[YL_USER_KEY].nunique():,}")
    if YL_LABEL_KEY in df.columns:
        labels = df[YL_LABEL_KEY]
        if hasattr(labels, "to_pandas"):
            labels = labels.to_pandas()
        n_pos = int((labels == 1).sum())
        print(f"  Labels:  {n_pos:,} pos / {len(labels):,} ({n_pos/len(labels):.4%})")
