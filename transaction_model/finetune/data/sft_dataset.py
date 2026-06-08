"""Route C 数据模块：NDJSON → [B,T,L] 张量 collate（CPU 友好）。

把每条记录（一个用户的交易序列）变成 collate 期望的样本：
    {
      "input_ids":       LongTensor [T', L]   每笔交易 15 + special token
      "attention_mask":  LongTensor [T', L]
      "his_time_stap":   LongTensor [T']      历史时间戳（秒）
      "delta_time_stap": LongTensor [T']      相邻交易小时差（首位为 0）
      "label":           int
      "amount":          float                末笔金额（用于 focal_with_amount）
      "user":            str                  cert_sm3
    }

`T'` 是该用户的实际交易数（按时间窗 hiswindow 截断前）。
最后由 `collate_fn` 把 variable T' 补齐成 batch 的 [B, T, L]。

关键决策：
  - 不复用 YLPipeline.transform（它一次处理整列 cuDF，对单交易不友好）
  - 直接用单条交易构造一个 1-row DataFrame 喂给 YLPipeline 的 transform
    （YLPipeline 是无状态的，fit 后可复用；transform 接受 DataFrame）
  - 词表语义与 route A 完全一致：每笔交易最终 <bos> t1...t15 <eos>
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from transaction_model.constants import (
    YL_AMOUNT_IDX,
    YL_FIELDS_POS,
    YL_LABEL_KEY,
    YL_TRANS_FIELDS_LEN,
    YL_TRANS_KEY,
    YL_USER_KEY,
)


def _time_period(hour, minute, second) -> int:
    hour = min(int(hour), 24)
    minute = min(int(minute), 60)
    second = min(int(second), 60)
    total = hour * 3600 + minute * 60 + second
    if total < 6 * 3600:
        return 0
    if total < 12 * 3600:
        return 1
    if total < 18 * 3600:
        return 2
    return 3


def _weekday(year, month, day) -> int:
    import datetime as _dt
    try:
        return _dt.date(int(year), int(month), int(day)).weekday()
    except (ValueError, TypeError):
        return 0


def txn_to_df_row(txn: list) -> dict:
    """把一笔交易的 list 展开成 YLPipeline.preprocess 期望的 DataFrame 行。

    与 ndjson_loader._expand_one_txn 一致，但不带 cert_sm3/label/seq_order
    （这些在 collate 层维护）。
    """
    if len(txn) < YL_TRANS_FIELDS_LEN:
        raise ValueError(
            f"txn has {len(txn)} fields; need >={YL_TRANS_FIELDS_LEN}"
        )
    row = {}
    for idx in range(YL_TRANS_FIELDS_LEN):
        row[YL_FIELDS_POS[idx]] = txn[idx]
    row["cups_星期"] = _weekday(row["year"], row["month"], row["day"])
    row["cups_时间段"] = _time_period(
        row["hour"], row["minutes"], row["seconds"]
    )
    return row


def encode_one_txn_via_pipeline(
    pipeline, vocab: dict, txn: list,
    bos_token_id: int, eos_token_id: int, sep_token_id: int,
    pad_token_id: int, max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """单条交易 → [L] token ids + [L] attention mask。

    用法：构造一个 1-row 的 pandas DataFrame，挂上 cert_sm3/unix_timestap 等
    列让 preprocess 不报错，跑 preprocess→transform，把 transform 出来的 15 个
    token 字符串 → ids。
    """
    import pandas as pd
    try:
        import cudf
        _have_cudf = True
    except ImportError:
        cudf = None
        _have_cudf = False

    row = txn_to_df_row(txn)
    # 给 preprocess 提供依赖列
    row["cert_sm3"] = "_"
    if "unix_timestap" not in row:
        row["unix_timestap"] = float(YL_FIELDS_POS[9] and row.get("unix_timestap", 0))
    row.setdefault("unix_timestap", 0.0)

    df = pd.DataFrame([row])
    if _have_cudf:
        try:
            df = cudf.DataFrame.from_pandas(df)
        except Exception:
            df = pd.DataFrame([row])

    proc = pipeline.preprocess(df)
    token_df = pipeline.transform(proc)

    # 取每个 step 对应列的第一行 token 字符串，查词表得 id
    ids: List[int] = [bos_token_id]
    unk = vocab.get("<unk>", 4)
    for col in pipeline.tokenizer_order:
        if col not in token_df.columns:
            ids.append(unk)
            continue
        val = token_df[col].iloc[0]
        val_str = str(val)
        ids.append(vocab.get(val_str, unk))
    ids.append(eos_token_id)

    # pad/trunc 到 max_len
    L = len(ids)
    if L > max_len:
        ids = ids[:max_len]
        ids[-1] = eos_token_id
    else:
        ids = ids + [pad_token_id] * (max_len - L)

    mask = (np.array(ids) != pad_token_id).astype(np.int64)
    return np.array(ids, dtype=np.int64), mask


class SftNDJsonDataset(Dataset):
    """索引式 SFT 数据集：从 NDJSON 加载到内存，按 sample 索引返回。

    对每个 NDJSON 行（一个用户），构造一个样本：
        input_ids         [T_user, L]   每笔交易的 token ids
        attention_mask    [T_user, L]
        his_time_stap     [T_user]
        delta_time_stap   [T_user]
        label             int
        amount            float         末笔金额
        user              str

    所有 T_user 不做 padding（变长）；padding 由 collate_fn 在 batch 时处理。
    """

    def __init__(
        self,
        ndjson_path: Path,
        pipeline,
        vocab: dict,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        sep_token_id: int,
        max_txn_len: int = 32,
        minleng: int = 1,
        max_hiswindow: int = 512,
    ):
        self.pipeline = pipeline
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.max_txn_len = max_txn_len
        self.minleng = minleng
        self.max_hiswindow = max_hiswindow

        self.samples: List[dict] = []
        self._load(ndjson_path)

    def _load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                trans_list = rec.get(YL_TRANS_KEY) or []
                if len(trans_list) < self.minleng:
                    continue
                # 截窗
                if len(trans_list) > self.max_hiswindow:
                    trans_list = trans_list[-self.max_hiswindow:]

                # 逐笔编码
                L = self.max_txn_len
                input_ids = np.full((len(trans_list), L), self.pad_token_id, dtype=np.int64)
                attention_mask = np.zeros((len(trans_list), L), dtype=np.int64)
                unix_ts = []
                for i, txn in enumerate(trans_list):
                    ids, mask = encode_one_txn_via_pipeline(
                        self.pipeline, self.vocab, txn,
                        self.bos_token_id, self.eos_token_id,
                        self.sep_token_id, self.pad_token_id, L,
                    )
                    input_ids[i] = ids
                    attention_mask[i] = mask
                    unix_ts.append(float(txn[9]) if len(txn) > 9 else 0.0)

                unix_ts_np = np.array(unix_ts, dtype=np.int64)
                delta_secs = np.zeros(len(unix_ts_np), dtype=np.int64)
                if len(unix_ts_np) > 1:
                    delta_secs[1:] = (unix_ts_np[1:] - unix_ts_np[:-1])
                delta_hours = (delta_secs // 3600).astype(np.int64)
                delta_hours = np.clip(delta_hours, 0, 2**31 - 1)

                # amount = 末笔金额
                last_txn = trans_list[-1]
                amount = float(last_txn[YL_AMOUNT_IDX]) if len(last_txn) > YL_AMOUNT_IDX else 0.0

                self.samples.append({
                    "input_ids": input_ids,             # [T, L]
                    "attention_mask": attention_mask,   # [T, L]
                    "his_time_stap": unix_ts_np,        # [T]
                    "delta_time_stap": delta_hours,     # [T]
                    "label": int(rec.get(YL_LABEL_KEY, 0)),
                    "amount": amount,
                    "user": str(rec.get(YL_USER_KEY, "")),
                })
        print(f"  Loaded {len(self.samples)} samples from {path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(
    batch: List[dict],
    pad_token_id: int,
    max_hiswindow: int,
):
    """把变长 sample list 补齐成 batch。

    Returns: dict 含
        input_ids          LongTensor [B, T, L]
        attention_mask     LongTensor [B, T, L]
        gpt2_attention_mask BoolTensor [B, T]
        his_time_stap      LongTensor [B, T]
        delta_time_stap    LongTensor [B, T]
        lens_in            LongTensor [B]
        label              LongTensor [B]
        amount             FloatTensor [B]
        user               list[str]   len=B
    """
    B = len(batch)
    T = min(max(len(s["input_ids"]) for s in batch), max_hiswindow)
    L = batch[0]["input_ids"].shape[1]

    input_ids = torch.full((B, T, L), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, T, L), dtype=torch.long)
    gpt2_mask = torch.zeros((B, T), dtype=torch.bool)
    his_ts = torch.zeros((B, T), dtype=torch.long)
    delta_ts = torch.zeros((B, T), dtype=torch.long)
    lens_in = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)
    amounts = torch.zeros(B, dtype=torch.float32)
    users: List[str] = []

    for b, s in enumerate(batch):
        n = min(len(s["input_ids"]), T)
        input_ids[b, :n] = torch.from_numpy(s["input_ids"][:n])
        attn[b, :n] = torch.from_numpy(s["attention_mask"][:n])
        gpt2_mask[b, :n] = True
        his_ts[b, :n] = torch.from_numpy(s["his_time_stap"][:n])
        delta_ts[b, :n] = torch.from_numpy(s["delta_time_stap"][:n])
        lens_in[b] = n
        labels[b] = s["label"]
        amounts[b] = s["amount"]
        users.append(s["user"])

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "gpt2_attention_mask": gpt2_mask,
        "his_time_stap": his_ts,
        "delta_time_stap": delta_ts,
        "lens_in": lens_in,
        "label": labels,
        "amount": amounts,
        "user": users,
    }


def prepare_collate(pad_token_id: int, max_hiswindow: int = 512):
    """给 DataLoader 用的 functools.partial 入口。"""
    return functools.partial(
        collate_fn,
        pad_token_id=pad_token_id,
        max_hiswindow=max_hiswindow,
    )
