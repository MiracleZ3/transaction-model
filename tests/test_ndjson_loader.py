"""Test NDJSON loader (CPU-only, pandas path).

用 new_ylformer/data_sample 的小样本验证 NDJSON → 行级 DataFrame 的展开逻辑。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from transaction_model.constants import (
    YL_FIELDS_POS,
    YL_LABEL_KEY,
    YL_TRANS_FIELDS_LEN,
    YL_TRANS_KEY,
    YL_USER_KEY,
)
from transaction_model.data.ndjson_loader import (
    expand_records_to_dataframe,
    load_ndjson,
    load_ndjson_to_records,
    temporal_split_ndjson,
)


# tests/test_ndjson_loader.py → tests/ → transaction-model/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 优先用仓库内置的 examples/sample_data/smoke.jsonl（clone 后即可跑）；
# 若不存在则回退到工作区外的 new_ylformer/sample（本地开发时）。
INREPO_SAMPLE = PROJECT_ROOT / "examples" / "sample_data" / "smoke.jsonl"
EXTERNAL_SAMPLE = PROJECT_ROOT.parent / "new_ylformer" / "data_sample" / "data_sample.jsonl"
SAMPLE_FILE = INREPO_SAMPLE if INREPO_SAMPLE.exists() else EXTERNAL_SAMPLE


def _sample_available() -> bool:
    return SAMPLE_FILE.exists()


@pytest.fixture
def ndjson_tmpfile(tmp_path):
    """构造一份最小的 NDJSON 测试 fixture（不依赖外部样本）。"""
    records = [
        {
            YL_USER_KEY: "u1",
            YL_LABEL_KEY: 0,
            YL_TRANS_KEY: [
                # 20 字段：发卡机构地址, 发卡机构银行, 卡等级, 年月日时分秒,
                # unix_timestap, 收单机构地址, 收单机构银行, 交易代码, 渠道,
                # 服务点输入方式, 应答码, 商户类型, 连接方式, 受卡方名称, 金额
                ["0000", "工商银行", "普",
                 2024, 4, 2, 22, 22, 17, 1712067737.0,
                 "银联代发", "支付宝", "消费类", "无线", "手工,不含PIN",
                 "资金不足, 失败", "6066", "间", "支付宝-转账/**华", 34],
                ["0001", "建设银行", "金",
                 2024, 4, 3, 10, 5, 0, 1712150000.0,
                 "广东省", "财付通", "贷记", "网上", "手工,不含PIN",
                 "成功", "未列入", "直", "微信支付", 100],
            ],
        },
        {
            YL_USER_KEY: "u2",
            YL_LABEL_KEY: 1,
            YL_TRANS_KEY: [
                ["0002", "招商银行", "白",
                 2024, 5, 1, 8, 0, 0, 1714540800.0,
                 "上海", "支付宝", "消费类", "无线", "免密",
                 "成功", "5812", "间", "美团", 50],
            ],
        },
    ]
    path = tmp_path / "fixture.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path, records


def test_load_and_expand(ndjson_tmpfile):
    path, records = ndjson_tmpfile
    loaded = load_ndjson_to_records(path)
    assert len(loaded) == len(records)
    # 总交易数 = sum 各用户 trans 长度
    n_total = sum(len(r[YL_TRANS_KEY]) for r in records)
    assert n_total == 3

    df = expand_records_to_dataframe(loaded, use_gpu=False)
    assert len(df) == n_total

    # 每笔交易所消费的 cups_* 字段都应在列里
    for idx in range(YL_TRANS_FIELDS_LEN):
        assert YL_FIELDS_POS[idx] in df.columns

    # 用户标识广播到每行
    assert YL_USER_KEY in df.columns
    assert set(df[YL_USER_KEY].unique()) == {"u1", "u2"}

    # 标签广播
    assert YL_LABEL_KEY in df.columns
    u1_rows = df[df[YL_USER_KEY] == "u1"]
    assert (u1_rows[YL_LABEL_KEY] == 0).all()
    u2_rows = df[df[YL_USER_KEY] == "u2"]
    assert (u2_rows[YL_LABEL_KEY] == 1).all()

    # 派生字段
    assert "cups_星期" in df.columns
    assert "cups_时间段" in df.columns
    # u1 第一笔：2024-04-02 是周二（weekday=1），22:22:17 ~ 时间段 3
    first_row = df[df[YL_USER_KEY] == "u1"].iloc[0]
    assert first_row["cups_星期"] == pytest.approx(1)
    assert first_row["cups_时间段"] == 3


def test_temporal_split(ndjson_tmpfile):
    path, _ = ndjson_tmpfile
    df = load_ndjson(path, use_gpu=False)
    train, val, test = temporal_split_ndjson(
        df, time_col="unix_timestap", train_ratio=0.6, val_ratio=0.2
    )
    # 三段加总应当覆盖全部交易
    assert len(train) + len(val) + len(test) == len(df)


def test_short_trans_raises(tmp_path):
    """trans 数组若 < YL_TRANS_FIELDS_LEN 字段应报错。"""
    bad = [{YL_USER_KEY: "uX", YL_TRANS_KEY: [["only", "two", "fields"]]}]
    with pytest.raises(ValueError, match="need >="):
        expand_records_to_dataframe(bad, use_gpu=False)


def test_real_sample_if_available():
    """加载仓库内置的 smoke.jsonl（或外部 sample 兜底）做冒烟测试。"""
    import json as _j
    if not _sample_available():
        pytest.skip("No sample data found.")
    df = load_ndjson(SAMPLE_FILE, use_gpu=False)
    assert len(df) > 0
    assert YL_USER_KEY in df.columns
    # 字段下标 18（受卡方名称地址）与 19（金额）必须存在
    assert "cups_受卡方名称地址" in df.columns
    assert "cups_交易金额" in df.columns
    # 字段 0..19 都应在列里
    for idx in range(YL_TRANS_FIELDS_LEN):
        assert YL_FIELDS_POS[idx] in df.columns
    # 行数应等于 sample 内所有用户交易数之和
    with open(SAMPLE_FILE) as f:
        records = [_j.loads(l) for l in f if l.strip()]
    assert len(df) == sum(len(r[YL_TRANS_KEY]) for r in records)
    # 仓库内置 fixture 应同时含正负样本；外部 sample 可能只有 1 类（放宽要求）
    if INREPO_SAMPLE.exists() and YL_LABEL_KEY in df.columns:
        assert df[YL_LABEL_KEY].nunique() >= 2, "smoke fixture must have >=2 label classes"

