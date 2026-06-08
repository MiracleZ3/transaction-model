"""Test YLPipeline + YLTabularTokenizer (GPU required — skipped on CPU).

YLPipeline.preprocess / fit / transform 都依赖 cuDF，所以这些测试仅在
GPU 环境下运行。CPU 环境下用静态 vocab 验证 YLTabularTokenizer 的 encode/decode。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


cudf = pytest.importorskip("cudf", reason="YLPipeline tests require cuDF (GPU only)")


def _make_mini_df():
    """构造一份 cuDF fixture：含 YLPipeline 期望的所有列。"""
    import cudf  # 复 import 因为 importorskip 把名字绑到模块不一定进 _
    rows = []
    for i in range(20):
        rows.append({
            "cups_发卡机构地址": f"addr_{i % 5}",
            "cups_发卡机构银行": f"bank_{i % 3}",
            "cups_卡等级": ["普", "金", "白", "钻"][i % 4],
            "cups_收单机构地址": f"saddr_{i % 4}",
            "cups_收单机构银行": f"sbank_{i % 3}",
            "cups_交易代码": f"code_{i % 5}",
            "cups_交易渠道": f"ch_{i % 3}",
            "cups_服务点输入方式": f"pos_{i % 4}",
            "cups_应答码": f"resp_{i % 6}",
            "cups_商户类型": f"mcc_{i % 7}",
            "cups_连接方式": ["直", "间"][i % 2],
            "cups_受卡方名称地址": f"商户名_{i}",
            "cups_交易金额": float(i * 100 + 50),
            "cups_星期": i % 7,
            "cups_时间段": i % 4,
            "unix_timestap": 1712067737.0 + i * 3600,
            "cert_sm3": "u1",
            "seq_order": i,
        })
    return cudf.DataFrame(rows)


def test_yl_pipeline_full_cycle(tmp_path):
    from transaction_model.tokenizer import YLPipeline, YLTabularTokenizer

    df = _make_mini_df()
    pip = YLPipeline(
        merchant_hash_size=100,
        merch_name_hash_size=200,
        amount_strategy="fixed",     # 避免依赖 cuml
        include_time_delta=True,
    )

    # preprocess 不应抛错
    proc = pip.preprocess(df)
    assert "time_delta_s" in proc.columns

    # hash step 应该已经预计算成整数列
    hash_step = "cups_发卡机构地址"
    assert hash_step in proc.columns

    # fit + transform
    pip.fit(proc)
    assert pip.global_vocab_size > 0
    token_df = pip.transform(proc)
    # 应当有与 tokenizer_order 同数的列
    assert len(token_df.columns) == len(pip.tokenizer_order)


def test_yl_tokenizer_save_load(tmp_path):
    from transaction_model.tokenizer import YLPipeline, YLTabularTokenizer

    df = _make_mini_df()
    tok = YLTabularTokenizer(
        merchant_hash_size=100,
        amount_strategy="fixed",
        include_time_delta=True,
    )
    proc = tok._pipeline.preprocess(df)
    tok.fit(proc)

    vocab_size_before = tok.get_vocab_size()
    assert vocab_size_before > 5  # specials + 字段 token

    # 特殊 token 都在词表里
    specials = {"<pad>", "<bos>", "<eos>", "<sep>", "<unk>"}
    assert specials.issubset(set(tok.vocab.keys()))

    # 编码 round-trip
    sample = "<bos> AMT_0 MERCH_NAME_0 <sep> AMT_1 <eos>"
    ids = tok.encode(sample)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id

    # save / load
    out = tmp_path / "yl_tok.json"
    tok.save(out)
    assert out.exists()
    tok2 = YLTabularTokenizer.from_file(out)
    assert tok2.get_vocab_size() == vocab_size_before
    ids2 = tok2.encode(sample)
    assert ids2 == ids
