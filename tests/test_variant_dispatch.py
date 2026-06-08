"""Test clm_data tokenizer-variant dispatching.

不需要 GPU —— 只验证 variant 切换逻辑能正确选到 FinancialTabularTokenizer
或 YLTabularTokenizer，以及在 yl 缺 state_path 时抛错。
"""
from __future__ import annotations

import pytest


def test_build_tokenizer_tabformer():
    from transaction_model.training.clm_data import _build_tokenizer

    tok = _build_tokenizer(variant="tabformer", merchant_hash_size=200)
    # FinancialTabularTokenizer 是静态可构建的
    assert tok.get_vocab_size() > 0
    assert tok.pad_token_id == 0


def test_build_tokenizer_yl_requires_state(tmp_path):
    from transaction_model.training.clm_data import _build_tokenizer

    # 无 state_path 必报错
    with pytest.raises(ValueError, match="tokenizer_state_path"):
        _build_tokenizer(variant="yl")

    # 不存在的 state_path 必报错
    with pytest.raises(FileNotFoundError):
        _build_tokenizer(
            variant="yl",
            tokenizer_state_path=str(tmp_path / "no_such.json"),
        )


def test_build_tokenizer_unknown_variant():
    from transaction_model.training.clm_data import _build_tokenizer

    with pytest.raises(ValueError, match="Unknown tokenizer_variant"):
        _build_tokenizer(variant="bogus")


def test_extract_labels_prefer_col():
    """_extract_labels 应优先匹配 prefer_col。"""
    import pandas as pd

    from transaction_model.inference.extract import _extract_labels

    df = pd.DataFrame({"label": [0, 1, 1], "Is Fraud?": ["No", "Yes", "No"]})
    # prefer_col=label → 取 label 列
    labels = _extract_labels(df, prefer_col="label")
    assert labels.tolist() == [0, 1, 1]

    # 不指定 → 默认顺序 Is Fraud? 优先
    labels2 = _extract_labels(df)
    assert labels2.tolist() == [0, 1, 0]
