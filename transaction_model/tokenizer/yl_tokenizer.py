"""Token-string → token-id 包装器（银联 YL pipeline 版）。

镜像 ``FinancialTabularTokenizer`` 的 API（encode / decode / vocab_size /
special token 属性），让 ``training/clm_data.py`` 和 ``inference/extract.py``
可以无差别地用 ``YLTabularTokenizer`` 或 ``FinancialTabularTokenizer``。

与 FinancialTabularTokenizer 的关键差别：
  - Financial 版的 vocab 完全静态可算（所有 step 都是 FixedVocab / 闭集 Mapping /
    闭集 Hash），所以可以在 __init__ 里用 dummy 数据 build_vocab。
  - YL 版含「数据驱动 fit」的 step（low-cardinality MappingTokenizer 用了
    `values=None`，AmountTokenizerOptBin 用 quantile），所以**真正训练前**必须先
    在真实数据上 fit 一次得到 vocab，再保存/复用。

提供两种使用模式：

1. **训练前 fit** (生成 corpus 时调用)：

    >>> tok = YLTabularTokenizer(merchant_hash_size=4000)
    >>> tok.fit(df)             # 接收 YLPipeline.preprocess 后的 cuDF
    >>> tok.get_vocab_size()    # 写回 configs/training.yaml::vocab_size
    >>> tok.save("yl_tokenizer.json")

2. **训练/推理用** (clm_data, extract 已有 corpus 文本)：

    >>> tok = YLTabularTokenizer.from_file("yl_tokenizer.json")
    >>> ids = tok.encode("<bos> AMT_3 MERCH_NAME_42 <sep> ... <eos>",
    ...                  max_length=4096)

`encode` 与 FinancialTabularTokenizer.encode 签名一致：接受一个空格分隔的
token 字符串，返回 list[int]（含 pad 到 max_length，若给定）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .yl_pipeline import YLPipeline


class YLTabularTokenizer:
    """银联 YL pipeline 的 tokenizer 接口包装器。

    Parameters
    ----------
    与 YLPipeline.__init__ 一致；special_tokens 可覆盖默认的
    ``<pad> <bos> <eos> <sep> <unk>``。
    """

    DEFAULT_SPECIAL_TOKENS = {
        "pad": "<pad>",
        "bos": "<bos>",
        "eos": "<eos>",
        "sep": "<sep>",
        "unk": "<unk>",
    }

    def __init__(
        self,
        merchant_hash_size: int = 4000,
        merch_name_hash_size: Optional[int] = None,
        amount_strategy: str = "quantile",
        amount_bins: int = 20,
        amount_thresholds: Optional[list] = None,
        include_time_delta: bool = True,
        time_delta_bins: int = 32,
        special_tokens: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.special_tokens = dict(self.DEFAULT_SPECIAL_TOKENS)
        if special_tokens:
            self.special_tokens.update(special_tokens)

        self._pipeline = YLPipeline(
            merchant_hash_size=merchant_hash_size,
            merch_name_hash_size=merch_name_hash_size,
            amount_strategy=amount_strategy,
            amount_bins=amount_bins,
            amount_thresholds=amount_thresholds,
            include_time_delta=include_time_delta,
            time_delta_bins=time_delta_bins,
            special_tokens=dict(self.special_tokens),
            **kwargs,
        )

        # __init__ 时不 build_vocab —— 等待 fit(df) 或 load(state)。
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._refresh_special_ids()

    # ------------------------------------------------------------------
    # 特殊 token id（与 FinancialTabularTokenizer 对齐）
    # ------------------------------------------------------------------

    def _refresh_special_ids(self) -> None:
        """根据 self.vocab 同步 special token id 属性。"""
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.bos_token_id = self.vocab.get("<bos>", 1)
        self.eos_token_id = self.vocab.get("<eos>", 2)
        self.sep_token_id = self.vocab.get("<sep>", 3)
        self.unk_token_id = self.vocab.get("<unk>", 4)
        self.special_token_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.sep_token_id,
            self.unk_token_id,
        }

    # ------------------------------------------------------------------
    # Fit（数据驱动）
    # ------------------------------------------------------------------

    def fit(self, df) -> "YLTabularTokenizer":
        """在 preprocess 后的 cuDF 上 fit YLPipeline，建立 vocab。

        Args:
            df: ``YLPipeline.preprocess`` 的输出（cuDF）

        Returns:
            self
        """
        self._pipeline.fit(df)
        self.vocab = dict(self._pipeline.vocab)
        self.id_to_token = dict(self._pipeline.id_to_token)
        self._refresh_special_ids()
        return self

    def fit_transform(self, df):
        """fit + transform 一步到位（返回 token DataFrame）。"""
        self.fit(df)
        return self._pipeline.transform(df)

    # ------------------------------------------------------------------
    # Vocab 持久化
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Union[str, Path]) -> None:
        """保存 vocab 与 pipeline 配置到 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 反转 vocab 用 int key（JSON 要求 str key 用 str(int) 序列化）
        state = {
            "config": {
                "merchant_hash_size": self._pipeline.merchant_hash_size,
                "merch_name_hash_size": self._pipeline.merch_name_hash_size,
                "amount_strategy": self._pipeline.amount_strategy,
                "amount_bins": self._pipeline.amount_bins,
                "amount_thresholds": self._pipeline.amount_thresholds,
                "include_time_delta": self._pipeline.include_time_delta,
                "time_delta_bins": self._pipeline.time_delta_bins,
            },
            "vocab": self.vocab,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "pipeline_state": self._get_pipeline_state(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "YLTabularTokenizer":
        """从 JSON 恢复（已 fit 状态）。"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        cfg = state["config"]
        tok = cls(**cfg)
        tok.vocab = dict(state["vocab"])
        tok.id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        tok._set_pipeline_state(state.get("pipeline_state", {}))
        tok._refresh_special_ids()
        # _set_pipeline_state 已把每个 step 的 _vocab_built 置 True，但 pipeline 自身
        # 的 is_fitted 标志只有 fit() 才会设。val/test 走 from_file 复用 train 已 fit
        # 的 state，不调 fit，导致后续 transform() 触发 "Must call fit() before
        # transform()" 断言。这里显式置位，让 transform() 通过（state 应已完整）。
        tok._pipeline.is_fitted = True
        return tok

    def _get_pipeline_state(self) -> dict:
        """序列化每个 step 的 fitted vocab + fitted_state（用于 from_file 重建 transform）。

        除了 _idx_to_token，还需存每个 step 的 _get_fitted_state()——尤其
        NumericalTokenizerOptBin 的 builder（cuML KBinsDiscretizer）不能 JSON 化，
        要单独把 bin_edges_/n_bins 拆出来存，from_file 时重建，否则 val/test 跑
        amount 列 transform 会 NotFittedError。
        """
        steps_state = {}
        for tok_id in self._pipeline.tokenizer_order:
            tok = self._pipeline.steps[tok_id]
            idx_to_tok = tok._idx_to_token
            serialized: dict = {
                "idx_to_token": (
                    {
                        str(int(k) if hasattr(k, "item") else k): v
                        for k, v in idx_to_tok.items()
                    }
                    if isinstance(idx_to_tok, dict)
                    else None
                ),
            }
            # 各 step 自带的 fitted_state（bin_edges_ 等）
            try:
                serialized["fitted_state"] = tok._get_fitted_state()
            except Exception:
                serialized["fitted_state"] = {}
            steps_state[tok_id] = serialized
        return {"steps_state": steps_state}

    def _set_pipeline_state(self, state: dict) -> None:
        steps_state = state.get("steps_state", {})
        for tok_id, ser in steps_state.items():
            if tok_id not in self._pipeline.steps:
                continue
            tok = self._pipeline.steps[tok_id]
            it = ser.get("idx_to_token")
            if it is not None:
                tok._idx_to_token = {int(k): v for k, v in it.items()}
                tok._vocab_built = True
            # 恢复 bin_edges_ 等 fitted_state（NumericalTokenizerOptBin 重建 builder）
            fitted = ser.get("fitted_state")
            if fitted:
                try:
                    tok._set_fitted_state(fitted)
                except Exception as _e:
                    # 不让单个 step 恢复失败阻塞加载；该 step transform 时会再报错
                    print(f"  [warn] step {tok_id!r} _set_fitted_state failed: {_e}")

    # ------------------------------------------------------------------
    # Encode / Decode（drop-in 与 FinancialTabularTokenizer 一致）
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        tokens = self.tokenize(text)
        if max_length is not None:
            tokens = tokens[:max_length]
            while len(tokens) < max_length:
                tokens.append("<pad>")
        unk = self.unk_token_id
        return [self.vocab.get(t, unk) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid))
            if tok and tok != "<pad>":
                tokens.append(tok)
        return " ".join(tokens)

    def __repr__(self) -> str:
        return (
            f"YLTabularTokenizer(vocab_size={len(self.vocab)}, "
            f"amount_strategy={self._pipeline.amount_strategy}, "
            f"include_time_delta={self._pipeline.include_time_delta})"
        )
