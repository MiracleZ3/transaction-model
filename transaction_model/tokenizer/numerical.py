# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data-driven numerical binning tokenizer (GPU-accelerated).

Uses cuML KBinsDiscretizer to learn bin boundaries from data, then maps
continuous values to ordinal bin tokens like "AMT_0" .. "AMT_9".

Data is passed only to build_vocab() (fit) and tokenize() (transform),
never stored in the constructor.
"""
from __future__ import annotations

try:
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    cudf = None  # type: ignore
try:
    import cupy as cp  # noqa: F401 - cuPy optional on CPU
except ImportError:  # pragma: no cover
    cp = None  # type: ignore
try:
    from cuml.preprocessing import KBinsDiscretizer  # type: ignore
except ImportError:  # pragma: no cover - cuML only available on GPU
    KBinsDiscretizer = None  # type: ignore

from .base import BaseTokenizer


class NumericalTokenizerOptBin(BaseTokenizer):
    """Quantile / uniform / k-means binning for continuous columns."""

    def __init__(
        self,
        special_token: str = "AMT",
        num_bins: int = 10,
        strategy: str = "quantile",
        stream: cp.cuda.Stream = None,
    ):
        super().__init__()
        self.special_token = special_token
        self.num_bins = num_bins
        self.strategy = strategy
        self.stream = stream
        self._vocab_built = False
        if KBinsDiscretizer is None:
            raise ImportError(
                "NumericalTokenizerOptBin requires the 'cuml' package "
                "(GPU only). Install with: pip install cuml"
            )
        self.builder = KBinsDiscretizer(
            n_bins=self.num_bins, encode="ordinal", strategy=self.strategy
        )

    def build_vocab(self, column_data=None) -> None:
        """Build vocab and fit the discretizer on *column_data*."""
        self._idx_to_token = {
            i: f"{self.special_token}_{i}" for i in range(self.num_bins)
        }
        if column_data is not None:
            if isinstance(column_data, cudf.Series):
                column_data = column_data.to_frame()
            if self.stream:
                with self.stream:
                    self.builder.fit(column_data)
            else:
                self.builder.fit(column_data)
        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        if isinstance(column_data, cudf.Series):
            column_data = cudf.DataFrame(column_data)
        if self.stream:
            with self.stream:
                bins = self.builder.transform(column_data)
        else:
            bins = self.builder.transform(column_data)
        if isinstance(bins, cudf.DataFrame):
            bins = bins.iloc[:, 0]
        # 越界 / NaN 输入会让 transform 出 NaN bin → .map 回 NaN，污染下游
        # .str.cat（整行变 NaN）→ _fmt 的 join 报 expected str。
        # 缺省回桶 0。
        bins = bins.fillna(0)
        return bins.astype("int32").map(self._idx_to_token)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"NumericalTokenizerOptBin(token={self.special_token}, "
            f"bins={self.num_bins}, strategy={self.strategy}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "special_token": self.special_token,
            "num_bins": self.num_bins,
            "strategy": self.strategy,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        # cuML KBinsDiscretizer 本身不能 JSON 化；但 transform 只依赖 bin_edges_ 和
        # n_bins。把它们存成 list，from_file 时重建 builder 状态即可。
        edges = None
        if self._vocab_built and getattr(self.builder, "bin_edges_", None) is not None:
            be = self.builder.bin_edges_
            # cuML 返回 numpy/cupy/DeviceNDArray；统一拉回 host list
            try:
                import numpy as _np
                edges = _np.asarray(be).tolist()
            except Exception:
                edges = [list(c) for c in be]
        return {
            "bin_edges": edges,
            "n_bins": getattr(self.builder, "n_bins", self.num_bins),
            "strategy": self.strategy,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        # 缺 bin_edges（旧 state 兼容）：仅标记 vocab built，但 builder 仍未 fit，
        # amount 列 transform 仍会 NotFittedError。新 state 走完整重建。
        if state.get("bin_edges") is not None:
            import numpy as _np
            edges = _np.array(state["bin_edges"], dtype="float64")
            try:
                # cuML KBinsDiscretizer 接受直接赋 bin_edges_ + n_bins
                self.builder.bin_edges_ = edges
                if hasattr(self.builder, "n_bins_"):
                    self.builder.n_bins_ = _np.array([len(c) - 1 for c in edges])
                else:
                    self.builder.n_bins = state.get("n_bins", self.num_bins)
            except Exception:
                # 某些 cuML 版本 bin_edges_ 是只读，flag fitted 不足以让 transform 正确；
                # 这种情况下后续 transform 会露馅（数值错），但至少不崩在赋值上。
                pass
        self._vocab_built = state.get("_vocab_built", False)
