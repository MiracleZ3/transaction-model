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
        # cuML KBinsDiscretizer 本身不能 JSON 化；transform 只依赖 bin_edges_ 和 n_bins。
        # bin_edges_ 形态随 cuML/sklearn 版本而变（List[ndarray] / 2D ndarray / cupy /
        # numba DeviceNDArray），统一递归拉回 host + float 化，保证 json.dump 不崩。
        edges = None
        n_per_feat: list[int] | None = None
        if self._vocab_built and getattr(self.builder, "bin_edges_", None) is not None:
            be = self.builder.bin_edges_
            edges = _to_jsonable_floats(be)
            # 顺带记录每条特征的 bin 数（edges[i] 有 k+1 个点 → k 个 bin）
            try:
                n_per_feat = [len(row) - 1 for row in edges]  # type: ignore[arg-type]
            except Exception:
                n_per_feat = None
        return {
            "bin_edges": edges,
            "n_per_feat": n_per_feat,
            "n_bins": int(getattr(self.builder, "n_bins", self.num_bins)),
            "strategy": self.strategy,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        # 缺 bin_edges（旧 state 兼容）：仅标记 vocab built，但 builder 仍未 fit，
        # amount 列 transform 仍会 NotFittedError。新 state 走完整重建。
        import numpy as _np
        edges_raw = state.get("bin_edges")
        if edges_raw is not None:
            # 还原成 List[ndarray]（sklearn/cuML 都接受的 bin_edges_ 形态）
            rows = [_np.asarray(r, dtype="float64") for r in edges_raw]
            try:
                self.builder.bin_edges_ = (
                    _np.array(rows, dtype=object) if len(rows) > 1 else rows[0]
                )
            except Exception:
                # 只读属性 / 形状不匹配：直接 list 赋，transform 路径用容器访问也能跑
                try:
                    self.builder.bin_edges_ = rows
                except Exception:
                    pass
            # n_bins_：每特征实际 bin 数（n_per_feat 优先；否则按 edges 推算）
            n_per_feat = state.get("n_per_feat")
            try:
                if n_per_feat is not None:
                    self.builder.n_bins_ = _np.asarray(n_per_feat, dtype="int64")
                elif hasattr(self.builder, "n_bins_"):
                    self.builder.n_bins_ = _np.array(
                        [len(r) - 1 for r in rows], dtype="int64"
                    )
            except Exception:
                pass
        self._vocab_built = state.get("_vocab_built", False)


def _to_jsonable_floats(obj):
    """递归把 numpy/cupy/numba 数组、标量、嵌套结构拉回纯 Python float/list。

    cuML KBinsDiscretizer.bin_edges_ 跨版本形态不一（List[ndarray]、2D ndarray、
    cupy.ndarray、numba DeviceNDArray），np.asarray(be).tolist() 有时会保留
    np.float64 或 ndarray 元素，json.dump 会 TypeError: Object of type
    ndarray/float64 is not JSON serializable。这里彻底压平。
    """
    import numpy as _np
    # 优先按"类数组"统一摘出
    try:
        if hasattr(obj, "get"):           # cupy / numba DeviceNDArray
            obj = obj.get()
    except Exception:
        pass
    try:
        arr = _np.asarray(obj)
    except Exception:
        arr = obj
    if isinstance(arr, _np.ndarray):
        # ndarray.tolist() 已把标量数组压成 python float；多错保险再处理
        flat = arr.tolist()
        return _coerce_python(flat)
    if isinstance(arr, (list, tuple)):
        return _coerce_python(arr)
    if isinstance(arr, (int,)):
        return arr
    try:
        return float(arr)
    except (TypeError, ValueError):
        return arr


def _coerce_python(seq):
    out = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            out.append(_coerce_python(item))
        elif hasattr(item, "tolist"):
            out.append(_coerce_python(item.tolist()))
        elif isinstance(item, (int,)):
            out.append(item)
        else:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                # 元组/其它：递归尽力 coerce
                out.append(item)
    return out
