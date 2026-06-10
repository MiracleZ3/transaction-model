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
        # 关键：不走 self.builder.transform！cuML KBinsDiscretizer 在单特征 / 数据稀疏
        # 场景会把 bin_edges_ 拍平成 1D numpy（而非 List[ndarray]），transform 内部
        # 索引 bin_edges[jj][1:] 时炸 IndexError（详见服务器 step_06 traceback）。
        # 我们直接读 bin_edges_，自己 digitize——既避免该 bug，又能在 cupy/cuml/numpy
        # 后端一致。
        import numpy as _np

        # 1. 拉到 host numpy 1D float64 列
        if hasattr(column_data, "to_pandas"):
            pdf = column_data.to_pandas()
            try:
                vals = pdf.to_numpy(dtype="float64")
            except Exception:
                vals = _np.asarray(pdf, dtype="float64")
        elif hasattr(column_data, "values"):
            v = column_data.values
            try:
                vals = _np.asarray(v.get() if hasattr(v, "get") else v, dtype="float64")
            except Exception:
                vals = _np.asarray(v, dtype="float64")
        else:
            vals = _np.asarray(column_data, dtype="float64")
        # cuDF DataFrame 输入（to_frame 后）vals 是 2D (n,1)；统一展平成 1D
        if vals.ndim == 2:
            vals = vals.reshape(-1, vals.shape[-1])[:, 0]
        elif vals.ndim == 0:
            vals = vals.reshape(1)

        # 2. 读 builder.bin_edges_（缺失则用当下数据兜底 fit）。
        # 缺失场景：服务器旧 yl_tokenizer.json（在新 serializer 落地前保存的）
        # 没带 fitted_state.bin_edges → from_file 时 builder 还是新的、没 fit。
        # 这里 JIT-fit 一次：从当前 tokenize 的列重新学 quantile 边界，
        # 与 build_vocab 里 fit 阶段的行为一致；词表大小仍以 self.num_bins 为准。
        if getattr(self.builder, "bin_edges_", None) is None:
            self._jit_fit(vals)
        bin_edges = _normalize_bin_edges(self.builder.bin_edges_)
        # 单特征 KBinsDiscretizer（这是我们的情况——只 tokenize 一个数值列）：
        edges_for_this = bin_edges[0]
        # NaN → 落到桶 0；按 sklearn 规则 digitize(vals+eps, edges[1:]) 然后 clip。
        eps = 1e-8
        masked = _np.where(_np.isnan(vals), -_np.inf, vals)
        ids = _np.digitize(masked + eps, edges_for_this[1:])
        ids = _np.clip(ids, 0, self.num_bins - 1).astype("int32")
        # NaN 的样本：digitize(-inf) 会落到桶 0；显式补 0 防异常
        ids[_np.isnan(vals)] = 0

        # 3. 映射到 token 字符串，包成 cudf Series 返回（保持原 index 对齐）
        import pandas as _pd
        if hasattr(column_data, "index"):
            idx = column_data.index
            if hasattr(idx, "to_pandas"):
                idx = idx.to_pandas()
        else:
            idx = _pd.RangeIndex(len(ids))
        token_strs = _pd.Series(
            [self._idx_to_token.get(int(i), f"{self.special_token}_0") for i in ids],
            index=idx,
        )
        if cudf is None:
            # CPU-only 回退（如单测）：直接返 pandas Series
            return token_strs
        _from_pandas = getattr(cudf.Series, "from_pandas", None)
        return _from_pandas(token_strs) if _from_pandas else cudf.Series(token_strs)

    def _jit_fit(self, vals: "np.ndarray") -> None:
        """bin_edges_ 缺失时用当下数据兜底 fit。用于兼容旧 yl_tokenizer.json
        （新 serializer 落地前生成的 state 没带 bin_edges → builder 没恢复）。

        学到的边界可能与 train split 时的略有偏差，但仅影响 Route C 单笔 txn
        的 amount 桶；不会影响 Route A 训练（那边走 corpus text，不经此路径）。
        """
        import numpy as _np
        clean = vals[~_np.isnan(vals)]
        if len(clean) == 0:
            clean = _np.array([0.0, 1.0])
        try:
            # 优先用 cuML/sklearn builder.fit 自带的逻辑（保证 bin_edges_ 形态一致）
            self.builder.fit(_np.asarray(clean).reshape(-1, 1))
        except Exception:
            # 极端回退：手算 quantile 边界
            qs = _np.linspace(0, 1, self.num_bins + 1)
            self.builder.bin_edges_ = _np.quantile(clean, qs)


def _normalize_bin_edges(bin_edges_) -> list:
    """把 cuML / sklearn 的 KBinsDiscretizer.bin_edges_ 归一化成 List[ndarray]。

    sklearn 标准形态：``List[ndarray]``（每特征一个 1D 边界数组）；
    cuML 单特征下偶发返回 flat 1D ndarray，``bin_edges[jj][1:]`` 直接 IndexError。
    这里返回 ``list of np.ndarray``，让下游索引统一。
    """
    import numpy as _np
    if isinstance(bin_edges_, (list, tuple)):
        return [_np.asarray(c) for c in bin_edges_]
    arr = _np.asarray(bin_edges_)
    if arr.ndim == 1:
        # 单特征被拍平
        return [arr]
    if arr.dtype == object:
        return [c if isinstance(c, _np.ndarray) else _np.asarray(c) for c in arr]
    # 2D 纯数值：每行是一个特征的 edges
    return [_np.asarray(row) for row in arr]

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
        # 缺 bin_edges（旧 state 兼容）：仅标记 vocab built，但 builder 仍未 fit。
        # tokenize() 现在会在 bin_edges_ 缺失时 _jit_fit 重新补上，所以这条
        # "残缺 state 路径"也能跑通——但更好做法是 step_02 --force 重新保存。
        import logging as _logging
        import numpy as _np
        _log = _logging.getLogger(__name__)
        edges_raw = state.get("bin_edges")
        if edges_raw is not None:
            # 还原成 List[ndarray]（sklearn/cuML 都接受的 bin_edges_ 形态）
            rows = [_np.asarray(r, dtype="float64") for r in edges_raw]
            assigned = False
            for assignment in (
                lambda: (_np.array(rows, dtype=object) if len(rows) > 1 else rows[0]),
                lambda: rows,
                lambda: rows[0],
            ):
                try:
                    self.builder.bin_edges_ = assignment()
                    assigned = True
                    break
                except Exception:
                    continue
            if not assigned:
                # 不再 except:pass 静默——让 from_file 露出问题，由 tokenize 的 _jit_fit 兜
                _log.warning(
                    "NumericalTokenizerOptBin: builder.bin_edges_ 赋值失败"
                    "（只读属性/形态不匹配），tokenize 阶段会用数据 JIT-fit 重建。"
                    " 建议重跑 step_02 --force 让 serializer 重新保存。"
                )
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
                pass  # n_bins_ 冗余，不影响 transform
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
