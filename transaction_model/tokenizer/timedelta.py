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
Log-compressed time-delta tokenizer (GPU-accelerated).

Bins time differences (in seconds) into logarithmically-spaced buckets.
Useful for capturing recency signals — small deltas (seconds/minutes) get
fine-grained bins while large deltas (months) share coarser bins.

Data is never stored in the constructor.
"""
from __future__ import annotations

try:
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    cudf = None  # type: ignore
try:
    import cupy as cp
except ImportError:  # pragma: no cover - cuPy optional on CPU
    cp = None  # type: ignore
import numpy as np

import pandas as pd

from .base import BaseTokenizer

_SECONDS_PER_JULIAN_YEAR = 31556951.999999996


class TimeDeltaTokenizer(BaseTokenizer):
    """Log-scale binning for time deltas (in seconds)."""

    def __init__(
        self,
        num_bins: int = 32,
        special_token: str = "TDIF",
        max_years: float = 10.0,
        stream: cp.cuda.Stream = None,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.special_token = special_token
        self.max_years = max_years
        self.stream = stream

        self.max_horizon = int(max_years * _SECONDS_PER_JULIAN_YEAR)
        if cp is None:
            raise ImportError(
                "TimeDeltaTokenizer requires the 'cupy' package "
                "(GPU only). Install with: pip install cupy"
            )
        log_max = cp.log(float(self.max_horizon) + 1.0)
        self.boundaries = cp.linspace(0, log_max, self.num_bins + 1)
        self._vocab_built = False

    def build_vocab(self, column_data=None) -> None:
        if self.stream:
            with self.stream:
                self._idx_to_token = {
                    i: f"{self.special_token}_{i}" for i in range(self.num_bins)
                }
        else:
            self._idx_to_token = {
                i: f"{self.special_token}_{i}" for i in range(self.num_bins)
            }
        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        if self.stream:
            with self.stream:
                return self._tokenize_internal(column_data)
        return self._tokenize_internal(column_data)

    def _tokenize_internal(self, column_data) -> cudf.Series:
        clamped = column_data.clip(0, self.max_horizon)
        # 统一拉回 host float64 计算 log/digitize（cupy 和 numpy 路径都走 host 控件，
        # 小算量上 host 更稳；避免 pandas 列没 cupy 转换路径、cuDF 24.x .values trap）。
        # ⚠️ 关键：不能 np.asarray(cupy_array)！cupy 会通过 __array__ 直接抛
        # "Implicit conversion to a NumPy array is not allowed"。必须走 .get() /
        # cp.asnumpy() / .to_numpy() 显式拷出。
        if hasattr(clamped, "to_pandas"):
            host_arr = clamped.to_pandas().to_numpy(dtype="float64")
        else:
            host_arr = _to_cpu_numpy(clamped, dtype="float64")
        boundaries = _to_cpu_numpy(self.boundaries)   # self.boundaries 是 cupy 数组
        log_vals = np.log(host_arr + 1.0)
        token_ids = np.clip(
            np.digitize(log_vals, boundaries), 0, self.num_bins - 1
        ).astype("int64")
        # 包装回 cuDF Series 并映射 token 字符串（cupy/numpy 步骤已统一）
        ids_series = pd.Series(token_ids, index=_host_index(column_data.index))
        _from_pandas = getattr(cudf.Series, "from_pandas", None)
        cu_ids = _from_pandas(ids_series) if _from_pandas else cudf.Series(ids_series)
        return cu_ids.map(self._idx_to_token)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"TimeDeltaTokenizer(token={self.special_token}, "
            f"bins={self.num_bins}, {status})"
        )


def _host_index(idx) -> "pd.Index":
    """统一把 cuDF / pandas / RangeIndex 摘到 host pandas Index。"""
    if hasattr(idx, "to_pandas"):
        return idx.to_pandas()
    return pd.Index(idx)


def _to_cpu_numpy(x, dtype=None) -> "np.ndarray":
    """Deprecated alias kept for backward-compat. Delegates to
    ``transaction_model._gpu_numpy.to_cpu_numpy``（单一真源）。

    为什么不用 np.asarray(x)：cupy.ndarray 的 __array__ 会 raise
    'Implicit conversion to a NumPy array is not allowed. Please use .get() ...'。
    """
    from transaction_model._gpu_numpy import to_cpu_numpy
    return to_cpu_numpy(x, dtype=dtype)

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "num_bins": self.num_bins,
            "special_token": self.special_token,
            "max_years": self.max_years,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {
            "boundaries": (
                self.boundaries.get()
                if isinstance(self.boundaries, cp.ndarray)
                else self.boundaries
            ),
            "max_horizon": self.max_horizon,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        self.boundaries = cp.array(state["boundaries"])
        self.max_horizon = state["max_horizon"]
        self._vocab_built = state.get("_vocab_built", False)
