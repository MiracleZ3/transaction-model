"""GPU ↔ CPU 数组适配层（cupy / cuml / numba DeviceNDArray / cuDF / pandas / numpy 互通）。

为什么需要这个模块：
  cupy.ndarray 的 ``__array__`` 会主动 raise
    "Implicit conversion to a NumPy array is not allowed. Please use .get() ..."
  防止静默把大头数据从 GPU 拷到 host。所以 ``np.asarray(cupy_arr)`` 直接崩。
  连环 bug 已经发生多次（timedelta.boundaries、numerical.bin_edges_、各种
  cuDF Series 派生）——这里统一一个入口，所有调 ``np.*`` 之前用 ``to_cpu_numpy``
  包一层。

API：
    to_cpu_numpy(x, dtype=None) -> np.ndarray   # 任意 array-like → host float64/int/...

设计原则：
  1. **显式 .get() 优先**：cupy / numba DeviceNDArray 都暴露 ``.get()`` /
     ``.copy_to_host()``；先摘到 host 再做转换。
  2. **cuDF / pandas Series/Index 走 to_pandas().to_numpy()**：避免 cuDF 24.x
     Series ``.values`` 的 cupy-only 路径在 object dtype 上 TypeError。
  3. **最后一道兜底 ``np.array(x, dtype)``**：兼容纯 python list / tuple / scalar
     / 1D ndarray / 2D ndarray。

适用范围：cuML estimator 属性（如 ``KBinsDiscretizer.bin_edges_``）、cuDF
derived Series / Index、cupy 设备张量、numba DeviceNDArray、以及普通 numpy /
list。错误信息友好——若实在变不出 ndarray，抛出本身的类型让上游决策。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def to_cpu_numpy(x: Any, dtype: Any = None) -> np.ndarray:
    """把任意 array-like 拉到 host numpy ndarray。

    Args:
        x: cupy / numba DeviceNDArray / cuML wrapper / cuDF 或 pandas Series/Index
           / numpy ndarray / python list / tuple / scalar。
        dtype: 目标 dtype；None 则沿用源。

    Returns:
        host numpy ndarray。

    Raises:
        TypeError: x 完全不像数组时（让上游决定怎么报错）。
    """
    # 1. cupy / numba DeviceNDArray：先显式 .get() / .copy_to_host()
    #    （np.asarray(cupy) 会因 __array__ 守卫直接抛错）
    if hasattr(x, "get") and callable(getattr(x, "get")):
        try:
            x = x.get()
        except Exception:
            pass  # 罕见：方法不是我们想要的 .get()，后续分支兜
    elif hasattr(x, "copy_to_host") and callable(getattr(x, "copy_to_host")):
        try:
            x = x.copy_to_host()
        except Exception:
            pass

    # 2. cuDF / pandas Series / Index：走 to_pandas().to_numpy() 路径
    if hasattr(x, "to_pandas") and callable(getattr(x, "to_pandas")):
        try:
            pd_x = x.to_pandas()
            return pd_x.to_numpy(dtype=dtype) if dtype is not None else pd_x.to_numpy()
        except Exception:
            x = x.to_pandas()  # 进一步退到纯 host 后再处理

    # 3. 普通路径：numpy / list / tuple / scalar
    if dtype is not None:
        return np.array(x, dtype=dtype)
    # 若 x 已经是 ndarray（避免拷贝）
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def is_cupy_array(x: Any) -> bool:
    """快速判别 x 是否是 cupy / numba 设备对象（含 ``.get``）。"""
    return (hasattr(x, "get") and callable(getattr(x, "get"))) or \
           (hasattr(x, "copy_to_host") and callable(getattr(x, "copy_to_host")))
