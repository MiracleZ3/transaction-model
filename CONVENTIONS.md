# 编码规范 — transaction-model

> 本文档是后续编码 agent 执行任务时的权威参考。所有新增/修改代码必须遵循本文档中的规范。

---

## 1. 项目概况

本项目是一个 Python 包 (`transaction_model`)，从 NVIDIA transaction-foundation-model 蓝图重构而来。包内包含两个代码子文化：

| 特征 | NVIDIA 迁移代码 | 项目新增代码 |
|------|----------------|-------------|
| 位置 | `tokenizer/`, `inference/decoder_inference.py`, `training/clm_data.py` | `data/`, `detection/`, `corpus/`, `visualization/`, `training/train.py`, `scripts/` |
| 许可证头 | 保留 SPDX Apache-2.0 | 不添加 |
| 语言 | 英文 | 中文 (docstring, print 输出) |
| cuDF 导入 | 直接 `import cudf` (必须 GPU) | `try/except` 回退 pandas |

**规则**: 修改 NVIDIA 迁移文件时保持原有风格不变；新增文件一律遵循本规范。

---

## 2. 文件结构规范

### 2.1 模块文件头部

```python
"""模块的简短中文描述 (一行)"""
from __future__ import annotations

import stdlib_module
from pathlib import Path

import third_party_module
import numpy as np

from transaction_model.config import load_config, resolve_path
from transaction_model.data.loader import load_parquet
```

四组 import 之间各空一行，顺序：
1. `from __future__ import annotations` — **所有新增文件必须添加**
2. 标准库 (`time`, `pathlib`, `os`, `json`, `argparse`, `subprocess` ...)
3. 第三方 (`numpy`, `pandas`, `torch`, `sklearn`, `xgboost` ...)
4. 项目内部 (`transaction_model.xxx` 或 `.xxx`)

### 2.2 包的 `__init__.py`

- 普通子包: 空文件
- 顶层包 `transaction_model/__init__.py`: 仅含 `__version__`
- 需要对外暴露的包 (如 `tokenizer/`): 使用 re-export + `__all__`

```python
# transaction_model/tokenizer/__init__.py
from .financial_tokenizer import FinancialTabularTokenizer
from .financial_pipeline import FinancialTokenizerPipeline

__all__ = ["FinancialTabularTokenizer", "FinancialTokenizerPipeline"]
```

### 2.3 文件命名

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | `snake_case.py` | `loader.py`, `feature.py`, `xgboost.py` |
| 脚本 | `step_XX_descriptive.py` | `step_01_dataset_baseline.py` |
| 配置 | `snake_case.yaml` | `dataset.yaml`, `xgboost.yaml` |
| 测试 | `test_<module>.py` | `test_loader.py` |

---

## 3. 类型标注规范

### 3.1 统一使用现代语法 (PEP 604 / PEP 585)

由于所有新增文件都有 `from __future__ import annotations`，使用现代类型语法：

```python
# 正确
def load_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
def extract_split_embeddings(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
def run_umap_2d(labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
```

```python
# 错误 (仅在修改 NVIDIA 迁移文件时可接受)
from typing import Optional, List, Dict, Union
def load_parquet(path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
```

### 3.2 所有公开函数必须有返回类型标注

```python
# 正确
def load_config(name: str) -> dict[str, Any]: ...
def print_dataset_summary(df) -> None: ...
def temporal_split(gdf, train_ratio: float = 0.8) -> tuple: ...

# 错误 — 缺少返回类型
def load_config(name: str): ...
```

参数类型标注尽量完整，但允许对动态类型参数 (如 cuDF/pandas 混合的 `df`) 省略。

---

## 4. 文档字符串规范

### 4.1 模块级 docstring

每个 `.py` 文件的第一行必须是模块 docstring，中文，一行：

```python
"""数据加载工具（cuDF GPU 加速 + pandas 回退）"""
```

### 4.2 函数级 docstring — Google 风格 + 中文

```python
def load_raw_csv(csv_path: str | Path, use_gpu: bool = True) -> pd.DataFrame:
    """加载原始 CSV，优先使用 cuDF GPU 加速

    Args:
        csv_path: CSV 文件路径
        use_gpu: 是否使用 cuDF GPU 加速

    Returns:
        DataFrame (cuDF 或 pandas)
    """
```

关键规则：
- 简短描述用中文
- `Args:` 下每个参数一行，`参数名: 说明`
- `Returns:` 描述返回值
- 需要时加 `Raises:` 和 `Note:`

---

## 5. 配置驱动模式

### 5.1 常量归属

| 放在 `constants.py` | 放在 `configs/*.yaml` |
|---------------------|----------------------|
| 运行时不变的映射表 (MCC_INDUSTRY_RANGES) | 文件路径 |
| 列名常量 (FRAUD_COL) | 超参数 (learning_rate, batch_size) |
| 可视化固定参数 (UMAP_N_NEIGHBORS) | 采样大小、分割比例 |
| 项目根路径 (PROJECT_ROOT) | 特征列列表 |

### 5.2 配置加载模式

```python
from transaction_model.config import load_config, resolve_path

# 加载配置
cfg = load_config("dataset")          # configs/dataset.yaml
tok_cfg = load_config("tokenizer")    # configs/tokenizer.yaml

# 读取配置值
raw_csv = cfg["dataset"]["raw_csv"]

# 路径必须通过 resolve_path 解析
csv_path = resolve_path(raw_csv)       # 相对路径 → 绝对路径
```

**规则**: 禁止硬编码路径。所有路径从 YAML 读取，通过 `resolve_path()` 解析。

### 5.3 YAML 配置风格

```yaml
# snake_case 键名
merchant_hash_size: 2000
context_window: 4096

# 注释说明枚举值和默认推断
device: "auto"  # auto / cuda / cpu
# test_ratio 自动推断为 1 - train - val

# 路径使用相对路径 (基于项目根)
raw_csv: "data/TabFormer/raw/card_transaction.v1.csv"
```

---

## 6. GPU/CPU 自适应模式

### 6.1 数据加载层 — `try/except ImportError` 模式

```python
if use_gpu:
    try:
        import cudf
        # GPU 路径
        return cudf.read_csv(str(csv_path))
    except ImportError:
        print("cuDF not available, falling back to pandas")

# CPU 回退
return pd.read_csv(csv_path)
```

### 6.2 DataFrame 兼容 — `hasattr(obj, 'to_pandas')` 模式

任何需要从可能是 cuDF 的 DataFrame 提取数据给 pandas/sklearn 的地方：

```python
if hasattr(gdf, 'to_pandas'):
    pdf = gdf.to_pandas()
else:
    pdf = gdf
```

**注意**: 在 cuDF 上直接调用 `.to_pandas()` 是 cuDF 的方法；pandas DataFrame 没有此方法会报 `AttributeError`。

### 6.3 ML 库回退 — 返回 (类, is_gpu) 元组模式

```python
def _get_umap():
    try:
        from cuml.manifold import UMAP
        return UMAP, True
    except ImportError:
        from sklearn.manifold import UMAP
        return UMAP, False
```

### 6.4 XGBoost 设备选择模式

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
```

---

## 7. 输出格式规范

### 7.1 使用 `print()` 而非 `logging`

所有数据管道、检测、可视化模块统一使用 `print()` 输出进度信息。仅 `decoder_inference.py` 使用 `logging`。

### 7.2 进度输出格式

```python
# 阶段标题: = 分隔线
print(f"\n{'='*60}")
print(f"Extracting {split} embeddings")
print(f"{'='*60}")

# 子项信息: 两空格缩进
print(f"  Loaded {len(gdf):,} rows in {time.time()-t0:.1f}s")
print(f"  Model loaded on {inference.device} (embed_dim={inference.embedding_dim})")

# 完成总结
print(f"  Generated {len(corpus_lines):,} sequences in {elapsed:.1f}s")
print(f"  Saved to: {corpus_path}")
```

### 7.3 数字格式化

| 场景 | 格式 | 示例输出 |
|------|------|---------|
| 数量 | `{n:,}` | `1,000,000` |
| 耗时 | `{t:.2f}s` | `3.14s` |
| 百分比 | `{rate:.4%}` 或 `{rate:.4f}` | `12.34%` 或 `0.1234` |
| 右对齐表列 | `{n:>12,}` | `   1,000,000` |
| 左对齐表列 | `{label:<35}` | `Raw Features (baseline)       ` |

### 7.4 表格输出格式

```python
print(f"{'Split':<8} {'Rows':>12} {'%':>7} {'Fraud':>8}")
print("-" * 40)
for name, gdf in [('Train', train_gdf), ('Val', val_gdf), ('Test', test_gdf)]:
    print(f"{name:<8} {n:>12,} {pct:>6.2f}% {fraud:>8,}")
```

---

## 8. 性能计时模式

统一使用 `time.time()` (wall-clock)，不使用 `time.perf_counter()` 或上下文管理器：

```python
t0 = time.time()
# ... 执行操作 ...
print(f"Operation time: {time.time()-t0:.2f}s")
```

---

## 9. 路径处理规范

- 函数签名接受 `str | Path`，进入函数体第一件事转 `Path`: `csv_path = Path(csv_path)`
- 配置中的路径使用 `resolve_path()` 解析
- 目录创建使用 `mkdir(parents=True, exist_ok=True)`

```python
def save_splits(train_gdf, val_gdf, test_gdf, output_dir: str | Path) -> Path:
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ...
```

---

## 10. 断点续跑 / 缓存模式

耗时操作（语料生成、嵌入提取）支持跳过已存在结果：

```python
if output_path.exists() and not force:
    result = load_cached(output_path)
    print(f"[{split_name}] Already exists: {summary}")
    return result

# ... 执行计算 ...

output_path.parent.mkdir(parents=True, exist_ok=True)
save_result(output_path, result)
```

所有支持此模式的函数必须提供 `force: bool = False` 参数。

---

## 11. 错误处理规范

### 11.1 前置条件检查 — `assert`

用于不可恢复的配置/依赖错误：

```python
assert train_corpus.exists(), f"Training corpus not found: {train_corpus} (run step_02 first)"
```

### 11.2 可预期的运行时错误 — `raise`

```python
if not csv_path.exists():
    raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
```

### 11.3 静默容错 — 仅用于可选功能

```python
try:
    n_users = df['User'].nunique()
    print(f"  Users: {n_users:,}")
except Exception:
    pass  # 可选统计信息，失败不影响流程
```

---

## 12. 函数签名规范

### 12.1 参数顺序

```
def function(
    主要数据参数,          # 路径、DataFrame、数组
    配置/选项参数,         # hyperparameters, flags
    random_state=42,      # 可重现性参数放最后
):
```

### 12.2 常见默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `random_state` | `42` | 全局一致 |
| `use_gpu` | `True` | 优先 GPU |
| `force` | `False` | 不强制重生成 |
| `batch_size` | `1024` | 推理批次 |
| `merchant_hash_size` | `2000` | 商户哈希空间 |

---

## 13. 返回值规范

### 13.1 简单结果 — `dict`

```python
return {"auc": 0.95, "ap": 0.72}
```

### 13.2 多值 — `tuple` (带 docstring 说明)

```python
return train_gdf, val_gdf, test_gdf, train_cutoff, test_cutoff

return X, y, sampled_idx

return clf, {"val_auc": 0.95, "test_auc": 0.94}
```

当 tuple 超过 3 个元素时，考虑改用 `dataclass` 或 `NamedTuple`。

### 13.3 复杂结果 — 嵌套 `dict`

```python
return {
    "baseline": metrics_baseline,
    "embed": metrics_embed,
    "combined": metrics_combined,
    "clfs": {"baseline": clf_baseline, ...},
    "metadata": {...},
}
```

---

## 14. CLI 脚本规范

### 14.1 标准模板

```python
"""Step XX: 中文简短描述"""
from __future__ import annotations

import argparse

from transaction_model.xxx import yyy


def main():
    parser = argparse.ArgumentParser(description="Step XX: Description")
    parser.add_argument("--force", action="store_true", help="强制重新生成")
    args = parser.parse_args()

    # ... 逻辑 ...


if __name__ == "__main__":
    main()
```

### 14.2 常用 CLI 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--force` | `store_true` | 强制重新生成 |
| `--demo` | `store_true` | 快速演示模式 |
| `--skip-xxx` | `store_true` | 跳过某步骤 |
| `--no-plot` | `store_true` | 跳过可视化 |
| `--max-steps` | `int` | 覆盖训练步数 |
| `--nproc` | `int`, default=1 | GPU 数量 |

### 14.3 可重现性初始化

Step 脚本入口处设置全局随机种子：

```python
import random
import numpy as np
import torch

RANDOM_STATE = cfg["sampling"]["random_state"]
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
```

---

## 15. NeMo 集成规范

### 15.1 `_target_` 路径格式

YAML 中使用 `module_path:function_name` 格式：

```yaml
dataset:
  _target_: transaction_model.training.clm_data:build_financial_clm_dataset
```

### 15.2 `sys.path` 操纵

仅 `clm_data.py` 和 `run_training.py` 需要，确保 NeMo 的文件路径解析可找到包：

```python
_pkg_root = str(Path(__file__).resolve().parent.parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
```

### 15.3 `**kwargs` 前向兼容

NeMo 入口函数必须接受 `**kwargs`：

```python
def build_financial_clm_dataset(data_path: str, **kwargs) -> FinancialCLMDataset:
```

---

## 16. 重构检查清单

修改代码时对照以下清单：

- [ ] 新文件是否添加了 `from __future__ import annotations`
- [ ] 所有公开函数是否有 Google 风格中文 docstring
- [ ] 路径是否通过 `resolve_path()` 解析而非硬编码
- [ ] DataFrame 操作是否兼容 cuDF/pandas (`hasattr(obj, 'to_pandas')`)
- [ ] 数字输出是否使用千分位 (`{n:,}`) 和适当精度
- [ ] 计时是否使用 `t0 = time.time()` 模式
- [ ] 随机操作是否接受 `random_state` 参数
- [ ] 耗时操作是否支持断点续跑 (`force` 参数)
- [ ] import 是否按 stdlib / third-party / local 三组排列
- [ ] 类型标注是否使用现代语法 (`str | None`, `list[str]`)
