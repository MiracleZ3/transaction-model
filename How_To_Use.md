# Transaction Model 使用指南

本指南覆盖金融交易基础模型（Financial Transaction Foundation Model）的**完整生命周期**：环境准备 → 数据处理 → 分词 → 训练 → 嵌入提取 → 欺诈检测 → 评估与决策。第 [8 节](#8-是否需要重新预训练决策树)给出"是否需要重训"的判断框架。

> 项目结构与编码规范请参考 [`README.md`](README.md) 与 [`CONVENTIONS.md`](CONVENTIONS.md)。本指南只关注**怎么用**。

> **银联风控 NDJSON 用户请先读 §13**：本指南 §3–§7 主要覆盖 **TabFormer CSV** 路径。
> 银联 `risk_control_2` 风格 NDJSON（每行一个用户的 `{"cert_sm3", "trans": [[...20+字段...]], "label"}`）
> 走两条独立路线，命令入口与配置文件都与 TabFormer 不同：
> - **路线 A**（NDJSON → tokenized corpus → Llama decoder CLM → 嵌入 → XGBoost）：`step_01b_load_ndjson.py`
>   → `step_02_tokenize_ndjson.py` → `step_03_train_model.py --variant yl` → `step_04_extract_embeddings.py --dataset-config dataset_yl`
>   → `step_05_fraud_detection.py --dataset-config dataset_yl`。详见 [`upgrade/ylformer.md`](upgrade/ylformer.md)。
> - **路线 C**（Llama route-A ckpt + GPT2 cross-txn + 分类头 + LoRA + 业务损失）：
>   `step_06_finetune_routec.py --config configs/routec/default.json`。详见本指南 [§13](#13-route-c--llamagpt2-业务损失微调)。
> 冒烟数据见 [`examples/sample_data/smoke.jsonl`](examples/sample_data/smoke.jsonl)。

---

## 目录

1. [生命周期总览](#1-生命周期总览)
2. [环境准备与安装](#2-环境准备与安装)
3. [Step 1 — 数据准备与基线](#3-step-1--数据准备与基线)
4. [Step 2 — 语料库生成](#4-step-2--语料库生成)
5. [Step 3 — 模型训练 / 续训](#5-step-3--模型训练--续训)
6. [Step 4 — 嵌入向量提取](#6-step-4--嵌入向量提取)
7. [Step 5 — 欺诈检测评估](#7-step-5--欺诈检测评估)
8. [是否需要重新预训练（决策树）](#8-是否需要重新预训练决策树)
9. [一键全流程](#9-一键全流程)
10. [常见问题与故障排查](#10-常见问题与故障排查)
11. [清理与重置](#11-清理与重置)
12. [在容器中运行](#12-在容器中运行)
13. [Route C — Llama+GPT2 业务损失微调](#13-route-c--llamagpt2-业务损失微调)

---

## 1. 生命周期总览

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Step 1     │   │ Step 2     │   │ Step 3     │   │ Step 4     │   │ Step 5     │
│ 数据基线   │ → │ 语料生成   │ → │ 模型训练   │ → │ 嵌入提取   │ → │ 欺诈检测   │
│ (CPU/GPU)  │   │ (CPU)      │   │ (GPU)      │   │ (GPU)      │   │ (CPU/GPU)  │
└────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
   parquet          *.txt          checkpoints      *.npy          metrics+图表
```

| 阶段 | 输入 | 输出 | 是否必选 | 典型耗时 |
|------|------|------|----------|---------|
| Step 1 | 原始 CSV | `temporal_split/*.parquet`、基线 AUC | 必选 | 5–30 min |
| Step 2 | parquet | `data/decoder_corpus/*.txt` | 必选 | 1–10 min |
| Step 3 | train/val 语料 | `models/decoder-*/checkpoints/` | 训练时必选；推理可复用预训练模型 | 单卡 demo 1 min；全量训练 8×GPU 数小时 |
| Step 4 | 预训练模型 + parquet | `data/embeddings/*.npy` | 必选 | 5–30 min |
| Step 5 | embeddings + 评估 parquet | ROC-AUC / AP / 对比图 | 必选 | < 1 min |

**已训练过的模型可跳过 Step 3**，直接复用 `models/decoder-foundation-model/`。

---

## 2. 环境准备与安装

### 2.1 硬件要求

| 用途 | 最低配置 | 推荐 |
|------|---------|------|
| 数据管道 + XGBoost + 可视化 | 任意 CPU，16 GB RAM | — |
| 嵌入提取 / Demo 训练 | 1× GPU, ≥16 GB 显存 | 1× A100 / RTX 4090 |
| 全量训练 | 1× GPU | 8× A100 80GB + FSDP2 |

### 2.2 安装命令

```bash
git clone https://github.com/MiracleZ3/transaction-model.git
cd transaction-model

# 基础安装（数据 + 检测 + 可视化）
pip install -e .

# GPU 加速（cuDF 数据加载 + cuML UMAP）
# RAPIDS wheel 只发布在 NVIDIA 索引上，必须带 --extra-index-url
pip install -e ".[gpu]" --extra-index-url https://pypi.nvidia.com

# NeMo 训练框架（仅 Step 3 需要）
pip install -e ".[nemo]"

# 开发与测试
pip install -e ".[dev]"
```

### 2.3 验证安装

```bash
python -c "import transaction_model; print(transaction_model.__version__)"
pytest tests/ -v          # 可选：跑测试
```

### 2.4 预训练模型准备（仅推理场景）

若**只做嵌入提取 + 欺诈检测、不做训练**，将上游提供的 decoder 模型放置到：

```
transaction-model/
└── models/
    └── decoder-foundation-model/     # 包含 config.json + *.safetensors
```

之后可跳过 Step 3，直接执行 `make all-pretrained`。

---

## 3. Step 1 — 数据准备与基线

**目的**：下载/加载原始交易数据 → 时间分割 → 特征工程 → XGBoost 原始特征基线。

### 3.1 执行

```bash
# 完整流程
python scripts/step_01_dataset_baseline.py

# 跳过下载（已有 raw CSV）
python scripts/step_01_dataset_baseline.py --skip-download

# 只做数据准备，不训基线（节省时间）
python scripts/step_01_dataset_baseline.py --skip-baseline
```

### 3.2 产出

```
data/TabFormer/temporal_split/
├── train.parquet         # 时间分割后的训练集
├── val.parquet
├── test.parquet
├── val_eval.parquet      # 配置 sampling.eval_samples 的评估子集
└── test_eval.parquet
```

### 3.3 关键配置（`configs/dataset.yaml`）

```yaml
dataset:
  raw_csv: "data/TabFormer/raw/card_transaction.v1.csv"
  temporal_split_dir: "data/TabFormer/temporal_split"

split:
  train_ratio: 0.8         # 80% 训练
  val_ratio: 0.1           # 10% 验证；test_ratio 自动推断

sampling:
  balanced_train_size: 1000000    # 训练样本数
  eval_samples: 100000            # 评估样本数
  random_state: 42
```

### 3.4 使用自有数据

替换 `dataset.raw_csv` 与 `feature_cols`；若列名与 TabFormer 不同，需要按 [`README.md`](README.md#3-适配数据加载器) 修改 `constants.py`、`data/feature.py`、`data/split.py` 等适配点。

---

## 4. Step 2 — 语料库生成

**目的**：将流水线化表格 → 金融领域 token 序列，喂给 decoder 模型。

### 4.1 执行

```bash
python scripts/step_02_tokenize_corpus.py            # 增量（已存在则跳过）
python scripts/step_02_tokenize_corpus.py --force    # 强制重生
```

### 4.2 产出

```
data/decoder_corpus/
├── train_corpus.txt      # 每行一个 <bos> ... <eos> 序列（约 315 笔交易/行）
├── val_corpus.txt
└── test_corpus.txt
```

### 4.3 关键配置（`configs/tokenizer.yaml`）

```yaml
tokenizer:
  merchant_hash_size: 2000    # 商户哈希空间；商户数多时按 log2(商户数) 估计
  chunk_size: 315             # 每序列交易数
  context_window: 4096        # 与 training.yaml 的 seq_length 对齐
  tokens_per_txn: 12          # 每笔交易约 12 tokens
```

### 4.4 复用提示

- 此步骤**与模型无关**，新增模型架构时**不必重跑**——语料可缓存复用。
- 但若 `merchant_hash_size` 调整，文本中的 merchant token ID 会变化，需 `--force` 重新生成。

---

## 5. Step 3 — 模型训练 / 续训

**目的**：在金融语料上做因果语言建模（CLM），训练 Llama 架构 decoder。

### 5.1 三种运行模式

```bash
# A. Demo（单卡，30 步，验证训练流水线可用性）
python scripts/step_03_train_model.py --demo

# B. 单卡完整训练
python scripts/step_03_train_model.py --max-steps 3000

# C. 多卡分布式训练（推荐 8 卡，配合 FSDP2）
python scripts/step_03_train_model.py --nproc 8 --max-steps 3000

# D. 从预训练 checkpoint 续训
torchrun --nproc-per-node=8 scripts/step_03_train_model.py \
    -c configs/training.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt \
    --validation_dataset.data_path data/decoder_corpus/val_corpus.txt
```

### 5.2 关键配置（`configs/training.yaml`）

```yaml
model:
  config:
    vocab_size: 6251              # 必须与 tokenizer 输出匹配
    hidden_size: 512              # 可增至 768 / 1024 提升容量
    num_hidden_layers: 8
    num_attention_heads: 8
    num_key_value_heads: 2        # GQA 4:1
    max_position_embeddings: 8192 # RoPE 外推

step_scheduler:
  max_steps: 3000
  global_batch_size: 16
  local_batch_size: 16
  val_every_steps: 15
  ckpt_every_steps: 15

optimizer:
  lr: 0.0002
  weight_decay: 0.077

lr_scheduler:
  lr_decay_style: cosine
  lr_warmup_steps: 10

paths:
  pretrained_model: "models/decoder-foundation-model"   # 续训起点（不存在则从零训）
  train_corpus: "data/decoder_corpus/train_corpus.txt"
  val_corpus: "data/decoder_corpus/val_corpus.txt"
```

### 5.3 产出

```
models/decoder-demo/checkpoints/      # --demo 模式
├── step_*/mp_rank_*/model.pt
└── consolidated/safetensors/         # save_consolidated=true 时生成 HuggingFace 兼容格式
```

将 `consolidated/` 移动 / 软链到 `models/decoder-foundation-model/` 即可用于 Step 4。

### 5.4 训练完后的健康检查

| 指标 | 健康范围 | 异常处置 |
|------|---------|---------|
| Train loss | 单调下降至 ~1.x–2.x | 不下降 → 检查 `vocab_size` / 数据流 |
| Val loss | 与 train 接近，差距 < 30% | 严重发散 → 减小 lr 或加大 `weight_decay` |
| Grad norm | < 5 | 爆炸 → 降低 lr；消失 → 检查初始化 |
| Step time | 单卡 ~1–3 s/step (seq=4096) | 过慢 → 检查 dataloader workers / GPU 利用率 |

---

## 6. Step 4 — 嵌入向量提取

**目的**：用训练好的 decoder 提取每笔交易的 512-dimensional 嵌入（last-token pooling）。

> **银联 NDJSON 用户**：命令需加 `--dataset-config dataset_yl`，会从 `configs/dataset_yl.yaml`
> 推断 `tokenizer.variant: "yl"` + `state_path`，复用 YLPipeline fit 后保存的词表。
> 否则会回退到 FinancialTokenizerPipeline + TabFormer 词表，标签列也错位。

### 6.1 执行

```bash
python scripts/step_04_extract_embeddings.py            # 增量
python scripts/step_04_extract_embeddings.py --force    # 强制重提
```

### 6.2 产出

```
data/embeddings/
├── train_embeddings.npy        # (N_train, 512) float32
├── train_labels.npy
├── train_row_ids.npy           # __row_id__，与 parquet 行对齐
├── val_embeddings.npy
├── val_labels.npy
├── val_row_ids.npy
├── test_embeddings.npy
├── test_labels.npy
└── test_row_ids.npy
```

### 6.3 关键配置（`configs/xgboost.yaml`）

```yaml
inference:
  batch_size: 1024
  max_length: 128
  pooling: "last_token"      # 或 "mean"
  embed_dir: "data/embeddings"
```

### 6.4 标签对齐机制

`__row_id__` 在 preprocess 之前即被记录，确保即便 `preprocess` 重排/过滤列后，labels 仍可对齐到原始 parquet 行。**不要在提取过程中手动修改 parquet**。

---

## 7. Step 5 — 欺诈检测评估

> **银联 NDJSON 用户**：命令需加 `--dataset-config dataset_yl`。Baseline 列改成 14 个 `cups_*`
> 字段（见 `configs/dataset_yl.yaml::feature_cols`），由 OrdinalEncoder 处理字符串字段。
> 用户级 `label` 在 NDJSON 加载时已广播到每行；fraud_col 自动切换到 `"label"`。

**目的**：用 XGBoost 在三种特征空间对比欺诈检测效果。

### 7.1 执行

```bash
python scripts/step_05_fraud_detection.py
python scripts/step_05_fraud_detection.py --no-plot    # 跳过图表
```

### 7.2 三模型对比

| 模型 | 特征维度 | 说明 |
|------|---------|------|
| **Baseline** | 13d 原始 | OrdinalEncoder 编码后的表格特征 |
| **Embedding** | 64d（PCA from 512d） | 纯 decoder 嵌入 |
| **Combined** | 13d + 64d | 表格特征 + 嵌入，**通常最优** |

### 7.3 产出

- 终端打印：每个模型的 Val/Test **ROC-AUC、Average Precision**、训练耗时、`best_iteration`
- 可视化：`data/outputs/xgb_auc_ap_comparison.png`

### 7.4 解读经验

- `Combined > Baseline` 越明显，说明 decoder 学到了表格外的序列信息（用户行为模式）。
- `Embedding` 单独 vs `Baseline`：若 `Embedding` 显著低于 `Baseline`，可能 decoder 欠训或数据类型信号过强（建议检查 tokenizer 配置）。
- `Combined ≈ Baseline`：可考虑不重训，仅调表格超参。

---

## 8. 是否需要重新预训练（决策树）

**核心问题**：什么改动需要"从头训"，什么改动只需"续训"，什么改动根本不需要动模型？

### 8.1 决策树

```
你做了什么改动？
│
├── ❶ 只改下游（XGBoost 超参 / PCA 维度 / 采样数量 / 评估指标）
│      └─ 不需要重训。直接重跑 Step 4–5（如已改）或 Step 5。
│
├── ❷ 改了 tokenizer.yaml 的非词表参数（chunk_size / context_window / pooling）
│      └─ 不需要重训。只需 Step 2 --force + Step 4 --force + Step 5。
│         （词表 ID 不变，decoder 权重仍可用。）
│
├── ❸ 改了 merchant_hash_size / 新增特征列 / 词表扩张
│      └─ ❗ vocab_size 变化 → 至少需要**续训**。
│         建议：续训 500–1500 步，让 embedding layer 适应新 token。
│
├── ❹ 切换到新数据集（同领域，例如另一家银行的交易数据）
│      └─ **续训**。从 models/decoder-foundation-model 加载，
│         用新语料续训 1000–3000 步，监控 val loss 是否收敛。
│
├── ❺ 切换到**显著不同领域**（例如从信用卡到对公账户 / 跨境支付）
│      └─ **从零预训练**。鉴领域偏移太大，预训练权重可能误导。
│         先用 --demo 跑通流水，再放大 max_steps。
│
├── ❻ 改了模型架构（hidden_size / num_layers / attention 配置）
│      └─ **从零预训练**。权重 shape 对不齐，无法续训。
│
└── ❼ vocab_size 不变，但训练数据规模 ×10
       └─ 可选续训；先评估 Step 5 结果是否已饱和，饱和则不必。
```

### 8.2 判断"是否饱和"的指标

| 信号 | 判断 |
|------|------|
| **val loss 已平台期** | 训练 loss 仍下降但 val loss 不降 → 已饱和，更多数据收益小 |
| **Combined AUC 显著高于 Baseline** | decoder 还在"产生价值" → 可继续训 |
| **Embedding AUC ≈ Baseline** | decoder 信号弱 → 续训收益小，建议先排查 tokenizer |
| **train loss 持续下降但 val 上升** | 过拟合 → 减小 max_steps 或加大 weight_decay |

### 8.3 续训 vs 从零训的成本对比

| 项目 | 续训（推荐） | 从零训 |
|------|------------|--------|
| GPU 时长（8×A100） | 1–3 h | 8–24 h |
| 数据规模需求 | 新数据 10–100 万笔 | 旧数据 + 新数据 ≥ 1000 万笔 |
| 配置改动 | 改 `paths.pretrained_model` 指向现有 checkpoint | 删除 `paths.pretrained_model` 或置空 |
| 风险 | 低；最坏情况退化到原模型 | 中；需要重新调 lr / warmup |

### 8.4 重训练前的 Checklist

执行任何"动模型"决策前，请逐项确认：

- [ ] 已备份当前 `models/decoder-foundation-model/`（重命名加日期，例如 `decoder-2025-09-15/`）
- [ ] `configs/training.yaml` 的 `vocab_size` 与新 tokenizer 输出一致
- [ ] `paths.train_corpus` / `paths.val_corpus` 已更新到新数据路径
- [ ] `step_scheduler.max_steps` 已根据数据规模调整（经验：10⁶ 笔交易 ~3000 步）
- [ ] `step_scheduler.val_every_steps` 不至于拖慢训练（建议 15–50 步）
- [ ] 已用 `--demo` 模式跑通 30 步，loss 能正常下降
- [ ] GPU 显存、磁盘空间充足（每 checkpoint ~120 MB × `num_layers`）
- [ ] `data/embeddings/` 已用 `--force` 重提

---

## 9. 一键全流程

### 9.1 Pipeline 调度器（推荐）

```bash
# 全流程（含训练）
python scripts/run_pipeline.py --steps all

# 全流程（复用预训练模型，跳过 Step 3）
python scripts/run_pipeline.py --steps data,tokenize,extract,detect

# 只跑某几步
python scripts/run_pipeline.py --steps tokenize,extract,detect

# Demo 模式（训练步骤自动用 --demo）
python scripts/run_pipeline.py --steps all --demo

# 强制重生成（影响 tokenize / extract 步骤）
python scripts/run_pipeline.py --steps all --force
```

### 9.2 Make 快捷命令

```bash
make install              # pip install -e .
make data                 # Step 1
make tokenize             # Step 2
make train                # Step 3
make extract              # Step 4
make detect               # Step 5
make all                  # 全流程（含训练）
make all-pretrained       # 全流程（跳过训练，使用 models/decoder-foundation-model）
make clean                # 清理训练 / 嵌入 / 语料缓存（见第 11 节）
make test                 # pytest
```

---

## 10. 常见问题与故障排查

### 10.1 训练失败

| 报错 | 原因 | 处置 |
|------|------|------|
| `AssertionError: Training corpus not found` | 没跑 Step 2 | 先 `python scripts/step_02_tokenize_corpus.py` |
| `CUDA out of memory` | batch / seq 太大 | 减小 `local_batch_size`（如 16 → 4）或启用 gradient checkpointing |
| `vocab_size mismatch` | YAML 与实际 tokenizer 不一致 | 用 `FinancialTabularTokenizer(...).get_vocab_size()` 校准后改 `training.yaml` |
| `nemo_automodel not found` | 未安装 `[nemo]` | `pip install -e ".[nemo]"` |
| `loss = nan` | 数值不稳定 | 降低 lr（×0.1）或加大 `weight_decay` |

### 10.2 嵌入提取失败

| 报错 | 原因 | 处置 |
|------|------|------|
| `Model path does not exist` | `paths.pretrained_model` 未配置 | 放置模型或修改 `configs/training.yaml` |
| `embed_dim mismatch` | checkpoint 与配置 hidden_size 不一致 | 用对应 hidden_size 的 checkpoint |
| 标签数量与嵌入不一致 | parquet 改过 | `--force` 重新提取；不要在 extract 之间改 parquet |

### 10.3 数据 / 性能问题

| 现象 | 处置 |
|------|------|
| 数据加载很慢 | 安装 `[gpu]` 走 cuDF；或减弱 `sampling.balanced_train_size` |
| UMAP 内存爆 | 减小 `eval_samples` 到 50k 以下 |
| XGBoost CPU 跑很久 | 确保 `xgboost.device=auto` 触发 GPU；或调小 `n_estimators` |
| 验证集 AUC 大幅高于测试集 | 时间分割泄漏？检查 `temporal_split` 顺序 |

### 10.4 GPU / CPU 环境

- 数据加载：自动尝试 cuDF，回落 pandas（`print: "cuDF not available, falling back to pandas"`）
- UMAP：自动尝试 cuML，回落 sklearn
- 训练 / 推理：必须有 CUDA + PyTorch
- XGBoost：`device: auto` 自动选

如要强制走 CPU 路径，将 `configs/xgboost.yaml` 的 `device` 改为 `cpu`。

---

## 11. 清理与重置

### 11.1 清理中间产物（保留模型与数据）

```bash
make clean
# 等价于：
# rm -rf data/TabFormer/temporal_split/
# rm -rf data/decoder_corpus/
# rm -rf data/embeddings/
# rm -rf data/outputs/
# rm -rf models/decoder-demo/
```

⚠️ `make clean` **不会删除** `data/TabFormer/raw/` 与 `models/decoder-foundation-model/`。

### 11.2 重置全部（含原始数据与预训练模型，**不可逆**）

```bash
rm -rf data/ models/
```

### 11.3 仅清理某一步的缓存

```bash
# 重新跑 Step 2
rm -rf data/decoder_corpus/ && python scripts/step_02_tokenize_corpus.py

# 重新跑嵌入提取
rm -rf data/embeddings/ && python scripts/step_04_extract_embeddings.py

# 仅重训欺诈检测（不动 embedding）
python scripts/step_05_fraud_detection.py
```

---

## 附录 A — 模型规模与默认超参速查

| 项 | 值 |
|----|----|
| 参数量 | ~29M |
| 架构 | Llama decoder (RoPE + GQA + SwiGLU + RMSNorm) |
| `hidden_size` | 512 |
| `num_hidden_layers` | 8 |
| `num_attention_heads` / `num_key_value_heads` | 8 / 2 |
| Tokens / 交易 | 12 |
| 序列长度 | 4096（≈ 315 笔交易） |
| `vocab_size` | 6251（须与 tokenizer `get_vocab_size()` 一致） |
| Optimizer | AdamW (lr=2e-4, wd=0.077, betas=[0.9, 0.95]) |
| LR Schedule | Cosine + 10 步 warmup |
| 分布式 | FSDP2, dp=自动 |
| Checkpoint | `safetensors`, 每 15 步存一次 |

---

## 附录 B — 典型场景一分钟上手

### B.1 "我只想看看 demo 效果"

```bash
pip install -e ".[nemo,gpu]" --extra-index-url https://pypi.nvidia.com
python scripts/run_pipeline.py --steps all --demo
```

### B.2 "我有预训练模型，想跑端到端评估"

```bash
# 1. 把模型放到 models/decoder-foundation-model/
pip install -e ".[gpu]" --extra-index-url https://pypi.nvidia.com
make all-pretrained
```

### B.3 "新数据来了，要不要重训？"

1. 替换 `configs/dataset.yaml` → 跑 Step 1 → 看 baseline AUC 是否合理
2. 跑 Step 2 + Step 4 + Step 5（用现有预训练模型）
3. 看 **Combined vs Baseline** AUC 差距：
   - 差距 > 0.005 → 预训练仍有效，**不重训**
   - 差距 ≈ 0 → 预训练信号弱，按[第 8 节决策树](#8-是否需要重新预训练决策树)评估是否续训
4. 决定续训时：`python scripts/step_03_train_model.py --max-steps 1500 --nproc 8`

### B.4 "我做了一个 tokenizer 改进，想知道有没有用"

```bash
# 词表变了 → 需要重训
rm -rf data/decoder_corpus/ models/decoder-demo/
python scripts/step_02_tokenize_corpus.py --force
python scripts/step_03_train_model.py --demo    # 先试 30 步看 loss
python scripts/run_pipeline.py --steps extract,detect --force
```

---

## 12. 在容器中运行

本节介绍如何用 `docker` / `docker compose` 运行整个 pipeline。**容器化方案不需要在宿主机装任何 Python 依赖**，仅需 Docker 24+（GPU 需要 `nvidia-container-toolkit`）。

### 12.1 三个镜像

| 镜像 | 体积 | 用途 | Dockerfile |
|------|------|------|------------|
| `tm-base` | ~800 MB | 共享层（不直接 run） | `docker/Dockerfile.base` |
| `tm-cpu` | ~1.2 GB | Step 1 / Step 5 / 测试 / 可视化 | `docker/Dockerfile.cpu` |
| `tm-gpu` | ~10 GB | Step 2 (tokenize) + Step 3 (train) | `docker/Dockerfile.gpu` |
| `tm-gpu` | ~8 GB | Step 4 (extract) | `docker/Dockerfile.gpu` |

构建产物通过 `./data` / `./models` volume 持久化，镜像内**不打包数据/模型**。

### 12.2 镜像构建

```bash
# CPU 镜像（最常用，无 NVIDIA 依赖）
make docker-build-cpu

# 拉满 4 个镜像（含 GPU），需 nvidia-container-toolkit
make docker-build

# 指定 tag（用 git sha，CI 友好）
make docker-build-cpu DOCKER_TAG=$(git rev-parse --short HEAD)

# 推送到远程（必须设置非 local 的 DOCKER_REG）
make docker-push DOCKER_REG=ghcr.io/miraclez3 DOCKER_TAG=v1.0
```

### 12.3 单步运行（docker compose）

`docker-compose.yml` 通过 `profile` 隔离不同任务，一条命令只起一个 service：

| Profile | 镜像 | 命令 |
|---------|------|------|
| `data` | tm-cpu | `make compose-data` |
| `detect` | tm-cpu | `make compose-detect` |
| `test` | tm-cpu | `make compose-test` |
| `cpu-pipeline` | tm-cpu | `make compose-cpu-pipeline` |
| `tokenize` | tm-gpu | `make compose-tokenize` |
| `train` | tm-gpu | `TRAIN_NUM_GPUS=8 make compose-train` |
| `extract` | tm-gpu | `make compose-extract` |
| `all` | tm-gpu | `make compose-all` |

**示例 1：在 CPU 上跑 Step 5（fraud detection）**
```bash
make compose-detect
```

**示例 2：8 GPU 全量训练**
```bash
TRAIN_NUM_GPUS=8 make compose-train
```

**示例 3：运行 pytest**
```bash
make compose-test
```

### 12.4 直接 `docker run`

如果不想经过 compose：

```bash
# 查看 entrypoint 帮助
docker run --rm local/tm-cpu:latest --help

# Step 1（CPU）
docker run --rm \
    -v $(pwd)/data:/workspace/data \
    local/tm-cpu:latest data --skip-download

# Step 3 demo 训练（GPU，1 卡）
docker run --rm --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    local/tm-gpu:latest train --demo

# 进入 shell 调试
docker run --rm -it \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    local/tm-cpu:latest bash
```

### 12.5 开发模式：把宿主源码挂入容器

复用 `docker-compose.override.yml.example`，无须改 `docker-compose.yml`：

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
make compose-test    # 此时会用宿主机 ./transaction_model/ 代码
```

### 12.6 健康检查

每次容器启动 30s 后会跑 `docker/healthcheck.py`：

```bash
docker inspect --format='{{.State.Health.Status}}' <container_id>
# → healthy / unhealthy / starting
```

### 12.7 容器化场景的 §8 决策树变体

容器场景的"是否需要重训"判断与 [§8](#8-是否需要重新预训练决策树) 完全一致，但执行命令替换如下：

| 原生命令 | 容器命令 |
|---------|---------|
| `python scripts/step_03_train_model.py --demo` | `make compose-train -- --demo` * |
| `torchrun --nproc-per-node=8 ...` | `TRAIN_NUM_GPUS=8 make compose-train` |
| `python scripts/step_04_extract_embeddings.py --force` | `make compose-extract -- --force` |

*注意：`make compose-train -- --demo` 这种透传需通过 `.env` 的 `TRAIN_EXTRA_ARGS=--demo` 实现（compose 模板已留位置）。更直接的方式是 `docker run ... tm-gpu train --demo`。

### 12.8 清理

```bash
make compose-down          # 停掉所有本项目的容器
make docker-clean          # 删本项目的 4 个镜像
docker system prune        # 系统级清理（小心）
```

---

## 13. Route C — Llama+GPT2 业务损失微调

> 设计背景与字段映射详见 [`upgrade/ylformer.md`](upgrade/ylformer.md) §第二阶段。本节只讲怎么跑。

### 13.1 前置条件

| 件 | 说明 |
|----|------|
| Route A 预训 checkpoint | `models/decoder-yl/` 下有 `config.json` + `safetensors`（由 `step_03_train_model.py --variant yl` 产出） |
| YL tokenizer state | `data/yl/yl_tokenizer.json`（由 `step_02_tokenize_ndjson.py` 产出，**不可删**） |
| 银联 NDJSON 数据 | `data/yl/raw/*.jsonl` 由 `step_01b_load_ndjson.py` 加载（亦可直接读 `examples/sample_data/smoke.jsonl` 冒烟） |
| 依赖 | `pip install -e ".[routec]"` 拉取 `peft>=0.10` 与 `transformers>=4.46` |

### 13.2 命令

```bash
# 冒烟（30 步，CPU 也可，--device cpu 退化为纯 fp32）
python scripts/step_06_finetune_routec.py \
    --config configs/routec/default.json --demo --device cuda

# 正式微调
python scripts/step_06_finetune_routec.py \
    --config configs/routec/default.json --device cuda --max-steps 5000

# 续训（自动找 models/routec/ 下最新 ckpt_step*.pt）
python scripts/step_06_finetune_routec.py --auto-load
```

支持命令行参数：`--config` / `--demo` / `--max-steps` / `--auto-load` / `--device`。

### 13.3 配置速查（`configs/routec/default.json`）

| 字段 | 取值 | 说明 |
|------|------|------|
| `task_type` | `lora`（默认）/ `freeze` / `all_params` | LoRA 注入 query/value；freeze 完全锁住 Llama；all_params 全量微调 |
| `llama.pool_mode` | `last_token` / `mean` / `cls` | 单笔交易如何池化 Llama 输出为 per-txn embedding |
| `loss_fn.name` | 见 §13.4 | 切换业务损失 |
| `lora.r` / `alpha` / `target_modules` | 8 / 32 / `["q_proj","v_proj"]` | LoRA 容量；r→r/alpha=8/32 是常见起点 |
| `data_config.hiswindow` | 512 | 单用户序列最大长度（截窗） |
| `data_config.max_txn_len` | 32 | 单笔交易内部 token 数（含 `<bos>`/`<eos>` padding） |
| `step_scheduler.grad_accum_steps` | 4 | 梯度累积；与 batch_size 一起决定有效 batch |

### 13.4 业务损失仓库

`transaction_model/finetune/losses/__init__.py::LOSS_REGISTRY` 注册了 4 个损失，全部以
`sft_*` 前缀命名（移植自 `risk_control_2/losses/`，去掉 registry 依赖）：

| `loss_fn.name` | 行为 | amount 参数 |
|----------------|------|-------------|
| `sft_cross_loss` | 纯交叉熵，对照基线 | 忽略 |
| `sft_focal_loss_weight` | focal + `pos_weight` 加权（无金额加权） | 忽略 |
| `sft_focal_loss_with_amount` | **金额加权 focal**（核心业务损失） — 正样本权重 = clamp(0.5·amount/100k, 1, amount_clip) | **必传**（从 `trans[-1][19]` 取末笔金额） |
| `sft_pAUC_sigmoid_loss` | partial-AUC pairwise sigmoid（top-k 困难负样本） | 忽略 |

切换损失：编辑 `configs/routec/default.json::loss_fn.{name,params}` 即可，无需改代码。

### 13.5 输出

| 路径 | 内容 |
|------|------|
| `models/routec/ckpt_step<k>.pt` | checkpoint（含 model + optimizer + scheduler + scaler） |
| `models/routec/adapter_step<k>/` | LoRA adapter 单独保存（便于部署到不同的 base 模型） |
| `log/routec/<k>_val.json` | 验证指标（acc/f1/auc/ap） |
| `log/routec/<k>_val_prob.json` | 每用户概率（用于外部对账）：`{cert_sm3: {pred, p0, p1, label, amount}}` |
| `log/routec/train_stats.jsonl` | 训练 loss + lr 历史 |

### 13.6 已知边界

- DeepSpeed / FSDP 暂未启用；trainer 是单 GPU 线性结构（多卡 DDP 友好但未实测）。
- 仅做下游微调，不重训 Llama；移植的 4 个 loss 也都是分类损失，**不含** `dense_multi_loss` 多变量预训损失。
- Route C 与 Route A Step 4/5（嵌入 + XGBoost）**互补但解耦**：Route C 端到端出二分类，Route A 出 512d 嵌入喂 XGBoost。两者可平行对比。

---

如本指南与代码行为出现矛盾，**以代码为准**。改进本指南欢迎提 PR。
