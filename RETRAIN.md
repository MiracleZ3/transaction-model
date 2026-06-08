# 领域迁移指南：从信用卡交易到理财产品申赎

> 本文档回答一个问题：**如何把当前 transaction-model 从信用卡欺诈检测，迁移到理财产品申赎数据上训练？**
>
> 适用场景包括但不限于：申赎方向预测、流失预警、客户分层、产品推荐、异常行为检测。
>
> **银联风控 NDJSON 用户**：本文档面向"自定义 CSV 领域迁移"，**不需要照本文改任何代码**——
> 银联 `risk_control_2` NDJSON 数据已通过 Route A（独立入口 `step_01b`/`step_02_tokenize_ndjson.py`）
> + Route C（`step_06_finetune_routec.py`，Llama+GPT2+分类头+业务损失）**直接落地**。
> 请直接读 [`README.md`](README.md) §"银联（YL）NDJSON 路线"、[`How_To_Use.md`](How_To_Use.md) §13、
> [`upgrade/ylformer.md`](upgrade/ylformer.md)，并用 [`examples/sample_data/smoke.jsonl`](examples/sample_data/smoke.jsonl) 跑冒烟。
>
> 项目原结构（[`README.md`](README.md)）与训练流程（[`How_To_Use.md`](How_To_Use.md)）保持不变，本文只描述"哪些代码要改、怎么改、改的顺序"。

---

## 目录

1. [迁移评估：哪些能复用，哪些要重写](#1-迁移评估哪些能复用哪些要重写)
2. [数据字段映射设计](#2-数据字段映射设计)
3. [必改代码清单（按依赖顺序）](#3-必改代码清单按依赖顺序)
4. [下游任务（Step 5）的重新定义](#4-下游任务step-5的重新定义)
5. [是否需要重新预训练：决策](#5-是否需要重新预训练决策)
6. [推荐实施顺序与工作量](#6-推荐实施顺序与工作量)
7. [实战建议与避坑指南](#7-实战建议与避坑指南)
8. [附录 A：完整字段映射示例](#8-附录-a完整字段映射示例)
9. [附录 B：tokenizer schema 参数化草案](#9-附录-btokenizer-schema-参数化草案)

---

## 1. 迁移评估：哪些能复用，哪些要重写

| 复用度 | 模块 | 说明 |
|--------|------|------|
| ✅ 完全复用 | `config.py` | 与领域无关 |
| ✅ 完全复用 | `tokenizer/pipeline.py` 通用编排 | 抽象层，不动 |
| ✅ 完全复用 | `tokenizer/{base,fixed_vocab,mapping,categorical_hash,numerical,timedelta}` 6 个底层 tokenizer | 通用工具 |
| ✅ 完全复用 | `training/clm_data.py`、`training/run_training.py` | CLM Dataset 与 NeMo 入口与领域无关 |
| ✅ 完全复用 | `inference/decoder_inference.py` | HF 推理通用 |
| ✅ 完全复用 | `scripts/run_pipeline.py` + 全部 step 脚本 | 编排通用 |
| ✅ 训练循环复用 | `detection/xgboost.py` 训练逻辑 | 三模型对比框架可保留；**load_and_align_raw_features() 内部硬编码欺诈列名要改** |
| ⚠️ 改几行 | `data/feature.py`、`data/sampling.py`、`inference/extract.py`、`constants.py` | 只是硬编码列名和欺诈字面量 `"Yes"` |
| ⚠️ 改字段映射 | `data/split.py` | 时间分割逻辑保留；`add_date_column()` 拼接逻辑看你的日期列结构 |
| 🔴 重写 | `tokenizer/financial_pipeline.py` | **领域知识的硬编码全在这里**（KNOWN_MCCS / CHIP_MAPPING / AMOUNT_THRESHOLDS / INDUSTRY_RANGES 等） |
| 🔴 重新设计 | 下游任务（Step 5） | 欺诈二分类几乎不再适用，见 §4 |
| 🔴 重新预训练 | 模型权重 | 见 §5 |

**总评**：约 60% 代码原状可用、30% 改少量行、10% 重写。

---

## 2. 数据字段映射设计

把理财申赎字段对齐到 TabFormer 的 13 列概念。**不要硬塞，缺的就砍，新的就加。**

### 2.1 直接映射

| TabFormer 字段 | 理财申赎对应 | 处理方式 |
|---------------|------------|---------|
| `User` | 客户 ID | 直接用，剪枝到 FixedVocabTokenizer 的 max_val |
| `Card` | 账户 ID / sub-portfolio | 没有就丢；有就当辅助 group |
| `Year/Month/Day/Time` | 申赎日期 + 时刻 | 直接复用；`add_date_column()` 不变 |
| `Amount` | 申赎金额 | ⚠ 分桶要重做（见 §3.6） |
| `Is Fraud?` | **替换为目标列**（流失/赎回/下一动作） | 见 §4 |
| `Merchant Name` | **产品代码 / 产品名称** | hash 到 bucket |
| `MCC` | **产品类型**（货基/债基/股基/混合…） | 替换 KNOWN_MCCS |
| `Use Chip` | **渠道**（APP/网银/柜台/第三方） | 替换 CHIP_MAPPING |
| `Merchant State` | 客户地区 / 渠道机构 | 或直接砍 |
| `Zip`、`Merchant City` | 无对应 | 砍掉 |

### 2.2 新增字段（理财特有，强烈建议加 token）

| 新字段 | 是否进 tokenizer | 处理方式 |
|--------|----------------|---------|
| 申赎方向（申/赎） | ✅ 必加 | 新 tokenizer step，2 个 token |
| 申赎费率 | ✅ 推荐 | NumericalTokenizerOptBin (quantile) |
| 持有时长 | ✅ 推荐 | TimeDeltaTokenizer 现成可用 |
| 客户风险等级（5 档） | ✅ 推荐 | FixedVocabTokenizer(prefix="RISK") |
| 产品 7 日年化 / 万份收益 | ✅ 推荐 | NumericalTokenizerOptBin |
| 客户年龄/性别（脱敏后） | 可选 | 视隐私要求 |

### 2.3 Tokens / 交易的设计

信用卡是 12 token/笔 + ~315 笔/序列（4096 上下文）。

申赎信号更丰富，建议 **14-16 token/笔**，`chunk_size` 调到 ~250（保证 4096 上下文塞得下）。同步修改 `configs/tokenizer.yaml`：

```yaml
tokenizer:
  merchant_hash_size: 1000    # 产品数远少于商户数，可减
  chunk_size: 250             # ~250 笔/序列
  context_window: 4096
  tokens_per_txn: 15
```

---

## 3. 必改代码清单（按依赖顺序）

### 3.1 `configs/dataset.yaml`
更新 `raw_csv` 路径、`temporal_split_dir`、`feature_cols`、采样规模。

```yaml
dataset:
  raw_csv: "data/Fund/raw/orders.csv"
  temporal_split_dir: "data/Fund/temporal_split"

split:
  train_ratio: 0.8
  val_ratio: 0.1

sampling:
  balanced_train_size: 500000   # 视数据规模调整
  eval_samples: 100000
  random_state: 42

feature_cols:
  - cust_id
  - acct_id
  - year
  - month
  - day
  - hour
  - amount
  - direction
  - product_code
  - product_type
  - channel
  - region
  - holding_days
```

### 3.2 `transaction_model/constants.py`
改 `FRAUD_COL` 与 `FRAUD_POSITIVE_VALUES`：

```python
# 你的目标列名，按 §4 任务决定
FRAUD_COL = "is_churn"           # 或 "is_redemption" / "next_action"
FRAUD_POSITIVE_VALUES = (1, "1", True)

# MCC 行业映射（如果不再用 MCC，可删除或重命名）
MCC_INDUSTRY_RANGES = [...]      # 视业务需要
```

### 3.3 `transaction_model/data/feature.py`
`engineer_features()` 现在干 3 件事：
- 从 `Time` 提取 `Hour` — **保留**
- 清洗 `Amount`（去 `$` 和逗号）— 如果你的 Amount 是 float，**删掉**
- 生成 `_target` 二值标签 — **改逻辑**

```python
def engineer_features(gdf):
    gdf['Hour'] = gdf['Time'].str.split(':', n=1, expand=True)[0].astype(int)
    # 你的 Amount 已经是数字 → 跳过清洗
    gdf['_target'] = (gdf[FRAUD_COL].isin(FRAUD_POSITIVE_VALUES)).astype(int)
    return gdf
```

### 3.4 `transaction_model/data/split.py`
`add_date_column()` 拼 `Year-Month-Day`。
- 如果你的数据是 `txn_date` 单列，要改拼接逻辑
- 时间分割本身（80/10/10）**继续可用**，是项目里最稳的部分

### 3.5 `transaction_model/data/sampling.py`
`save_eval_subsets()` 第 147 行硬编码读 `'Is Fraud?'`。改成读 `FRAUD_COL` 常量或目标列参数。

### 3.6 `transaction_model/tokenizer/financial_pipeline.py`（**最重的工作量**）

#### 3.6.1 重建"领域常量区"（38-82 行）

| 现状 | 理财场景 |
|------|---------|
| `KNOWN_MCCS = [1711, 3000, ...]` | `FUNDS_TYPES = ["MMF","BOND","EQUITY","MIXED",...]` |
| `INDUSTRY_RANGES = [(0,1499,"AGRICULTURAL")]` | `RISK_LEVELS = [(0,1,"R1"),(1,2,"R2"),...]` 或 `PRODUCT_RISK_RANGES = [...]` |
| `CHIP_MAPPING = {"CHIP TRANSACTION": "CHIP"}` | `CHANNEL_MAPPING = {"APP":"APP","WEB":"WEB","COUNTER":"COUNTER",...}` |
| `ALL_STATES = ["AL","AK",...]` | `REGIONS = [...]` 或干脆删 |
| `AMOUNT_THRESHOLDS = [0,10,50,100,500,1000,5000]` | **见下 A/B 方案** |

#### 3.6.2 AMOUNT_THRESHOLDS 最坑的点

信用卡金额集中在 $1-$500，理财申赎通常 ¥1k-¥1M。继续用 `[0,10,50,100,...]` 会让 99% 落进同一个桶，token 失去区分度。

| 方案 | 适用 | 优缺 |
|------|------|------|
| **A. 改成理财产品量级的固定阈值** `[0, 1k, 10k, 50k, 100k, 500k, 1M, inf]` | 探索阶段 | 简单、可解释 |
| **B. `amount_strategy="quantile"`** (pipeline 已支持，会自动 call `NumericalTokenizerOptBin`) | 数据稳定后 | 自适应但分布漂移时要重 fit |

#### 3.6.3 重排 `_configure_steps()` 的 12 个 step

- 砍：`zip3`（可能 `state_clean`）
- 改名/换表：`merch_hash`→`product_hash`、`mcc_int`/`mcc_str`→`product_type`、`chip_upper`→`channel`
- 加：`direction`（申/赎）、`fee_rate`、`holding_days`、`cust_risk`

#### 3.6.4 `preprocess()`（208-282 行）

里面对原始列名 `amount` / `merchant_name` / `mcc` / `use_chip` / `zip` 做了一连串 cuDF 字符串处理。新数据列名不同，全部要调整。

**强烈建议 schema 参数化**（见 §9 草案），让 `financial_pipeline` 真正通用：

```python
class FinancialTokenizerPipeline(TokenizerPipeline):
    def __init__(self, schema: dict | None = None, ...):
        self.schema = schema or DEFAULT_FUND_SCHEMA
```

### 3.7 `transaction_model/inference/extract.py`
`_extract_labels()` 已经支持多列名匹配（`Is Fraud? / is_fraud / Is_Fraud / label / fraud`），把新标签名（如 `"is_churn"`、`"is_redemption"`、`"next_action"`）加进去即可。

### 3.8 `transaction_model/detection/xgboost.py`
`load_and_align_raw_features()` 第 168-170 行硬编码 `Is Fraud?` 与 `Yes/1`，改成读 `FRAUD_COL` 常量。

---

## 4. 下游任务（Step 5）的重新定义

当前 Step 5 是**欺诈二分类**，理财域几乎不需要。**先想清楚目标再动 tokenizer。**

### 4.1 候选目标

| 候选目标 | 监督信号 | 是否适合用 embedding | 改动量 |
|---------|---------|--------------------|-------|
| 客户下一动作预测（继续申/转赎/静默） | 历史 + 当前点 | ✅ 与 pretrain 同构，最佳 | 改 XGB 多分类 |
| 流失/赎回预警（30 天内有赎回） | 二分类 | ✅ 直接替换 Is Fraud | 改 1 个常量 |
| 客户分层 / 相似客户检索 | 无监督 | ✅ embedding 聚类即可 | 跳过 Step 5，直接 UMAP |
| 产品推荐 | 多分类 | ✅ 但需新增产品 ID embedding head | 加分类 head |
| 风险客户识别（异常申赎行为） | 二分类 | ✅ 替换 Is Fraud | 改 1 个常量 |

### 4.2 三个具体场景的实施差异

**场景 A：流失/赎回预警（最有 ROI，改造最少）**

- 改 `FRAUD_COL = "is_churn"` 一行常量
- `configs/dataset.yaml` 替换 `feature_cols`
- XGBoost 三模型对比（baseline / embedding / combined）**结构完全保留**
- AUROC / AP 指标也仍然适用，只是含义从"抓欺诈"变成"抓流失"

**场景 B：下一动作预测（多分类）**

Step 5 需要改：
- 替换 `XGBClassifier(objective='multi:softprob')`
- 替换 metrics：AP 改成 confusion matrix + top-3 accuracy
- 可考虑加一个分类 head（直接用 embedding → MLP），而不是只 XGBoost

**场景 C：客户聚类/相似检索**

跳过 Step 5 完全不跑 XGBoost，直接拿 `data/embeddings/*.npy` 跑 UMAP / HDBSCAN——
这些代码已经在 `visualization/embedding_viz.py` 里现成可用。

### 4.3 替代方案：Route C（Llama+GPT2+分类头 + 业务损失）

如果下游希望**保留端到端分类头并使用业务损失**（如金额加权 focal、partial-AUC pairwise），
而不放弃 Step 3 训练的 decoder 表征，应走 **Route C**：

```bash
pip install -e ".[routec]"        # peft>=0.10, transformers>=4.46
python scripts/step_06_finetune_routec.py --config configs/routec/default.json --demo
```

Route C 在 Llama（route A 预训 ckpt，冻结 + LoRA）之后接一个 GPT2 跨交易时序编码器，
再加分类头。支持 4 个业务损失：
`sft_focal_loss_with_amount`（默认，金额加权）/ `sft_pAUC_sigmoid_loss` /
`sft_focal_loss_weight` / `sft_cross_loss`。完整说明见
[`How_To_Use.md`](How_To_Use.md) §13 与 [`upgrade/ylformer.md`](upgrade/ylformer.md) §第二阶段。

> 这是 §4.1 表里"加分类 head"的现成实现，不必再自己写。

---

## 5. 是否需要重新预训练：决策

直接套用 [`How_To_Use.md` §8 决策树](How_To_Use.md#8-是否需要重新预训练决策树)：

| 检查项 | 理财场景的结果 |
|-------|-------------|
| 改动了字段映射 | 属于 **❸ 词表变化** |
| merchant_hash_size 是否变 | 几乎肯定变（产品数 ≠ 商户数） |
| 是否同领域不同数据 | 否——**属于 ❺ 显著不同领域** |
| 模型架构变 | 否 |

→ **结论：必须从零预训练**。

**为什么续训不行？**
"续训"在领域跨度过大时不仅收益小，还可能误导——信用卡学到的"金额越大越像欺诈"先验，在理财域反而不一定成立（大额申购可能是高净值客户的正常行为）。

**实操**：
1. 先用 `--demo`（30 步）跑通，看 loss 能不能单调下降（参考 [How_To_Use.md §5.4 健康检查](How_To_Use.md)）
2. 再放大到 3000 步正式训练
3. 训练完毕跑 Step 4+5，看 **Combined vs Baseline AUC** 是否 lift > 0.005，证明 embedding 有效

---

## 6. 推荐实施顺序与工作量

| Phase | 工作 | 预估 | 验收 |
|-------|------|------|------|
| 1 | **数据 schema 调研**：列名、量纲、缺失率、字段语义 | 1 天 | forming `data/sample.csv` + schema 文档 |
| 2 | **改 `configs/dataset.yaml` + `constants.py`** | 0.5 天 | `make data` 跑通 Step 1 时间分割 |
| 3 | **改 `data/feature.py` + `split.py` + `sampling.py`** | 1 天 | `val_eval.parquet` 含新标签列 |
| 4 | **重写 `tokenizer/financial_pipeline.py`（schema 化）** | 2-3 天 | `tokenizer.get_vocab_size()` 输出合理值（~5000-8000） |
| 5 | **改 `inference/extract.py` + `detection/xgboost.py`** 对新标签 | 0.5 天 | `_extract_labels()` 能找到新标签 |
| 6 | **`make compose-tokenize`**：跑一次 tokenize，检查 sample 输出 | 0.5 天 | 人工 review 前 10 条 corpus lines 是否合理 |
| 7 | **`make compose-train --demo`**：30 步训练，看 loss | 0.5 天 | train loss 单调下降到 ~2-3 即可 |
| 8 | **决策：续训 vs 从零**（默认从零） | — | val loss 收敛 |
| 9 | **正式训练** `TRAIN_NUM_GPUS=8 make compose-train` | 视数据量 4-12h | val loss 平台期 |
| 10 | **Step 4+5 评估** + 业务指标对照 | 1 天 | combined AUC > baseline ≥ 0.005 |

**合计 ≈ 7-10 天**（不含大规模训练 wall-clock）。

---

## 7. 实战建议与避坑指南

1. **不要急着改 tokenizer，先把 schema 设计文档写出来**。
   Token id 一旦写入 vocab，迁移成本就锁定。先在纸上把字段、token 数、分桶策略列出来。

2. **保留"信用卡模式"作为回归测试 fixture**。
   给 `financial_pipeline.py` 加一个 `preset="tabformer" | "fund"` 开关，而不是直接覆盖原值。
   以后调试或上线混部时不踩对方。

3. **`amount_strategy="quantile"` 比死阈值更稳**——
   只要接受多一个 `cuml` 依赖（已经在 GPU 镜像里）。

4. **申赎方向一定要做 token**。
   理财最关键的信号是"动作方向"。信用卡是隐式的（只有支出），理财是显式的双向——把它放进序列里能让 decoder 学到真实的状态切换。

5. **Step 5 可以暂时不换算法**：
   先复用 XGBoost + 三模型对比框架，等证实 embedding 有效再考虑加 transformer head。
   深度学习只在 XGBoost 抓不到的复杂模式上才赢。

6. **`group_by` 改成 `["user"]`**：
   信用卡按 `["user", "card"]` group（一人多卡），理财往往一个人一个资金账户。
   或者你想按 `["user", "product"]` 切，把产品当成"卡"也合理——这取决于任务目标。

7. **金额分桶两种制式并存**：
   申 vs 赎的金额分布有显著差异（赎回有持有期约束）。**先按方向分别 quantile 分桶** 比一刀切效果更好。

8. **时间特征比交易更重要**：
   申赎行为有强周期性（月末、季末、年底）。`MONTH`、`DOW`、`HOUR` 三个 token 必留；可考虑加 `IS_MONTH_END` 等衍生 token。

9. **目标列泄漏防范**：
   如果做"流失预警"，确保 `_target` 的标签不会通过下游特征泄漏到训练集（例如"过去 30 天赎回次数"这种特征本身就和标签强相关）。

10. **从 demo 训练开始排错**：
    第一次跑通新 tokenizer 后，**先跑 `make compose-train --demo`（30 步）**，看 loss 能否下降。
    不能下降说明 tokenizer/数据有问题，不要急着扩规模。

---

## 8. 附录 A：完整字段映射示例

假设你的原始数据是：

```csv
cust_id,acct_id,order_date,order_time,product_code,product_type,amount,direction,channel,region,fee_rate,holding_days,risk_level,pred_target
C001,   A001,   2024-03-15,  10:23,    F0001,        MMF,         50000, BUY,  APP,  SH,        0.0015,    30,           R2,       0
C001,   A001,   2024-04-15,  14:55,    F0001,        MMF,         50000, SELL, APP,  SH,        0.0050,    61,           R2,       1
...
```

映射到 tokenizer 后产生的 token（每行约 15 个）：

```
<bos>
  DIR_BUY                    ← direction
  AMT_4                      ← amount bucket (50K)
  PROD_829                   ← product_code hash → bucket
  TYPE_MMF                   ← product_type
  RISK_R2                    ← cust risk
  HOUR_10                    ← order_time hour
  DOW_5                      ← day of week
  MONTH_03                   ← month
  CHAN_APP                   ← channel
  REG_SH                     ← region
  FEE_3                      ← fee_rate quantile bin
  HDAY_30                    ← holding_days (existing TimeDeltaTokenizer)
  CUST_001                   ← cust_id
<sep>
  ...next txn...
<eos>
```

`vocab_size` 估算：5 specials + 2 (direction) + 10 (amount) + 1000 (product) + 5 (type) + 5 (risk) + 24 (hour) + 7 (dow) + 12 (month) + 5 (chan) + 40 (region) + 10 (fee) + 30 (hday) + 200 (cust，按规模) ≈ **~1385-1500**（远小于 TabFormer 的 6251）。

实际值用 `tokenizer.get_vocab_size()` 验证，写入 `configs/training.yaml` 的 `model.config.vocab_size`。

---

## 9. 附录 B：tokenizer schema 参数化草案

为了让 `financial_pipeline.py` 同时支持 tabformer 与 fund，建议改成 schema 驱动：

```python
# transaction_model/tokenizer/financial_pipeline.py（重构草案）

DEFAULT_TABFORMER_SCHEMA = {
    "amount": "amount",
    "product": "merchant_name",     # 旧：merchant_name, 新：product_code
    "product_type": "mcc",
    "channel": "use_chip",
    "location": "merchant_state",
    "zip": "zip",
    "cust": "user",
    "card": "card",
    "year": "year",
    "month": "month",
    "day": "day",
    "time": "time",
}


DEFAULT_FUND_SCHEMA = {
    "amount": "amount",
    "product": "product_code",
    "product_type": "product_type",
    "channel": "channel",
    "location": "region",
    "zip": None,                   # 不存在
    "cust": "cust_id",
    "card": "acct_id",
    "year": "year",
    "month": "month",
    "day": "day",
    "time": "order_time",
    # 新增字段
    "direction": "direction",
    "fee_rate": "fee_rate",
    "holding_days": "holding_days",
    "risk": "risk_level",
}


class FinancialTokenizerPipeline(TokenizerPipeline):
    PRESETS = {
        "tabformer": (DEFAULT_TABFORMER_SCHEMA, _tabformer_configure_steps),
        "fund":      (DEFAULT_FUND_SCHEMA,      _fund_configure_steps),
    }

    def __init__(self, preset: str = "tabformer", schema: dict | None = None, ...):
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        default_schema, configure_fn = self.PRESETS[preset]
        self.schema = schema or default_schema
        super().__init__(**kwargs)
        configure_fn(self)

    def preprocess(self, df):
        # 用 self.schema 查列名，而不是硬编码
        amt_col = self.schema["amount"]
        prod_col = self.schema["product"]
        ...
```

之后在新数据上：

```python
pip = FinancialTokenizerPipeline(
    preset="fund",
    merchant_hash_size=1000,
    amount_strategy="quantile",
)
```

或在 `configs/tokenizer.yaml` 里：

```yaml
tokenizer:
  preset: fund
  merchant_hash_size: 1000
  amount_strategy: quantile
  amount_bins: 10
  chunk_size: 250
  context_window: 4096
```

---

## 10. 快速决策树（一图流）

```
你确定要迁移到理财数据吗？
│
├── 是 → 看你的目标
│       │
│       ├── 流失预警（最简）→ §4 场景 A
│       ├── 异常监测（最简）→ §4 场景 A，换 _target 含义
│       ├── 下一动作预测  → §4 场景 B
│       ├── 客户聚类       → §4 场景 C（跳过 Step 5）
│       └── 其他           → 先沉淀目标定义再回来
│
├── 工作量预估
│       └── §6（7-10 天 + 训练）
│
└── 执行节奏（强烈推荐）
        Phase 1-3：改非 tokenizer 文件 → 跑通 Step 1
        Phase 4-6：改 tokenizer → demo tokenize → 人工 review
        Phase 7：demo 训练看 loss
        Phase 8-9：正式训练
        Phase 10：业务评估
```

---

如本文档与代码实际行为出现矛盾，**以代码为准**。改进本文档欢迎提 PR。
