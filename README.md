# Transaction Model

Financial transaction foundation model for fraud detection.

基于 NVIDIA transaction-foundation-model 蓝图重构，将 Jupyter Notebook 原型拆解为标准 Python 包，提供配置驱动、CLI 一键执行的端到端流水线。

## 架构概览

```
原始交易数据 ──► 时间分割 ──► 领域分词器 ──► 语料库
                                              │
                    Decoder-Only Transformer (~29M)
                    (Llama 架构: RoPE + GQA + SwiGLU)
                                              │
              原始特征(13d)              嵌入向量(512d→64d PCA)
                  │                           │
                  └───── XGBoost 三模型对比 ─────┘
                    Baseline    Embedding    Combined
```

**核心创新**: 金融领域专用分词器 (Financial Tokenizer)，每笔交易仅 12 tokens（GPT-2 BPE 需 30-50+），在 4096 上下文窗口内可容纳约 315 笔交易。

## 项目结构

```
transaction-model/
├── configs/                        # YAML 配置文件
│   ├── dataset.yaml                # 数据源、分割比例、采样参数
│   ├── tokenizer.yaml              # 分词器参数、语料路径
│   ├── training.yaml               # 模型架构、训练超参 (NeMo AutoModel)
│   └── xgboost.yaml                # XGBoost 超参、PCA 维度、推理参数
├── transaction_model/
│   ├── config.py                   # 统一配置加载器
│   ├── constants.py                # MCC 行业映射、UMAP 参数、标签常量
│   ├── data/                       # 数据管道
│   │   ├── download.py             # 数据集下载与解压
│   │   ├── loader.py               # cuDF/pandas 加载 (GPU 自动回退 CPU)
│   │   ├── split.py                # 时间分割 (按日期累计行数)
│   │   ├── feature.py              # 特征工程 + OrdinalEncoder 编码
│   │   └── sampling.py             # 平衡采样 / 分层采样
│   ├── tokenizer/                  # 金融领域分词器 (9 个模块)
│   ├── corpus/generate.py          # 语料库生成 (parquet → token 文本)
│   ├── training/                   # 模型训练
│   │   ├── clm_data.py             # FinancialCLMDataset (NeMo _target_ 入口)
│   │   ├── train.py                # 训练命令构建与启动
│   │   └── run_training.py         # torchrun 入口点
│   ├── inference/                  # 推理与嵌入提取
│   │   ├── decoder_inference.py    # HF 模型推理封装 (last-token/mean pooling)
│   │   └── extract.py              # 批量嵌入提取 (含 __row_id__ 对齐机制)
│   ├── detection/                  # 欺诈检测
│   │   ├── xgboost.py              # 三模型 XGBoost 对比实验
│   │   └── metrics.py              # ROC-AUC / Average Precision
│   └── visualization/              # 可视化
│       ├── data_viz.py             # 数据探索 (欺诈分布、MCC 欺诈率)
│       ├── tokenizer_viz.py        # 分词器对比 (Financial vs GPT-2)
│       ├── embedding_viz.py        # UMAP 2D/3D 嵌入可视化
│       └── results_viz.py          # 模型对比柱状图
├── scripts/                        # CLI 入口
│   ├── step_01_dataset_baseline.py
│   ├── step_01b_load_ndjson.py          # 银联 NDJSON → parquet (Route A)
│   ├── step_02_tokenize_corpus.py
│   ├── step_02_tokenize_ndjson.py       # 银联 parquet → corpus + tokenizer state (Route A)
│   ├── step_03_train_model.py           # --variant {tabformer,yl}
│   ├── step_04_extract_embeddings.py    # --dataset-config dataset_yl 切到银联
│   ├── step_05_fraud_detection.py       # --dataset-config dataset_yl 切到银联
│   ├── step_06_finetune_routec.py       # Route C：Llama+GPT2+分类头+LoRA+业务损失
│   └── run_pipeline.py             # 全流程一键执行（仅 TabFormer Step 1-5）
├── configs/
│   ├── {dataset,tokenizer,training,xgboost}.yaml   # TabFormer 路线
│   ├── dataset_yl.yaml, training_yl.yaml           # 银联 NDJSON 路线 (Route A)
│   └── routec/default.json                          # Route C 微调主配置
├── examples/                                  # 仓库内置冒烟样例
│   ├── sample_data/smoke.jsonl
│   └── generate_smoke_sample.py
├── tests/                                     # 25 个 CPU 测试（含 Route A + Route C）
├── pyproject.toml                             # extras: [dev][gpu][nemo][routec]
├── requirements.txt
└── Makefile
```

## 环境要求

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | >= 3.9 | CPU 流程已测试 3.13.5；GPU 流程（RAPIDS）建议 3.11，已测试 3.11 |
| CUDA | 12.x | 仅训练/推理时需要，数据管道支持纯 CPU |
| PyTorch | >= 2.1 | |
| Transformers | >= 4.40 (CLM 预训) / >= 4.46 (Route C 微调) | Route C 需 `[routec]` extra |
| peft | >= 0.10 | 仅 Route C 需要（`pip install -e ".[routec]"`） |

## 快速开始

### 安装

```bash
# 基础安装
pip install -e .

# GPU 加速 (可选，数据加载和 UMAP)
# RAPIDS wheel 只发布在 NVIDIA 索引上，必须带 --extra-index-url
pip install -e ".[gpu]" --extra-index-url https://pypi.nvidia.com

# NeMo 训练框架 (可选，仅训练步骤需要)
pip install -e ".[nemo]"

# Route C 混合下游微调（Llama+GPT2+分类头+LoRA，需要 route A 预训 ckpt）
# 拉取 peft>=0.10 与 transformers>=4.46；可与 [nemo]/[gpu] 组合：
pip install -e ".[routec]"
# 或：pip install -e ".[routec,nemo]"
```

### Docker（推荐用于跨机器一致性）

```bash
# ─── CPU 镜像（开发机或无 NVIDIA 驱动的服务器） ───
make docker-build-cpu          # ~3 min, 1.2 GB（含 base）

# ─── GPU 镜像（需 nvidia-container-toolkit + NGC 登录；见下文手动章节） ───
make docker-build              # ~15 min, 三镜像共 ~16 GB

# ─── 单步运行（CPU） ───
make compose-test              # pytest
make compose-data              # Step 1: 数据基线（首次需下载数据）
make compose-detect            # Step 5: 欺诈检测

# ─── 单步运行（GPU） ───
make compose-tokenize          # Step 2
make compose-train             # Step 3 (TRAIN_NUM_GPUS=8 可覆盖)
make compose-extract           # Step 4

# ─── 端到端 ───
make compose-cpu-pipeline      # CPU: data + detect
make compose-all               # GPU 全流程

# ─── 直接 docker run ───
docker run --rm local/tm-cpu:latest --help
docker run --rm --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    local/tm-gpu:latest train --demo
```

完整的镜像/Compose 设计参见 [`docker/`](docker/) 与 [`How_To_Use.md`](How_To_Use.md) §12。

### 手动 docker build（不通过 make）

`make docker-build-*` 失败或服务器不便用 make 时，直接 `docker build`。
镜像之间存在依赖（`tm-cpu` 继承 `tm-base`；`tm-gpu` 独立 FROM NGC）。

#### 镜像清单与文件

| 镜像 tag | Dockerfile | base | 用途 | 大小 |
|----------|-----------|------|------|------|
| `local/tm-base:latest` | `docker/Dockerfile.base` | `python:3.11-slim` | CPU base（不直接运行，给 tm-cpu FROM） | ~600 MB |
| `local/tm-cpu:latest`  | `docker/Dockerfile.cpu`  | `local/tm-base`    | Step 1/5 + pytest + 可视化（CPU 即可） | ~1.2 GB |
| `local/tm-gpu:latest`  | `docker/Dockerfile.gpu`  | `nvcr.io/nvidia/pytorch:24.10-py3` | GPU 统一：Step 2/3/4 + Route C | ~14 GB |

#### NGC 登录（仅 GPU 镜像首次需要）

`nvcr.io/nvidia/pytorch` 是 NGC 私有 registry，需要 free NGC 账号：

```bash
# 1. 在 https://ngc.nvidia.com/setup/api-key 生成 API key
# 2. 登录（用户名固定 $oauthtoken，密码填 API key）：
docker login nvcr.io -u '$oauthtoken' -p '<YOUR_NGC_API_KEY>'
# 3. 测试可拉到 base：
docker pull nvcr.io/nvidia/pytorch:24.10-py3
```

#### 仅 CPU（无需 NGC，推荐作为冒烟/CI 起步）

```bash
# 1. base（与 `make docker-build-base` 等价；不用 BuildKit 也能跑）
docker build -f docker/Dockerfile.base -t local/tm-base:latest .

# 2. cpu（依赖 tm-base）
docker build -f docker/Dockerfile.cpu \
    --build-arg REGISTRY=local --build-arg TAG=latest \
    -t local/tm-cpu:latest .

# 验证
docker run --rm local/tm-cpu:latest --help
```

#### 全量（CPU + GPU）

```bash
# 同上的 base + cpu 后，再构建 GPU 镜像：
#   - 首次需要 docker login nvcr.io（见上节）
#   - NGC_TAG 可覆盖（默认 24.10-py3，pin 到当前 release）
docker build -f docker/Dockerfile.gpu \
    --build-arg NGC_TAG=24.10-py3 \
    -t local/tm-gpu:latest .

# 列出 3 张镜像确认
docker images | grep '^local/tm-'
```

可选 `--build-arg`：
- `NGC_TAG=24.10-py3`（默认）—— 选其它 NGC release 时改这里
- `RAPIDS_VERSION=24.10`（默认）—— 与 NGC release 对齐

#### 故障回退：禁用 BuildKit

如果 `make docker-build-*` 在第一步直接报：
```
runc run failed: ... dial unix /var/run/docker/<hash>.sock: connect: no such file or directory
error running prestart hook #0: exit status 1
```
这是 **Docker daemon 的 libnetwork 子系统异常**（不是 Dockerfile 问题）。
根因：Docker daemon 启动时应在 `/var/run/docker/libnetwork/` 下创建
`<hash>.sock`，但因 kernel 升级、`daemon.json` 改成 `"iptables": false`、
AppArmor/SELinux 拦截、或 containerd 状态污染等，socket 没创建出来。
BuildKit 的每条 `RUN` 都会起一个 runc 子容器去走 libnetwork → 必崩。

Makefile 默认已经把 `DOCKER_BUILDKIT=0`，所以 `make docker-build-*` 会自动
切到 **legacy builder**——它不依赖 libnetwork socket，直接在 build 容器内跑
`RUN`，所以你大概率不必做任何额外操作。

如果 `make docker-build-*` 仍报错（说明有人 export 了 `DOCKER_BUILDKIT=1`
or CI 设置了），手动跑：

```bash
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.base -t local/tm-base:latest .
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.cpu \
    --build-arg REGISTRY=local --build-arg TAG=latest \
    -t local/tm-cpu:latest .
DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile.gpu \
    --build-arg NGC_TAG=24.10-py3 \
    -t local/tm-gpu:latest .
```

`Dockerfile.gpu` / `Dockerfile.cpu` 没用 `--mount=type=cache`，classic builder
完全兼容。`Dockerfile.base` 用了 cache mount，classic builder 会忽略该子句
（不报错，只是不缓存）。

如果切到 classic builder **仍**报错（意味 daemon 已经彻底崩了）：
```bash
# 1. 看日志找根因
sudo journalctl -u docker --no-pager -n 80 | grep -E 'libnetwork|docker-init|failed'
# 2. 看 socket 在不在
sudo ls -la /var/run/docker/libnetwork/ 2>&1
# 3. 重启 daemon（会拉起 libnetwork 子系统）
sudo systemctl restart containerd docker
sleep 3
# 4. 验证 daemon 自身健康
docker run --rm hello-world
# 5. 如果 hello-world 也报同样的 libnetwork 错，进阶排查：
#    - /etc/docker/daemon.json 是否设 "iptables": false / "bridge": "none"
#    - 用 nftables 的系统是否需要 modprobe iptable_nat
#    - 强删 containerd 状态：sudo rm -rf /var/lib/containerd/io.containerd.*  后重启
```

> BuildKit 模式（`DOCKER_BUILDKIT=1`）有 pip/apt cache 跨 build 复用的加速，
> 但代价是依赖 daemon 健康。在本仓库中 3 个 image 的 build 都是分钟级，cache
> 节省的时间不够抵消 daemon 飘起来的修复成本，所以默认禁用。想启用加速：
>
> ```bash
> make docker-build DOCKER_BUILDKIT=1
> ```

#### 镜像构建后怎么用

`Dockerfile.cpu` 与 `Dockerfile.gpu` 都把 `docker/entrypoint.sh` 作为
ENTRYPOINT，把 `docker run <img> <subcmd>` 映射到对应 step 脚本：

```bash
docker run --rm local/tm-cpu:latest --help      # 看 subcommand 列表
docker run --rm -v $(pwd)/data:/workspace/data \
    local/tm-cpu:latest data --skip-download    # Step 1
docker run --rm -v $(pwd)/data:/workspace/data \
    -v $(pwd)/models:/workspace/models \
    --gpus all local/tm-gpu:latest train --demo    # Step 3
docker run --rm local/tm-cpu:latest test        # pytest
docker run --rm -it local/tm-cpu:latest bash    # 进 shell
```

可用的 subcommand：`pipeline / data / tokenize / train / extract / detect / test /
bash`（详见 [`docker/entrypoint.sh`](docker/entrypoint.sh)）。

> Route A（银联 NDJSON）：`tokenize` / `train` / `extract` / `detect` 仍走
> TabFormer 分支；银联脚本（`step_01b` / `step_02_tokenize_ndjson` /
> `step_06_finetune_routec`）目前没接入 entrypoint subcommand，请进
> `bash` 后用 `python scripts/step_0X_*.py <args>` 直接调用。



### 使用预训练模型运行完整流程

```bash
# 将预训练模型放置到 models/decoder-foundation-model/ 下
python scripts/run_pipeline.py --steps data,tokenize,extract,detect

# 或使用 Make
make all-pretrained
```

### 从零训练

```bash
# 单 GPU demo (30 步)
python scripts/run_pipeline.py --steps all --demo

# 多 GPU 全量训练
torchrun --nproc-per-node=8 scripts/step_03_train_model.py \
    -c configs/training.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt \
    --validation_dataset.data_path data/decoder_corpus/val_corpus.txt
```

### 单步执行

```bash
python scripts/step_01_dataset_baseline.py              # 数据下载 + 分割
python scripts/step_02_tokenize_corpus.py               # 语料生成
python scripts/step_03_train_model.py --demo            # 训练 (demo)
python scripts/step_04_extract_embeddings.py             # 嵌入提取
python scripts/step_05_fraud_detection.py               # 欺诈检测对比
```

每步均支持 `--help` 查看可用参数。

### 银联（YL）NDJSON 路线：Route A + Route C

```bash
# 路线 A：NDJSON → tokenized corpus → Llama decoder CLM → 嵌入 → XGBoost
python scripts/step_01b_load_ndjson.py --ndjson-dir data/yl/raw   [--no-gpu]
python scripts/step_02_tokenize_ndjson.py --config dataset_yl --force
# ⚠️ 把上一步打印的 vocab_size 回填 configs/training_yl.yaml::model.config.vocab_size
python scripts/step_03_train_model.py --variant yl --demo
python scripts/step_04_extract_embeddings.py --dataset-config dataset_yl
python scripts/step_05_fraud_detection.py    --dataset-config dataset_yl

# 路线 C：Llama+GPT2+分类头 + LoRA + 业务损失（focal_with_amount / pAUC_sigmoid）
# 前置：路线 A 已产出 models/decoder-yl/ 与 data/yl/yl_tokenizer.json
pip install -e ".[routec]"          # 含 peft, transformers>=4.46
python scripts/step_06_finetune_routec.py --config configs/routec/default.json --demo --device cuda
```

> 无数据上手：仓库内置合成样例 [`examples/sample_data/smoke.jsonl`](examples/sample_data/smoke.jsonl)
> （20 用户 / 530 笔 / 10 正 10 负）。可直接跑 `pytest tests/` 与
> `python scripts/step_01b_load_ndjson.py --ndjson-dir examples/sample_data --no-gpu`，
> 详见 [`examples/README.md`](examples/README.md)。**smoke 数据只用于 pipeline 不崩，
> AUC/loss 数字无参考价值**。

## Pipeline 各步骤说明

### Step 1: 数据下载与基线

- 从数据源下载原始 CSV（默认 IBM TabFormer）
- 按日期累计行数做 80/10/10 时间分割
- 特征工程: `Hour` 从 `Time` 提取，`Amount` 去 `$` 和逗号，`_target` 二值化
- 保存 `val_eval.parquet` / `test_eval.parquet` 供后续评估
- XGBoost 基线 (13 维原始特征)

### Step 2: 语料库生成

- 加载时间分割后的 parquet
- `FinancialTokenizerPipeline`: preprocess → fit → transform
- 按用户/卡片分组，chunk 为 ~315 笔交易/序列
- 输出 `<bos> token1 token2 ... <sep> ... <eos>` 格式文本

### Step 3: 模型训练

- Decoder-Only Transformer (~29M 参数)
- Llama 架构: RoPE, GQA (8 Q heads / 2 KV heads), SwiGLU, RMSNorm
- NeMo AutoModel + FSDP2 分布式训练
- 因果语言建模 (每 token 都是训练信号)

### Step 4: 嵌入提取

- Last-token pooling 提取 512d 嵌入
- `__row_id__` 机制确保 preprocess 重排列后标签对齐
- 训练集平衡采样 (100 万, 10% 欺诈)
- 输出: `train_embeddings.npy`, `val_embeddings.npy`, `test_embeddings.npy`

### Step 5: 欺诈检测

- PCA 降维: 512d → 64d
- 三模型 XGBoost 对比:
  1. **Baseline**: 13d 原始特征
  2. **Embedding**: 64d PCA 嵌入
  3. **Combined**: 13d 原始 + 64d PCA
- 指标: ROC-AUC, Average Precision

## 在新数据集上续训练

本项目的配置驱动设计使得迁移到新的交易数据集非常简单。以下是需要修改的文件和步骤:

> **银联风控 NDJSON 数据**（risk_control_2 风格）有两条专属路线，详见
> [`upgrade/ylformer.md`](upgrade/ylformer.md)：
> - **路线 A**：NDJSON → tokenized corpus → 现有 NeMo + Llama decoder CLM
>   (`scripts/step_01b_load_ndjson.py` + `scripts/step_02_tokenize_ndjson.py`
>    + `scripts/step_03_train_model.py --variant yl`)
> - **路线 C**：Llama（route A 预训）+ GPT2（cross-txn 序列）+ 分类头 +
>   业务损失（金额加权 focal / pAUC）→
>   `python scripts/step_06_finetune_routec.py --config configs/routec/default.json`
>   （需要 route A 的预训 checkpoint 与 `data/yl/yl_tokenizer.json`）

### 1. 准备数据

将你的 CSV 数据放置到 `data/` 目录下，确保包含以下核心列（或根据实际情况调整）:

| 列名 | 类型 | 说明 | 必需 |
|------|------|------|------|
| `User` | int | 用户 ID | 是 |
| `Card` | int | 卡号 ID | 是 |
| `Year`, `Month`, `Day` | int | 交易日期 | 是 |
| `Time` | str | 交易时间 (HH:MM) | 建议保留 |
| `Amount` | str | 金额 (如 "$123.45") | 是 |
| `Is Fraud?` | str | 欺诈标签 ("Yes"/"No") | 是 |
| `Merchant Name` | str | 商户名称 | 是 |
| `MCC` | int | 商户类别代码 | 是 |
| `Use Chip`, `Merchant City`, `Merchant State`, `Zip` | str | 其他特征 | 可选 |

### 2. 修改配置文件

**`configs/dataset.yaml`** — 更新数据源:

```yaml
dataset:
  download_url: "你的数据下载地址"        # 或置空，手动放置 CSV
  raw_csv: "data/MyDataset/raw/data.csv"  # 你的 CSV 路径
  temporal_split_dir: "data/MyDataset/temporal_split"
  val_eval: "data/MyDataset/temporal_split/val_eval.parquet"
  test_eval: "data/MyDataset/temporal_split/test_eval.parquet"

split:
  train_ratio: 0.8
  val_ratio: 0.1

sampling:
  balanced_train_size: 1000000    # 根据数据规模调整
  eval_samples: 100000
  random_state: 42

feature_cols:                      # 根据你的列名调整
  - User
  - Card
  - Year
  - Month
  - Day
  - Hour
  - Amount
  - Use Chip
  - Merchant Name
  - Merchant City
  - Merchant State
  - Zip
  - MCC
```

**`configs/tokenizer.yaml`** — 调整分词参数:

```yaml
tokenizer:
  merchant_hash_size: 2000    # 商户哈希空间，商户数多时可增大
  chunk_size: 315             # 每序列交易数 (~4096 tokens)
  context_window: 4096        # 上下文窗口，与 training.yaml 的 seq_length 对应
  tokens_per_txn: 12          # 每笔交易 token 数

corpus:
  output_dir: "data/decoder_corpus"
  train: "data/decoder_corpus/train_corpus.txt"
  val: "data/decoder_corpus/val_corpus.txt"
  test: "data/decoder_corpus/test_corpus.txt"
```

**`configs/training.yaml`** — 调整模型和训练参数:

```yaml
model:
  config:
    vocab_size: 6251           # 需根据 tokenizer 重新计算
    hidden_size: 512           # 可增大至 768/1024 提升容量
    num_hidden_layers: 8       # 可调整
    num_attention_heads: 8
    num_key_value_heads: 2

step_scheduler:
  max_steps: 3000              # 根据数据量调整
  global_batch_size: 16
  local_batch_size: 16

paths:
  pretrained_model: "models/decoder-foundation-model"   # 续训起始 checkpoint
  train_corpus: "data/decoder_corpus/train_corpus.txt"
  val_corpus: "data/decoder_corpus/val_corpus.txt"
```

**`configs/xgboost.yaml`** — 调整欺诈检测参数:

```yaml
xgboost:
  params_raw:                  # 可能需要重新调参
    tree_method: "hist"
    n_estimators: 400
    ...
```

### 3. 适配数据加载器

如果你的数据列名与 TabFormer 不同，需要修改以下文件:

| 文件 | 适配内容 |
|------|---------|
| `constants.py` | `FRAUD_COL` 和 `FRAUD_POSITIVE_VALUES` — 欺诈标签列名和正值 |
| `data/feature.py` | `engineer_features()` — Hour 提取、Amount 清洗逻辑 |
| `data/split.py` | `add_date_column()` — 日期列构建方式 |
| `detection/xgboost.py` | `load_and_align_raw_features()` — 欺诈列名匹配 |
| `inference/extract.py` | `_extract_labels()` — 自动匹配多种欺诈列名 |

### 4. 执行续训练

```bash
# Step 1: 准备数据
python scripts/step_01_dataset_baseline.py

# Step 2: 生成语料库
python scripts/step_02_tokenize_corpus.py

# Step 3: 从预训练 checkpoint 续训练
torchrun --nproc-per-node=8 scripts/step_03_train_model.py \
    -c configs/training.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt \
    --validation_dataset.data_path data/decoder_corpus/val_corpus.txt

# Step 4: 提取新模型的嵌入
python scripts/step_04_extract_embeddings.py --force

# Step 5: 欺诈检测评估
python scripts/step_05_fraud_detection.py
```

### 5. 词汇表更新 (如需要)

新数据集可能引入未见过的类别值。`FinancialTokenizerPipeline` 的 `fit()` 方法会自动从数据中学习映射，但 `vocab_size` 需要更新:

```python
from transaction_model.tokenizer import FinancialTabularTokenizer

tokenizer = FinancialTabularTokenizer(
    merchant_hash_size=2000,
    category_hierarchy=True,
    temporal_encoding=True,
)
print(f"Vocab size: {tokenizer.get_vocab_size()}")
```

将输出的 `vocab_size` 更新到 `configs/training.yaml` 的 `model.config.vocab_size` 字段。

## GPU / CPU 自适应

| 步骤 | CPU | GPU | 说明 |
|------|-----|-----|------|
| Step 1 数据基线 | ✅ | ✅ 自动 cuDF 加速 | XGBoost 自动选 cuda/cpu |
| Step 1b NDJSON 加载（YL） | ✅（`--no-gpu` 走 pandas） | ✅ cuDF 加速 | Route A 入口 |
| Step 2 语料生成 | ❌ | ✅ **必需 cuDF** | `FinancialTokenizerPipeline.preprocess` 使用 cuDF 字符串/日期/hash |
| Step 2b NDJSON token 化（YL）| ❌ | ✅ **必需 cuDF** | 同 Step 2，YLPipeline.preprocess 走 cuDF |
| Step 3 模型训练 | ❌ | ✅ **必需 CUDA** | NeMo AutoModel + FSDP2 |
| Step 4 嵌入提取 | ❌ | ✅ **必需 cuDF + CUDA** | 复用 Step 2 的 tokenizer + GPU 模型推理 |
| Step 5 欺诈检测 | ✅ | ✅ 自动 | XGBoost 自动检测 cuda |
| Step 6 Route C 微调 | ✅（demo 可 `--device cpu`） | ✅ 建议 | `python scripts/step_06_finetune_routec.py`；trainer 走 torch 原生 DDP（无 DeepSpeed/FSDP） |

## Route C 速查

| 配置维度 | 取值 |
|----------|------|
| `task_type` | `lora`（默认）/ `freeze` / `all_params` |
| `llama.pool_mode` | `last_token` / `mean` / `cls` |
| `loss_fn.name` | `sft_cross_loss` / `sft_focal_loss_weight` / `sft_focal_loss_with_amount`（默认）/ `sft_pAUC_sigmoid_loss` |
| `lora.target_modules` 默认 | `["q_proj", "v_proj"]` |
| `strategy.name` | `single`（默认）/ `ddp`（多机多卡用） |
| `data_config.shard_mode` | `sampler`（默认，DistributedSampler）/ `file`（与 risk_control_2 一致，文件分片） |
| 输出 | `models/routec/ckpt_step*.pt`、`log/routec/<step>_val_prob.json`（每用户概率） |

### 多机多卡部署

> Route C 内置 torch native DDP（无 DeepSpeed 依赖）；Route A 走 NeMo FSDP2。
> 单卡入口完全向后兼容——未设 LOCAL_RANK 环境变量时 `step_06` 自动退化为单进程。

**单机多卡 Route C**：
```bash
NGPUS=8 bash scripts/routec_ddp_single_node.sh
# 或手动：
torchrun --nproc-per-node=8 --master-port=29500 \
    scripts/step_06_finetune_routec.py \
    --config configs/routec/default_multinode.json
```

**多机多卡 Route C**（每个节点各起一份，仅 `NNODE_RANK` 不同）：
```bash
# node 0 (master)
NNODES=2 NNODE_RANK=0 NGPUS=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/routec_ddp_multi_node.sh

# node 1
NNODES=2 NNODE_RANK=1 NGPUS=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
    bash scripts/routec_ddp_multi_node.sh
```

**多机多卡 Route A（NeMo FSDP2）**：
```bash
# 在每个节点上跑（仅 NODE_RANK 不同）。NeMo 自动从 world_size 推 FSDP2 dp_size。
torchrun --nproc-per-node=8 --nnodes=2 --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    scripts/step_03_train_model.py --variant yl --max-steps 3000
```

数据分片：
- **`shard_mode: "sampler"`**（小数据 / 默认）：torch `DistributedSampler` 切分
- **`shard_mode: "file"`**（大数据 / 千万用户级）：每个 rank 仅读 `record_idx % world_size == rank` 的 NDJSON record，与 risk_control_2 一致

关键约束（多机）：
1. **每个节点的 python/torch/transformers/peft 版本必须一致**（建议同一个 docker 镜像）
2. **`models/decoder-yl/` checkpoint 与 `data/yl/yl_tokenizer.json` 必须每个节点 path-reachable**（NFS 或 rsync）
3. **`MASTER_ADDR` 必须是所有节点都能解析的 IP**（master 节点的主网卡 IP）
4. **训练前每节点先 `docker login nvcr.io` 拉镜像**（参考 README §"手动 docker build")

| 步骤 | CPU | GPU | 说明 |
|------|-----|-----|------|
| 可视化 | ✅ | ✅ 可选 cuML | UMAP 优先 cuML，自动回退 sklearn |

**摘要**: CPU 环境可以运行 Step 1 / Step 1b（`--no-gpu`）/ Step 5 / 可视化；Step 2/2b 需要 cuDF；Step 3/4/6 涉及 CUDA。在仅 CPU 的开发机上能 `import transaction_model.tokenizer`（不会抛 `ImportError`），但调用 `preprocess` / `transform` 会抛 `ImportError` 报告缺失的 GPU 库。

## 开发

```bash
# 安装开发依赖（默认仅 dev；要跑 route C 测试再加 routec）
pip install -e ".[dev,routec]"

# 运行全部测试 —— 25 passed / 1 skipped (GPU-only)
pytest tests/ -v
# 仅 Route C 集成测试（CPU 即可）
pytest tests/test_routec_combined.py -v

# 代码格式检查
make test
```

## 许可证
