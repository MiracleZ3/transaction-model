# 示例数据与冒烟测试

本目录提供仓库内置的**最小可用冒烟测试数据**，避免每位 clone 者（包括 CI）
再为找数据样例而头疼。**只有冒烟测试价值，不能拿来判断模型性能**——样本规模
与分布都远不足以训练。

## 文件清单

| 文件 | 用途 |
|------|------|
| `sample_data/smoke.jsonl` | 20 用户 / 530 笔交易的合成 NDJSON；10 正 / 10 负 |
| `generate_smoke_sample.py` | 上面的生成脚本，可重跑以刷新 sample |

## smoke.jsonl 字段格式

与 `../risk_control_2/docs/data_schema.md` §1 完全一致，每行一个用户的 JSON：

```jsonc
{
  "cert_sm3": "u1",            // 用户标识
  "cert_type": "cert",         // 固定
  "trans": [ ...40 笔... ],    // 交易序列，每笔 22 字段（前 20 被代码消费）
  "label": 1                   // 用户级二分类（1=风险）
}
```

字段下标与 YL_FIELDS_POS 一一对应（见 `transaction_model/constants.py`）。

## 重建 sample

```bash
python examples/generate_smoke_sample.py
# → Wrote 20 users → examples/sample_data/smoke.jsonl
```

如要换 seed 或扩规模，编辑脚本里的 `random.Random(42)`、`range(1, 21)`。

## 端到端冒烟（无需 GPU、无需真实数据）

clone 后在 CPU 上即可跑通：

```bash
# 0. 安装
pip install -e ".[dev]"             # 跑 route C 测试再追加 routec: pip install -e ".[dev,routec]"

# 1. 单元 + 集成测试（CPU）
pytest tests/ -v                    # ≈ 25 passed / 1 skipped（GPU-only）
# 只有 test_ndjson_loader.py 在 smoke.jsonl 存在时跑 real-sample 用例；其余用 fixture/toy 模型。

# 2. Route A 路线：NDJSON → 行级 parquet（step_01b 是 CPU friendly 的）
python scripts/step_01b_load_ndjson.py \
    --ndjson-dir examples/sample_data \
    --no-gpu
# 输出路径由 configs/dataset_yl.yaml 的 dataset.temporal_split_dir 决定
# （默认 <repo>/data/yl/temporal_split/{train,val,test}.parquet）。

# 3. Route A 路线：行级 parquet → tokenized corpus
#    需 GPU + cuDF（CPU 跑不了）。命令：(pip install -e ".[gpu]" 先)
#    python scripts/step_02_tokenize_ndjson.py --config dataset_yl
#    产出 data/yl/decoder_corpus/{train,val,test}_corpus.txt + data/yl/yl_tokenizer.json

# 4. Route C 集成测试（CPU 即可，不读 sample）：
pytest tests/test_routec_combined.py -v

# 5. Route C 实际微调需要 route A 预训 ckpt（不在 smoke 范围内），见 How_To_Use.md §13。
```

## 重要：smoke 数据 ≠ 训练数据

| 项 | smoke.jsonl | 真实训练数据 |
|----|-------------|--------------|
| 用户数 | 20 | 千万级 |
| 每用户交易数 | 20–35 | 几十到几百 |
| 字段基数 | 4–8 个/字段 | 几十到几千 |
| 序列重复 | 0（去重保证） | — |
| 标签平衡 | 50/50 模拟 | 业务真实分布 |
| 适用 | pipeline 不崩 / API 正确 | 训练评估 / AUC / pAUC |

拿 smoke 跑出的 loss / AUC 数字毫无参考价值，断言模型能力之前必须先换真实数据。
