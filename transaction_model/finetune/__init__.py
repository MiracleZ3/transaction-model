"""Route C: Llama (前置 per-txn encoder) + GPT2 (跨交易时序) + 分类头的混合微调。

与 Route A（NeMo + 纯 decoder CLM）解耦，互不影响。

入口：
    - 训练：scripts/step_06_finetune_routec.py
    - 损失：transaction_model.finetune.losses.LOSS_REGISTRY
    - 模型：transaction_model.finetune.models.CombinedModel
    - 数据：transaction_model.finetune.data.SftNDJsonDataset + prepare_collate
"""

__all__ = ["losses", "models", "data", "trainer", "config"]
