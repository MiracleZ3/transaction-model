"""Route C CombinedModel forward/backward 集成测试（CPU + toy 模型）。

不依赖真实 route A checkpoint —— 用 LlamaConfig 构造一个 hidden=32, 1 layer 的 toy
Llama transformer，确保 [B,T,L] → logits 的前向、损失反传都不报错。
"""
from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from transaction_model.finetune.models import (
    ClassifierHead, CombinedModel, GPT2SeqEncoder, LlamaEncoder,
)


def _build_toy_combined(
    hidden: int = 32,
    n_layers: int = 1,
    n_heads: int = 2,
    max_len: int = 8,
    dt_dim: int = 8,
    pool_mode: str = "last_token",
) -> CombinedModel:
    """构造一个 toy CombinedModel（fake vocab，不依赖 route A checkpoint）。"""
    llama_cfg = LlamaConfig(
        vocab_size=50,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,  # toy 用 MHA 简化
        max_position_embeddings=max_len,
        rms_norm_eps=1e-5,
    )
    llama = LlamaEncoder(
        model_path=None,
        hidden_size=hidden,
        pool_mode=pool_mode,
        freeze=False,
        lora_cfg=None,
        pad_token_id=0,
        llama_config=llama_cfg,
    )
    gpt2 = GPT2SeqEncoder(
        bert_hidden_size=hidden,
        gpt2_hidden_size=hidden,
        n_layers=1,
        n_heads=n_heads,
        max_len=max_len,
        dt_dim=dt_dim,
        pad_token_id=0,
    )
    cls = ClassifierHead(
        gpt2_hidden_size=hidden,
        num_class=2,
        cls="cls",
        dropout=0.0,
    )
    return CombinedModel(llama=llama, gpt2=gpt2, classifier=cls)


def _make_batch(B=2, T=4, L=6):
    """构造 [B,T,L] 的 toy batch（token id 在 0..49，mask 同 input）。"""
    torch.manual_seed(42)
    input_ids = torch.randint(1, 50, (B, T, L))
    attention_mask = torch.ones_like(input_ids)
    gpt2_mask = torch.ones((B, T), dtype=torch.bool)
    delta_t = torch.randint(0, 24, (B, T)).long()  # 小时差
    lens_in = torch.full((B,), T, dtype=torch.long)
    label = torch.tensor([0, 1][:B])
    amount = torch.tensor([100.0, 5000.0][:B])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "gpt2_attention_mask": gpt2_mask,
        "delta_time_stap": delta_t,
        "lens_in": lens_in,
        "label": label,
        "amount": amount,
        "user": ["u0", "u1"][:B],
    }


def test_forward_output_shape():
    model = _build_toy_combined()
    batch = _make_batch(B=2, T=4, L=6)
    logits = model(batch)
    assert logits.shape == (2, 2), f"expected (2,2) got {logits.shape}"
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("pool_mode", ["last_token", "mean", "cls"])
def test_pool_modes_work(pool_mode):
    model = _build_toy_combined(pool_mode=pool_mode)
    batch = _make_batch()
    logits = model(batch)
    assert logits.shape == (2, 2)


def test_backward_passes():
    """端到端 backward：CombinedModel 应能反传到 Llama + GPT2 + Classifier。"""
    model = _build_toy_combined(hidden=32, n_layers=1)
    batch = _make_batch()

    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()

    # 三个子模块都应该有梯度信号
    has_llama_grad = any(
        p.grad is not None and torch.isfinite(p.grad).any()
        for p in model.llama.parameters() if p.requires_grad
    )
    has_gpt2_grad = any(
        p.grad is not None and torch.isfinite(p.grad).any()
        for p in model.gpt2.parameters() if p.requires_grad
    )
    has_cls_grad = any(
        p.grad is not None and torch.isfinite(p.grad).any()
        for p in model.classifier.parameters() if p.requires_grad
    )
    assert has_llama_grad, "no gradient on Llama"
    assert has_gpt2_grad, "no gradient on GPT2"
    assert has_cls_grad, "no gradient on ClassifierHead"


def test_with_real_loss_focal_with_amount():
    """端到端：CombinedModel + focal_with_amount 损失。"""
    from transaction_model.finetune.losses import build_loss

    model = _build_toy_combined(hidden=32)
    batch = _make_batch()
    logits = model(batch)
    loss_fn = build_loss("sft_focal_loss_with_amount")
    loss = loss_fn(logits, batch["label"], amount=batch["amount"])
    assert torch.isfinite(loss)
    loss.backward()  # 不应报错


def test_collate_fn_creates_valid_batch():
    """collate_fn 应把 4 条变长样本拼成 [B,T,L]。"""
    from transaction_model.finetune.data import collate_fn

    # 构造 4 条样本：长度 3 / 5 / 2 / 4；L=6
    L = 6
    samples = []
    for n in [3, 5, 2, 4]:
        samples.append({
            "input_ids": torch.randint(1, 50, (n, L)).numpy(),
            "attention_mask": torch.ones((n, L), dtype=torch.long).numpy(),
            "his_time_stap": torch.zeros(n, dtype=torch.long).numpy(),
            "delta_time_stap": torch.zeros(n, dtype=torch.long).numpy(),
            "label": 0,
            "amount": 100.0,
            "user": f"u{n}",
        })
    batch = collate_fn(samples, pad_token_id=0, max_hiswindow=512)
    assert batch["input_ids"].shape == (4, 5, L)  # T=5 = max
    # lens_in 准确反映各样本长度
    assert batch["lens_in"].tolist() == [3, 5, 2, 4]
    # gpt2_attention_mask 严格 True 到 lens_in
    for b, n in enumerate([3, 5, 2, 4]):
        assert batch["gpt2_attention_mask"][b, :n].all()
        if n < 5:
            assert not batch["gpt2_attention_mask"][b, n:].any()
