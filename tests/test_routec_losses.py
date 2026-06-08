"""Route C 业务损失数值合理性测试（纯 CPU）。

不验证相对业务损失绝对值，只验证关键契约：
  - 所有 loss forward 不报错且有限
  - focal_with_amount：amount 增大 → 正样本权重增大 → 损失增大
  - focal_weight：pos_weight 增大 → 正样本损失不变但梯度变大（同 logits 下，w>1 比 w=1 loss 大）
  - pAUC：单类 batch（全正或全负）返回 0
  - cross：与 pos_weight 无关
"""
from __future__ import annotations

import pytest
import torch

from transaction_model.finetune.losses import LOSS_REGISTRY, build_loss


def _make_logits_and_labels(seed: int = 0):
    """制造 [B=4, num_class=2] logits + 标签 (2正2负)。"""
    torch.manual_seed(seed)
    # 故意让模型对正样本比较确信（logit 高），以让 focal loss 数值稳定
    logits = torch.tensor([
        [3.0, -1.0],   # 负样本（class 0）
        [-2.0, 2.5],   # 正样本（class 1）
        [1.5, -0.5],   # 负样本
        [-1.0, 2.0],   # 正样本
    ], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1])
    return logits, labels


def test_all_losses_in_registry():
    """4 个注册名都应存在。"""
    expected = {
        "sft_cross_loss",
        "sft_focal_loss_weight",
        "sft_focal_loss_with_amount",
        "sft_pAUC_sigmoid_loss",
    }
    assert expected.issubset(set(LOSS_REGISTRY))


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("nonexistent_loss")


def test_all_losses_finite():
    """全部 loss 在标准 (B=4) batch 上 forward 返回有限标量。"""
    logits, labels = _make_logits_and_labels()
    for name in LOSS_REGISTRY:
        loss_fn = build_loss(name)
        out = loss_fn(logits, labels, amount=torch.tensor([0.0, 100.0, 0.0, 500.0]))
        assert torch.is_tensor(out), f"{name} should return tensor"
        assert torch.isfinite(out), f"{name} loss not finite: {out}"
        assert out.dim() == 0, f"{name} should be scalar"


def test_focal_with_amount_monotonic():
    """正向行为契约：amount 越大，正样本损失权重越大 → 总损失越大。"""
    logits, labels = _make_logits_and_labels()
    loss_fn = build_loss("sft_focal_loss_with_amount", pos_only=True)

    # 小金额
    amount_small = torch.tensor([0.0, 1000.0, 0.0, 1000.0])
    loss_small = loss_fn(logits, labels, amount=amount_small)

    # 大金额（正样本金额放大 100 倍）
    amount_large = torch.tensor([0.0, 1_000_000.0, 0.0, 1_000_000.0])
    loss_large = loss_fn(logits, labels, amount=amount_large)

    # 大额加权 + amount_clip=4.0，loss_large 应显著大于 loss_small
    assert loss_large > loss_small
    # 关键不变式：增大正样本金额不能降低 loss
    print(f"  small_amount loss={loss_small.item():.4f}, large={loss_large.item():.4f}")


def test_focal_with_amount_none_amount_matches_unit():
    """focal_with_amount(amount=None) 应该等于 amount=ones (退化为不加权)。"""
    logits, labels = _make_logits_and_labels()
    loss_fn = build_loss("sft_focal_loss_with_amount")
    out_none = loss_fn(logits, labels, amount=None)
    out_ones = loss_fn(logits, labels, amount=torch.ones(logits.size(0)))
    assert torch.allclose(out_none, out_ones, atol=1e-6)


def test_focal_weight_responds_to_pos_weight():
    """focal_weight 的 pos_weight 越大，loss 越大（仅作用于类 1）。"""
    logits, labels = _make_logits_and_labels()
    loss_low = build_loss("sft_focal_loss_weight", pos_weight=1.0)(logits, labels)
    loss_high = build_loss("sft_focal_loss_weight", pos_weight=50.0)(logits, labels)
    assert loss_high > loss_low


def test_cross_loss_independent_of_amount():
    """cross 不应该受 amount 影响。"""
    logits, labels = _make_logits_and_labels()
    loss_fn = build_loss("sft_cross_loss")
    out_a = loss_fn(logits, labels, amount=None)
    out_b = loss_fn(
        logits, labels,
        amount=torch.tensor([0.0, 1e10, 0.0, 1e10]),
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_pAUC_single_class_returns_zero():
    """pAUC 在单类 batch 上应有意义地返回 0（不崩）。"""
    logits, labels = _make_logits_and_labels()
    all_pos = torch.ones_like(labels)
    all_neg = torch.zeros_like(labels)
    loss_fn = build_loss("sft_pAUC_sigmoid_loss", fpr_max=0.1)
    pos_only = loss_fn(logits, all_pos)
    neg_only = loss_fn(logits, all_neg)
    assert pos_only.item() == 0.0
    assert neg_only.item() == 0.0


def test_pAUC_mixed_batch_positive():
    """混合 batch 时 pAUC 应返回正数。"""
    logits, labels = _make_logits_and_labels()
    loss_fn = build_loss("sft_pAUC_sigmoid_loss", fpr_max=0.5)
    out = loss_fn(logits, labels)
    assert out.item() > 0
    # sigmoid 输出范围 [0, 1]
    assert out.item() < 1.0 + 1e-6


def test_loss_gradient_backprop():
    """所有可微 loss 应能反传梯度到 logits。"""
    logits, labels = _make_logits_and_labels()
    logits.requires_grad_(True)
    loss_fn = build_loss("sft_focal_loss_with_amount")
    out = loss_fn(
        logits, labels,
        amount=torch.tensor([0.0, 5000.0, 0.0, 8000.0]),
    )
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
