"""Route C 损失 registry。

调用约定：所有 loss wrapper 的返回值是 `loss(logits, label, amount=None)`，
其中：
  - logits: `Tensor[B, num_class]`（下游分类头输出，通常 num_class=2）
  - label:  `Tensor[B]`，二分类
  - amount: `Tensor[B] | None`，正样本金额（用于 focal_with_amount 路线）

不传 amount 或传 None 时，focal_with_amount 退化为不加权。
"""
from __future__ import annotations

from typing import Callable, Dict

from .impl import (
    cEloss_wrapper,
    focal_weight_wrapper,
    focal_with_amount_wrapper,
    pAUC_sigmoid_wrapper,
)


# name → factory. 每个 factory 接受任意 kwargs（透传给损失构造），返回
# 一个 callable(logits, label, amount=None) -> scalar tensor。
LOSS_REGISTRY: Dict[str, Callable] = {
    "sft_cross_loss":               cEloss_wrapper,
    "sft_focal_loss_weight":        focal_weight_wrapper,
    "sft_focal_loss_with_amount":   focal_with_amount_wrapper,
    "sft_pAUC_sigmoid_loss":        pAUC_sigmoid_wrapper,
}


def build_loss(name: str, **params) -> Callable:
    """按名称构造 loss。未知名称抛 ValueError。"""
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name](**params)


__all__ = ["LOSS_REGISTRY", "build_loss"]
