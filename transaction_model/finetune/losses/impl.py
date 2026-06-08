"""Route C 业务损失（移植自 risk_control_2/losses/，去掉 core.registry 依赖）。

每个 loss 是一个 wrapper 工厂，调用签名约定见 __init__.py。

支持的损失：
  - sft_cross_loss          纯交叉熵，作对照基线
  - sft_focal_loss_weight   focal + pos_weight（无金额加权）
  - sft_focal_loss_with_amount  金额加权 focal（核心）
  - sft_pAUC_sigmoid_loss   partial AUC pairwise sigmoid（风控金标准）
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cEloss_wrapper(**kwargs):
    """纯交叉熵基线。amount 参数被忽略。"""
    def CEloss(logits, labels, amount=None):
        labels = labels.to(logits.device)
        return F.cross_entropy(logits, labels)
    return CEloss


def focal_weight_wrapper(alpha: float = 0.25, gamma: float = 2.0,
                         reduction: str = "mean", pos_weight: float = 10.0, **kwargs):
    """focal + pos_weight（无金额加权）。

    pos_weight 仅作用于 logits.size(1) > 1 时的类别 1，等价于 F.cross_entropy
    里 `weight` 参数把类 1 放大。
    """
    def FocalLoss(logits, labels, amount=None):
        device = logits.device
        labels = labels.to(device)

        weights = torch.ones(logits.size(1), device=device)
        if logits.size(1) > 1:
            weights[1] = pos_weight

        cross_entropy_loss = F.cross_entropy(
            logits, labels, weight=weights, reduction="none"
        )
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = alpha * (1 - pt) ** gamma * cross_entropy_loss

        if reduction == "mean":
            return focal_loss.mean()
        if reduction == "sum":
            return focal_loss.sum()
        return focal_loss
    return FocalLoss


def focal_with_amount_wrapper(alpha: float = 0.25, gamma: float = 2.0,
                              reduction: str = "mean", amount_clip: float = 4.0,
                              pos_only: bool = True, **kwargs):
    """金额加权 focal —— Route C 的核心损失。

    amount 越大，正样本的权重越大（上限 amount_clip）。
    金额加权公式（与 risk_control_2 一致）：weight = clamp(0.5 * amount / 100k, 1, amount_clip)。
    amount=None 或缺省时退化为不加权 focal（pos_only=False 时全样本权重 1.0）。
    """
    def FocalLoss(logits, labels, amount=None):
        device = logits.device
        labels = labels.to(device)
        if amount is None:
            amounts = torch.ones(logits.size(0), device=device)
        else:
            amounts = amount.to(device).float()

        cross_entropy_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = alpha * (1 - pt) ** gamma * cross_entropy_loss

        amounts_k = amounts / 100000.0
        amounts_weight = 0.5 * amounts_k
        amounts_weight = torch.clamp(amounts_weight, min=1.0, max=amount_clip)
        if pos_only:
            pos_mask = (labels == 1).float()
            final_weight = amounts_weight * pos_mask + 1.0 * (1 - pos_mask)
        else:
            final_weight = amounts_weight
        focal_loss = final_weight * focal_loss

        if reduction == "mean":
            return focal_loss.mean()
        if reduction == "sum":
            return focal_loss.sum()
        return focal_loss
    return FocalLoss


def pAUC_sigmoid_wrapper(fpr_max: float = 0.1, **kwargs):
    """partial AUC pairwise sigmoid loss。

    取 batch 内 top-k 困难负样本（k = fpr_max * num_neg），与所有正样本构成
    pairwise 损失；最小化 sigmoid(score_neg - score_pos)。

    需要 batch 内同时含正负样本，否则返回 0。
    """
    def pAUCSigmoidLoss(logits, labels, amount=None):
        device = logits.device
        labels = labels.to(device)
        assert 0 <= fpr_max <= 1, f"fpr_max got {fpr_max}"

        if logits.dim() == 2:
            logits = logits[:, 1]

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_scores = logits[pos_mask]
        neg_scores = logits[neg_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            # 当前 batch 单类，pairwise 无意义
            return torch.tensor(
                0.0, device=logits.device, requires_grad=True
            )

        num_neg = neg_scores.size(0)
        k = max(1, int(fpr_max * num_neg))
        top_k_neg_scores, _ = torch.topk(neg_scores, k=k)

        diff = pos_scores.unsqueeze(1) - top_k_neg_scores.unsqueeze(0)
        loss = torch.sigmoid(-diff)
        return loss.mean()
    return pAUCSigmoidLoss
