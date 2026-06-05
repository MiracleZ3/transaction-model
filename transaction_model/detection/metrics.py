"""评估指标计算"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """计算二分类评估指标

    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测概率 (0~1)

    Returns:
        {"auc": ..., "ap": ...}
    """
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "ap": average_precision_score(y_true, y_pred),
    }


def print_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """打印评估指标"""
    print(f"  {prefix} ROC-AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f}")


def print_results_summary(
    results: list[dict],
    labels: list[str],
    feat_dims: list[str],
) -> None:
    """打印三模型对比总结

    Args:
        results: [{'val_auc': ..., 'val_ap': ..., 'test_auc': ..., 'test_ap': ...}, ...]
        labels: 模型标签
        feat_dims: 特征维度标签
    """
    all_aucs = [r['test_auc'] for r in results]
    all_aps = [r['test_ap'] for r in results]
    best_auc_idx = all_aucs.index(max(all_aucs))
    best_ap_idx = all_aps.index(max(all_aps))

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Test Set - Final Holdout)")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Features':>10} {'ROC-AUC':>10} {'Avg Prec':>10}")
    print("-" * 70)

    for i, (label, dim, m) in enumerate(zip(labels, feat_dims, results)):
        badges = []
        if i == best_auc_idx:
            badges.append("AUC")
        if i == best_ap_idx:
            badges.append("AP")
        badge_str = " * " + ",".join(badges) if badges else ""
        print(f"  {label:<33} {dim:>10} {m['test_auc']:>10.4f} {m['test_ap']:>10.4f}{badge_str}")

    # Lift over baseline
    baseline = results[0]
    print("\n" + "=" * 70)
    print("LIFT OVER BASELINE (Test Set)")
    print("=" * 70)
    print(f"\n{'Model':<35} {'ROC-AUC Lift':>15} {'AP Lift':>15}")
    print("-" * 68)
    for i, (label, m) in enumerate(zip(labels[1:], results[1:])):
        lift_auc = (m['test_auc'] - baseline['test_auc']) / baseline['test_auc'] * 100
        lift_ap = (m['test_ap'] - baseline['test_ap']) / baseline['test_ap'] * 100
        print(f"  {label:<33} {'+' if lift_auc > 0 else ''}{lift_auc:>13.2f}% "
              f"{'+' if lift_ap > 0 else ''}{lift_ap:>13.2f}%")

    # 关键洞察
    if best_auc_idx != best_ap_idx:
        print(f"\n  NOTE: AUC and AP disagree on the best model!")
        print(f"     Best ROC-AUC: {labels[best_auc_idx]} ({all_aucs[best_auc_idx]:.4f})")
        print(f"     Best AP: {labels[best_ap_idx]} ({all_aps[best_ap_idx]:.4f})")
    else:
        print(f"\n  Both metrics agree: {labels[best_auc_idx]} is the best model.")
