"""XGBoost 模型对比可视化"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_model_comparison(
    metrics_list: list[dict],
    labels: list[str],
    feat_dims: list[str],
    save_path: str | None = None,
) -> None:
    """绘制三模型 ROC-AUC 和 Average Precision 对比图

    Args:
        metrics_list: [{'test_auc': ..., 'test_ap': ...}, ...]
        labels: 模型标签
        feat_dims: 特征维度标签
        save_path: 图片保存路径
    """
    model_labels = [f'{l}\n({d})' for l, d in zip(labels, feat_dims)]
    test_aucs = [m['test_auc'] for m in metrics_list]
    test_aps = [m['test_ap'] for m in metrics_list]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    best_auc = test_aucs.index(max(test_aucs))
    best_ap = test_aps.index(max(test_aps))

    # AUC
    bars1 = ax1.bar(model_labels, test_aucs, color=colors, edgecolor='black', linewidth=1)
    for i, (bar, val) in enumerate(zip(bars1, test_aucs)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    bars1[best_auc].set_edgecolor('red')
    bars1[best_auc].set_linewidth(3)
    ax1.axhline(y=test_aucs[0], color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_ylabel('ROC-AUC Score', fontsize=12)
    ax1.set_title('ROC-AUC (Overall Ranking)', fontsize=13, fontweight='bold')
    ax1.set_ylim(min(test_aucs) * 0.95, max(test_aucs) * 1.02)
    ax1.grid(axis='y', alpha=0.3)

    # AP
    bars2 = ax2.bar(model_labels, test_aps, color=colors, edgecolor='black', linewidth=1)
    for i, (bar, val) in enumerate(zip(bars2, test_aps)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    bars2[best_ap].set_edgecolor('red')
    bars2[best_ap].set_linewidth(3)
    ax2.axhline(y=test_aps[0], color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel('Average Precision', fontsize=12)
    ax2.set_title('Average Precision (Fraud Flagging Quality)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(test_aps) * 1.25)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Fraud Detection: Model Comparison (Test Set)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
