"""嵌入空间可视化 (UMAP)"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from transaction_model.constants import (
    UMAP_AXIS_RANGE,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_VIZ_SIZE,
)


def _get_umap():
    """获取 UMAP 实现（cuML GPU 优先，sklearn 回退）"""
    try:
        from cuml.manifold import UMAP
        return UMAP, True
    except ImportError:
        from sklearn.manifold import UMAP
        print("cuML not available, using sklearn UMAP (CPU)")
        return UMAP, False


def run_umap_2d(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    viz_size: int = UMAP_VIZ_SIZE,
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """运行 2D UMAP 降维

    Returns:
        (umap_2d, subset_labels, indices)
    """
    UMAP, is_gpu = _get_umap()

    np.random.seed(42)
    if len(embeddings) > viz_size:
        indices = np.random.choice(len(embeddings), viz_size, replace=False)
    else:
        indices = np.arange(len(embeddings))

    subset = embeddings[indices]
    subset_labels = labels[indices] if labels is not None else None

    print(f"Running {'GPU' if is_gpu else 'CPU'} UMAP on {len(subset):,} samples...")
    umap_kwargs = dict(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=2,
        min_dist=UMAP_MIN_DIST,
        metric='euclidean',
        random_state=42,
    )

    if is_gpu:
        import cupy as cp
        embeds_gpu = cp.asarray(subset)
        umap = UMAP(**umap_kwargs)
        umap_2d = umap.fit_transform(embeds_gpu)
        umap_2d = cp.asnumpy(umap_2d)
    else:
        umap = UMAP(**umap_kwargs)
        umap_2d = umap.fit_transform(subset)

    print(f"UMAP complete: {umap_2d.shape}")
    return umap_2d, subset_labels, indices


def plot_umap_2d(
    umap_2d: np.ndarray,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """绘制 2D UMAP 散点图 (Fraud vs Normal)"""
    plt.figure(figsize=(12, 8))
    if labels is not None:
        mask_normal = (labels == 0)
        mask_fraud = (labels == 1)
        plt.scatter(umap_2d[mask_normal, 0], umap_2d[mask_normal, 1],
                     c="blue", alpha=0.08, s=0.7, label="Normal")
        plt.scatter(umap_2d[mask_fraud, 0], umap_2d[mask_fraud, 1],
                     c="red", alpha=0.6, s=10, label="Fraud", edgecolor="k", linewidth=0.1)
        plt.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
                   markersize=10, alpha=0.6, label="Normal"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=10, alpha=0.9, label="Fraud"),
        ], loc="upper right")
    else:
        plt.scatter(umap_2d[:, 0], umap_2d[:, 1], alpha=0.3, s=1)

    plt.title(f"Transaction Embeddings (UMAP, n={len(umap_2d):,})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.xlim(-UMAP_AXIS_RANGE, UMAP_AXIS_RANGE)
    plt.ylim(-UMAP_AXIS_RANGE, UMAP_AXIS_RANGE)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_umap_3d(
    embeddings: np.ndarray,
    viz_size: int = UMAP_VIZ_SIZE,
) -> np.ndarray:
    """运行 3D UMAP 降维

    Returns:
        umap_3d array (n_samples, 3)
    """
    UMAP, is_gpu = _get_umap()

    np.random.seed(42)
    if len(embeddings) > viz_size:
        indices = np.random.choice(len(embeddings), viz_size, replace=False)
        subset = embeddings[indices]
    else:
        subset = embeddings

    print(f"Running {'GPU' if is_gpu else 'CPU'} 3D UMAP on {len(subset):,} samples...")
    umap_kwargs = dict(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=3,
        min_dist=UMAP_MIN_DIST,
        metric='euclidean',
        random_state=42,
    )

    if is_gpu:
        import cupy as cp
        embeds_gpu = cp.asarray(subset)
        umap = UMAP(**umap_kwargs)
        umap_3d = umap.fit_transform(embeds_gpu)
        umap_3d = cp.asnumpy(umap_3d)
    else:
        umap = UMAP(**umap_kwargs)
        umap_3d = umap.fit_transform(subset)

    print(f"3D UMAP complete: {umap_3d.shape}")
    return umap_3d


def create_3d_interactive_plot(
    umap_2d: np.ndarray,
    umap_3d: np.ndarray,
    raw_df: pd.DataFrame,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
) -> str:
    """创建 Plotly 3D 交互可视化

    Returns:
        HTML 文件路径
    """
    import plotly.graph_objects as go

    n = len(umap_2d)
    if labels is not None:
        color_array = np.where(labels == 1, 'Fraud', 'Normal')
    else:
        color_array = np.full(n, 'Unknown')

    fig = go.Figure()
    for label_name, color in [('Normal', 'blue'), ('Fraud', 'red')]:
        mask = color_array == label_name
        fig.add_trace(go.Scatter3d(
            x=umap_3d[mask, 0], y=umap_3d[mask, 1], z=umap_3d[mask, 2],
            mode='markers',
            marker=dict(size=2 if label_name == 'Normal' else 5, color=color, opacity=0.5),
            name=label_name,
        ))

    fig.update_layout(
        title=f'3D Transaction Embeddings (n={n:,})',
        scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'),
        width=900, height=700,
    )

    if save_path is None:
        save_path = "data/embeddings/umap_3d_interactive.html"

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path, include_plotlyjs=True)
    print(f"3D interactive plot saved to {save_path}")
    return save_path
