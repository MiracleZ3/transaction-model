"""Step 05: XGBoost 欺诈检测对比"""
from __future__ import annotations

import argparse

from transaction_model.config import resolve_path
from transaction_model.detection.xgboost import run_three_model_comparison
from transaction_model.visualization.results_viz import plot_model_comparison


def main():
    parser = argparse.ArgumentParser(description="Step 05: Fraud Detection")
    parser.add_argument("--no-plot", action="store_true", help="跳过可视化")
    args = parser.parse_args()

    results = run_three_model_comparison()

    if not args.no_plot:
        pca_dim = results["pca_dim"]
        n_raw = results["n_raw"]
        labels = [f'Raw Features\n({n_raw}d)', f'Embeddings\n(PCA {pca_dim}d)',
                  f'Combined\n({n_raw}+{pca_dim}d)']
        dims = [f'{n_raw}d', f'{pca_dim}d', f'{n_raw+pca_dim}d']
        plot_model_comparison(
            [results["baseline"], results["embed"], results["combined"]],
            labels, dims,
            save_path=str(resolve_path("data/outputs/xgb_auc_ap_comparison.png")),
        )


if __name__ == "__main__":
    main()
