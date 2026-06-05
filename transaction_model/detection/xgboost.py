"""XGBoost 欺诈检测：训练、评估、三模型对比"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA

from transaction_model.config import load_config, resolve_path
from transaction_model.data.feature import encode_categorical, engineer_features
from transaction_model.data.loader import load_parquet
from transaction_model.detection.metrics import compute_metrics, print_results_summary


def get_device() -> str:
    """获取 XGBoost 设备"""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def train_xgb_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    params: dict,
    name: str = "Model",
    early_stopping_rounds: int | None = None,
    eval_metric: str = "auc",
) -> tuple:
    """训练单个 XGBoost 模型

    Args:
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        X_test, y_test: 测试数据和标签
        params: XGBoost 超参数
        name: 模型名称
        early_stopping_rounds: 早停轮数 (None 表示不启用)
        eval_metric: 早停监控指标

    Returns:
        (model, metrics_dict) 元组
    """
    device = get_device()
    print(f"\nTraining {name}...")
    print(f"  Features: {X_train.shape[1]}d | Samples: {X_train.shape[0]:,}")
    if early_stopping_rounds is None:
        print("  Early stopping: disabled")

    t0 = time.time()
    clf = xgb.XGBClassifier(
        **params,
        device=device,
    )
    fit_kwargs: dict = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if early_stopping_rounds is not None:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        fit_kwargs["eval_metric"] = eval_metric
    clf.fit(X_train, y_train, **fit_kwargs)
    train_time = time.time() - t0

    val_preds = clf.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, val_preds)

    test_preds = clf.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_preds)

    best_iter = clf.best_iteration if clf.best_iteration is not None else "n/a"
    print(f"  Train time: {train_time:.1f}s (best_iteration={best_iter})")
    print(f"  Val  ROC-AUC: {val_metrics['auc']:.4f} | AP: {val_metrics['ap']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics['auc']:.4f} | AP: {test_metrics['ap']:.4f}")

    return clf, {
        'val_auc': val_metrics['auc'], 'val_ap': val_metrics['ap'],
        'test_auc': test_metrics['auc'], 'test_ap': test_metrics['ap'],
    }


def load_embeddings_and_labels(
    embed_dir: Path,
) -> tuple:
    """加载所有分割的嵌入和标签

    Returns:
        (train_emb, val_emb, test_emb, y_train, y_val, y_test,
         train_row_ids, val_row_ids, test_row_ids)
    """
    print("Loading embeddings...")
    train_emb = np.load(embed_dir / "train_embeddings.npy")
    y_train = np.load(embed_dir / "train_labels.npy")
    train_row_ids = np.load(embed_dir / "train_row_ids.npy")

    val_emb = np.load(embed_dir / "val_embeddings.npy")
    y_val = np.load(embed_dir / "val_labels.npy")
    val_row_ids = np.load(embed_dir / "val_row_ids.npy")

    test_emb = np.load(embed_dir / "test_embeddings.npy")
    y_test = np.load(embed_dir / "test_labels.npy")
    test_row_ids = np.load(embed_dir / "test_row_ids.npy")

    print(f"  Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}")
    return (train_emb, val_emb, test_emb,
            y_train, y_val, y_test,
            train_row_ids, val_row_ids, test_row_ids)


def apply_pca(
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    test_emb: np.ndarray,
    n_components: int = 64,
    random_state: int = 42,
) -> tuple:
    """PCA 降维

    Returns:
        (train_pca, val_pca, test_pca, pca_model)
    """
    print(f"PCA: {train_emb.shape[1]}d -> {n_components}d")
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=random_state)
    train_pca = pca.fit_transform(train_emb)
    val_pca = pca.transform(val_emb)
    test_pca = pca.transform(test_emb)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  PCA fit+transform time: {time.time()-t0:.1f}s")
    return train_pca, val_pca, test_pca, pca


def load_and_align_raw_features(
    y_train: np.ndarray,
    train_row_ids: np.ndarray,
    val_row_ids: np.ndarray,
    test_row_ids: np.ndarray,
    config_name: str = "dataset",
) -> tuple:
    """加载原始特征并对齐到嵌入分割

    Returns:
        (X_train_raw, X_val_raw, X_test_raw) pandas DataFrame
    """
    ds_cfg = load_config(config_name)
    feature_cols = ds_cfg["feature_cols"]
    temporal_dir = resolve_path(ds_cfg["dataset"]["temporal_split_dir"])

    print("Loading temporal split parquets...")
    train_gdf = load_parquet(temporal_dir / "train.parquet")
    val_gdf = load_parquet(temporal_dir / "val_eval.parquet")
    test_gdf = load_parquet(temporal_dir / "test_eval.parquet")

    # 特征工程
    for gdf in [train_gdf, val_gdf, test_gdf]:
        engineer_features(gdf)

    # 转换为 pandas
    if hasattr(train_gdf, 'to_pandas'):
        train_pdf = train_gdf.to_pandas()
    else:
        train_pdf = train_gdf
    if hasattr(val_gdf, 'to_pandas'):
        val_pdf = val_gdf.to_pandas()
    else:
        val_pdf = val_gdf
    if hasattr(test_gdf, 'to_pandas'):
        test_pdf = test_gdf.to_pandas()
    else:
        test_pdf = test_gdf

    # 重建平衡训练样本
    fraud_col = "Is Fraud?"
    BALANCED_TOTAL = len(y_train)
    fraud_mask = (train_pdf[fraud_col] == "Yes") | (train_pdf[fraud_col] == "1")
    fraud_idx = train_pdf.index[fraud_mask].tolist()
    normal_idx = train_pdf.index[~fraud_mask].tolist()

    np.random.seed(42)
    n_fraud = min(len(fraud_idx), int(BALANCED_TOTAL * 0.1))
    n_normal = min(len(normal_idx), BALANCED_TOTAL - n_fraud)
    balanced_idx = np.concatenate([
        np.random.choice(fraud_idx, n_fraud, replace=False),
        np.random.choice(normal_idx, n_normal, replace=False),
    ])
    np.random.shuffle(balanced_idx)

    X_train_raw = train_pdf.loc[balanced_idx, feature_cols].reset_index(drop=True).iloc[train_row_ids].reset_index(drop=True)
    X_val_raw = val_pdf.iloc[val_row_ids][feature_cols].reset_index(drop=True)
    X_test_raw = test_pdf.iloc[test_row_ids][feature_cols].reset_index(drop=True)

    return X_train_raw, X_val_raw, X_test_raw


def run_three_model_comparison(
    config_name: str = "xgboost",
) -> dict:
    """执行三模型 XGBoost 对比实验

    模型：
    1. Baseline: 13d 原始特征
    2. Embeddings: 64d PCA 嵌入
    3. Combined: 13d 原始 + 64d PCA 嵌入

    Returns:
        {'baseline': ..., 'embed': ..., 'combined': ..., 'clfs': {...}}
    """
    cfg = load_config(config_name)

    embed_dir = resolve_path(cfg["inference"]["embed_dir"])
    outputs_dir = resolve_path(cfg["inference"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载嵌入
    (train_emb, val_emb, test_emb,
     y_train, y_val, y_test,
     train_row_ids, val_row_ids, test_row_ids) = load_embeddings_and_labels(embed_dir)

    # 2. PCA 降维
    pca_dim = cfg["pca"]["n_components"]
    train_pca, val_pca, test_pca, pca_model = apply_pca(
        train_emb, val_emb, test_emb,
        n_components=pca_dim,
        random_state=cfg["pca"]["random_state"],
    )

    # 3. 加载原始特征
    X_train_raw, X_val_raw, X_test_raw = load_and_align_raw_features(
        y_train, train_row_ids, val_row_ids, test_row_ids,
    )

    # 4. 编码类别特征
    X_train_enc, X_val_enc, X_test_enc, preprocessor = encode_categorical(
        X_train_raw, X_val_raw, X_test_raw,
    )

    n_raw = X_train_enc.shape[1]

    train_cfg = cfg.get("xgboost", {}).get("train", {})
    early_stopping = train_cfg.get("early_stopping_rounds")
    eval_metric = train_cfg.get("eval_metric", "auc")

    # 5. 训练三个模型
    print("\n" + "=" * 60)
    print("Training XGBoost Models (HPO-optimized params)")
    print("=" * 60)

    # [1/3] Baseline
    clf_baseline, metrics_baseline = train_xgb_model(
        X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test,
        params=cfg["xgboost"]["params_raw"],
        name=f"Raw Features ({n_raw}d)",
        early_stopping_rounds=early_stopping,
        eval_metric=eval_metric,
    )

    # [2/3] Embeddings
    clf_embed, metrics_embed = train_xgb_model(
        train_pca, y_train, val_pca, y_val, test_pca, y_test,
        params=cfg["xgboost"]["params_embed"],
        name=f"Embeddings ({pca_dim}d PCA)",
        early_stopping_rounds=early_stopping,
        eval_metric=eval_metric,
    )

    # [3/3] Combined
    X_train_combined = np.hstack([X_train_enc, train_pca])
    X_val_combined = np.hstack([X_val_enc, val_pca])
    X_test_combined = np.hstack([X_test_enc, test_pca])
    clf_combined, metrics_combined = train_xgb_model(
        X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test,
        params=cfg["xgboost"]["params_combined"],
        name=f"Combined ({X_train_combined.shape[1]}d)",
        early_stopping_rounds=early_stopping,
        eval_metric=eval_metric,
    )

    # 6. 结果汇总
    labels = [f'Raw Features (baseline)', f'{pca_dim}d PCA Embeddings',
              f'Combined ({n_raw}+{pca_dim}d)']
    feat_dims = [f'{n_raw}d', f'{pca_dim}d', f'{X_train_combined.shape[1]}d']
    all_results = [metrics_baseline, metrics_embed, metrics_combined]

    print_results_summary(all_results, labels, feat_dims)

    return {
        "baseline": metrics_baseline,
        "embed": metrics_embed,
        "combined": metrics_combined,
        "clfs": {
            "baseline": clf_baseline,
            "embed": clf_embed,
            "combined": clf_combined,
        },
        "pca_model": pca_model,
        "pca_dim": pca_dim,
        "n_raw": n_raw,
    }
