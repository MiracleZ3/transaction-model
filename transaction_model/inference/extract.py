"""嵌入提取 pipeline：从预训练模型中提取交易嵌入向量"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from transaction_model.config import load_config, resolve_path
from transaction_model.data.loader import load_parquet
from transaction_model.data.sampling import balanced_subsample_by_index
from transaction_model.inference.decoder_inference import HuggingFaceDecoderInference


def extract_split_embeddings(
    split: str,
    parquet_path: Path,
    model_dir: Path,
    embed_dir: Path,
    merchant_hash_size: int = 2000,
    batch_size: int = 1024,
    max_length: int = 128,
    balanced_train_size: int = 1_000_000,
    force: bool = False,
    tokenizer_variant: str = "tabformer",
    tokenizer_state_path: Path | None = None,
    label_col: str = "Is Fraud?",
    group_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取单个数据分割的嵌入向量

    流程：
    1. 加载 parquet
    2. 提取标签（在 preprocess 重命名列之前）
    3. 平衡采样（仅 train split）
    4. 保留 __row_id__ 列
    5. preprocess → fit → transform → encode
    6. 批量推理提取嵌入

    Args:
        split: 分割名 ("train", "val", "test")
        parquet_path: 输入 parquet 路径
        model_dir: 预训练模型目录
        embed_dir: 嵌入输出目录
        merchant_hash_size: 商户哈希空间
        batch_size: 推理批次大小
        max_length: 最大序列长度
        balanced_train_size: 训练集平衡采样总数
        force: 是否强制重新提取
        tokenizer_variant: {"tabformer", "yl"}；yl 用已 fit 的 tokenizer state
        tokenizer_state_path: variant="yl" 时必填，``YLTabularTokenizer.save`` 输出
        label_col: 标签列名（tabformer 默认 "Is Fraud?"；yl 默认 "label"）
        group_col: 分组/用户列名（yl 用 "cert_sm3"）。None 时按行展开。

    Returns:
        (embeddings, labels, row_ids) 元组
    """
    if tokenizer_variant == "yl":
        return _extract_split_embeddings_yl(
            split=split,
            parquet_path=parquet_path,
            model_dir=model_dir,
            embed_dir=embed_dir,
            tokenizer_state_path=tokenizer_state_path,
            batch_size=batch_size,
            max_length=max_length,
            balanced_train_size=balanced_train_size,
            force=force,
            label_col=label_col,
            group_col=group_col or "cert_sm3",
        )

    # ---- TabFormer 路径（原逻辑） ----
    from transaction_model.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer

    embed_path = embed_dir / f"{split}_embeddings.npy"
    label_path = embed_dir / f"{split}_labels.npy"
    row_id_path = embed_dir / f"{split}_row_ids.npy"

    # 断点续跑：已存在则直接加载
    if embed_path.exists() and label_path.exists() and not force:
        emb = np.load(embed_path)
        labels = np.load(label_path)
        row_ids = np.load(row_id_path) if row_id_path.exists() else np.arange(len(emb), dtype=np.int64)
        print(f"[{split}] Already extracted: {emb.shape}, {labels.sum():,} fraud / {len(labels):,}")
        return emb, labels, row_ids

    print(f"\n{'='*60}")
    print(f"Extracting {split} embeddings")
    print(f"{'='*60}")

    # 1. 加载数据
    t0 = time.time()
    gdf = load_parquet(parquet_path)

    # 2. 提取标签
    labels = _extract_labels(gdf)

    # 3. 平衡采样（仅 train）
    if split == "train" and labels is not None:
        sampled = balanced_subsample_by_index(
            labels, total_samples=balanced_train_size, fraud_ratio=0.1, random_state=42
        )
        gdf = gdf.iloc[sampled].reset_index(drop=True)
        labels = labels[sampled]
        print(f"  Balanced sample: {len(gdf):,} rows, {labels.sum():,} fraud ({labels.mean():.1%})")

    # 4. 保留 row_id
    gdf["__row_id__"] = np.arange(len(gdf), dtype=np.int64)

    # 5. 分词管道
    pip = FinancialTokenizerPipeline(merchant_hash_size=merchant_hash_size)
    gdf = pip.preprocess(gdf)
    # 提取 __row_id__（preprocess 可能重排列）
    if hasattr(gdf["__row_id__"], "to_pandas"):
        row_ids = gdf["__row_id__"].to_pandas().to_numpy(dtype=np.int64)
    else:
        row_ids = gdf["__row_id__"].to_numpy(dtype=np.int64)
    if labels is not None:
        labels = labels[row_ids]
    pip.fit(gdf)
    token_df = pip.transform(gdf)
    padded_ids = pip.encode(token_df, max_length=max_length)
    tok_time = time.time() - t0
    print(f"  Tokenized {len(padded_ids):,} rows in {tok_time:.1f}s")

    # 6. 初始化推理器
    tokenizer = FinancialTabularTokenizer(
        merchant_hash_size=merchant_hash_size,
        category_hierarchy=True,
        temporal_encoding=True,
    )
    inference = HuggingFaceDecoderInference(
        model_path=model_dir,
        tokenizer=tokenizer,
        pooling="last_token",
    )
    print(f"  Model loaded on {inference.device} (embed_dim={inference.embedding_dim})")

    # 7. 批量提取
    print(f"  Extracting embeddings (batch_size={batch_size})...")
    t0 = time.time()
    emb = inference.extract_embeddings_batched(
        padded_ids, batch_size=batch_size, show_progress=True
    )
    inf_time = time.time() - t0
    print(f"  Extracted {emb.shape} in {inf_time:.1f}s ({len(emb)/inf_time:,.0f} samples/sec)")

    # 8. 保存
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_path, emb)
    if labels is not None:
        np.save(label_path, labels)
    np.save(row_id_path, row_ids)
    print(f"  Saved to {embed_path}")

    return emb, labels, row_ids


def _extract_split_embeddings_yl(
    split: str,
    parquet_path: Path,
    model_dir: Path,
    embed_dir: Path,
    tokenizer_state_path: Path | None,
    batch_size: int = 1024,
    max_length: int = 128,
    balanced_train_size: int = 1_000_000,
    force: bool = False,
    label_col: str = "label",
    group_col: str = "cert_sm3",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YL 路线嵌入提取：用已 fit 的 tokenizer state，不重新 fit。

    YL 数据 (NDJSON) 在 corpus 生成阶段已经 fit 过 tokenizer 并保存 state。
    这里只做：
      1. 加载展开后的 parquet（含 cups_* / cert_sm3 / label / unix_timestap）
      2. 用 YLTabularTokenizer.from_file 加载 pipeline → preprocess → transform → encode
      3. 平衡采样（仅 train）
      4. 批量推理
    """
    from transaction_model.tokenizer import YLTabularTokenizer

    if tokenizer_state_path is None:
        raise ValueError(
            "YL embedding extraction requires tokenizer_state_path "
            "(YLPipeline fit 后 save 出的 JSON)"
        )

    embed_path = embed_dir / f"{split}_embeddings.npy"
    label_path = embed_dir / f"{split}_labels.npy"
    row_id_path = embed_dir / f"{split}_row_ids.npy"

    if embed_path.exists() and label_path.exists() and not force:
        emb = np.load(embed_path)
        labels = np.load(label_path)
        row_ids = (
            np.load(row_id_path)
            if row_id_path.exists()
            else np.arange(len(emb), dtype=np.int64)
        )
        print(f"[{split}] Already extracted: {emb.shape}")
        return emb, labels, row_ids

    print(f"\n{'='*60}")
    print(f"Extracting {split} embeddings (YL variant)")
    print(f"{'='*60}")

    # 1. 加载
    t0 = time.time()
    gdf = load_parquet(parquet_path)

    # 2. 标签（YL：用户级 label 广播到每行）
    labels = _extract_labels(gdf, prefer_col=label_col)

    # 3. 平衡采样（仅 train）
    if split == "train" and labels is not None:
        sampled = balanced_subsample_by_index(
            labels, total_samples=balanced_train_size, fraud_ratio=0.1, random_state=42
        )
        gdf = gdf.iloc[sampled].reset_index(drop=True)
        labels = labels[sampled]
        print(f"  Balanced sample: {len(gdf):,} rows, {labels.sum():,} pos ({labels.mean():.1%})")

    # 4. row_id
    gdf["__row_id__"] = np.arange(len(gdf), dtype=np.int64)

    # 5. tokenize 用已 fit 的 YL pipeline
    tok = YLTabularTokenizer.from_file(tokenizer_state_path)
    pipeline = tok._pipeline
    gdf_proc = pipeline.preprocess(gdf)

    if hasattr(gdf_proc["__row_id__"], "to_pandas"):
        row_ids = gdf_proc["__row_id__"].to_pandas().to_numpy(dtype=np.int64)
    else:
        row_ids = gdf_proc["__row_id__"].to_numpy(dtype=np.int64)
    if labels is not None and len(labels) > len(row_ids):
        labels = labels[row_ids]

    token_df = pipeline.transform(gdf_proc)
    padded_ids = pipeline.encode(token_df, max_length=max_length)
    print(f"  Tokenized {len(padded_ids):,} rows in {time.time()-t0:.1f}s "
          f"(vocab_size={tok.get_vocab_size():,})")

    # 6. 推理
    inference = HuggingFaceDecoderInference(
        model_path=model_dir,
        tokenizer=tok,
        pooling="last_token",
    )
    print(f"  Model loaded on {inference.device} (embed_dim={inference.embedding_dim})")

    t0 = time.time()
    emb = inference.extract_embeddings_batched(
        padded_ids, batch_size=batch_size, show_progress=True
    )
    print(f"  Extracted {emb.shape} in {time.time()-t0:.1f}s")

    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_path, emb)
    if labels is not None:
        np.save(label_path, labels)
    np.save(row_id_path, row_ids)
    print(f"  Saved to {embed_path}")

    return emb, labels, row_ids


def _extract_labels(gdf, prefer_col: str | None = None) -> np.ndarray | None:
    """从 DataFrame 中提取欺诈标签。

    Args:
        gdf: DataFrame
        prefer_col: 优先匹配的列名（如 YL 的 "label"）。None 时按默认顺序推断。
    """
    if prefer_col is not None and prefer_col in gdf.columns:
        ordered = [prefer_col] + [
            c for c in ["Is Fraud?", "is_fraud", "Is_Fraud", "label", "fraud"]
            if c != prefer_col
        ]
    else:
        ordered = ["Is Fraud?", "is_fraud", "Is_Fraud", "label", "fraud"]

    for col in ordered:
        if col in gdf.columns:
            if hasattr(gdf[col], "to_pandas"):
                lbl = gdf[col].to_pandas()
            else:
                lbl = gdf[col]
            if lbl.dtype == object:
                labels = ((lbl == "Yes") | (lbl == "1")).astype(int).values
            else:
                labels = lbl.astype(int).values
            print(f"  Labels from '{col}': {labels.sum():,} fraud / {len(labels):,}")
            return labels
    return None


def extract_all_embeddings(
    config_name: str = "xgboost",
    training_config_name: str = "training",
    force: bool = False,
    dataset_config_name: str = "dataset",
) -> dict:
    """提取所有分割的嵌入向量

    Args:
        config_name: 包含推理参数的配置文件名
        training_config_name: 包含模型路径的配置文件名
        force: 是否强制重新提取
        dataset_config_name: 数据集配置名（"dataset" 走 TabFormer；
            "dataset_yl" 走银联 NDJSON 路径）

    Returns:
        包含所有嵌入、标签和元数据的字典
    """
    cfg = load_config(config_name)
    train_cfg = load_config(training_config_name)

    inf_cfg = cfg["inference"]
    ds_full = load_config(dataset_config_name)
    ds_cfg = ds_full["dataset"]
    sampling_cfg = ds_full.get("sampling", {
        "balanced_train_size": 1_000_000,
        "random_state": 42,
    })

    # 推断 variant：dataset_yl.yaml 含 source: "ndjson" 与 tokenizer.variant
    tokenizer_variant = "tabformer"
    tokenizer_state_path = None
    label_col = "Is Fraud?"
    group_col = None
    if ds_full.get("tokenizer", {}).get("variant") == "yl":
        tokenizer_variant = "yl"
        tokenizer_state_path = resolve_path(
            ds_full["tokenizer"].get("state_path", "data/yl/yl_tokenizer.json")
        )
        label_col = ds_cfg.get("label_col", "label")
        group_col = ds_cfg.get("cert_col", "cert_sm3")
        tok_cfg = ds_full["tokenizer"]
        merchant_hash_size = tok_cfg.get("merchant_hash_size", 2000)
    else:
        # TabFormer: tokenizer 配置在独立的 tokenizer.yaml
        tok_cfg = load_config("tokenizer")["tokenizer"]
        merchant_hash_size = tok_cfg["merchant_hash_size"]

    model_dir = resolve_path(train_cfg["paths"]["pretrained_model"])
    embed_dir = resolve_path(inf_cfg["embed_dir"])
    embed_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Decoder model directory not found: {model_dir}. "
            f"Place a HF compatible checkpoint there or update "
            f"configs/{training_config_name}.yaml paths.pretrained_model."
        )
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"HuggingFace config.json not found inside {model_dir}. "
            f"Expected a model directory containing config.json + "
            f"safetensors/pytorch_model weights."
        )

    if tokenizer_variant == "yl":
        # YL: 输入是展开后的 parquet（含 cups_* / cert_sm3 / label）
        split_to_parquet = {
            "train": resolve_path(ds_cfg["temporal_split_dir"]) / "train.parquet",
            "val": resolve_path(ds_cfg["temporal_split_dir"]) / "val.parquet",
            "test": resolve_path(ds_cfg["temporal_split_dir"]) / "test.parquet",
        }
    else:
        split_to_parquet = {
            "train": resolve_path(ds_cfg["temporal_split_dir"]) / "train.parquet",
            "val": resolve_path(ds_cfg["val_eval"]),
            "test": resolve_path(ds_cfg["test_eval"]),
        }

    all_embeddings = []
    all_labels = []
    split_sizes = {}

    for split in ("train", "val", "test"):
        emb, labels, row_ids = extract_split_embeddings(
            split=split,
            parquet_path=split_to_parquet[split],
            model_dir=model_dir,
            embed_dir=embed_dir,
            merchant_hash_size=merchant_hash_size,
            batch_size=inf_cfg["batch_size"],
            max_length=inf_cfg["max_length"],
            balanced_train_size=sampling_cfg.get("balanced_train_size", 1_000_000),
            force=force,
            tokenizer_variant=tokenizer_variant,
            tokenizer_state_path=tokenizer_state_path,
            label_col=label_col,
            group_col=group_col,
        )
        all_embeddings.append(emb)
        all_labels.append(labels)
        split_sizes[split] = len(emb)

    # 合并并保存
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    np.save(embed_dir / "embeddings.npy", embeddings)
    np.save(embed_dir / "labels.npy", labels)

    # 保存元数据
    metadata = {
        "backend": "huggingface_decoder",
        "pooling": inf_cfg["pooling"],
        "model_path": str(model_dir),
        "n_samples": len(embeddings),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": inf_cfg["batch_size"],
        "max_length": inf_cfg["max_length"],
        "splits": ["train", "val", "test"],
        "n_train": split_sizes.get("train", 0),
        "n_val": split_sizes.get("val", 0),
        "n_test": split_sizes.get("test", 0),
        "row_id_alignment": "explicit_split_row_ids",
    }
    with open(embed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll embeddings saved to {embed_dir}")
    print(f"  Total: {len(embeddings):,} x {embeddings.shape[1]}")
    for k, v in split_sizes.items():
        print(f"  {k.capitalize()}: {v:,}")

    return {
        "embeddings": embeddings,
        "labels": labels,
        "split_sizes": split_sizes,
        "metadata": metadata,
    }
