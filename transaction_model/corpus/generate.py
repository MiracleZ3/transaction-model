"""语料库生成：将时间分割数据转换为分词后的文本序列"""
from __future__ import annotations

import os
import time
from pathlib import Path

from transaction_model.config import load_config, resolve_path
from transaction_model.data.loader import load_parquet


def generate_corpus(
    split_name: str,
    parquet_path: str | Path,
    corpus_path: str | Path,
    merchant_hash_size: int = 2000,
    chunk_size: int = 315,
    force: bool = False,
) -> list[str]:
    """将一个数据分割转换为分词语料库

    流程：
    1. 加载 parquet
    2. preprocess → fit → transform (FinancialTokenizerPipeline)
    3. 按用户/卡片分组，chunk 为 ~315 交易的序列
    4. 保存为文本文件

    Args:
        split_name: 分割名 ("train", "val", "test")
        parquet_path: 输入 parquet 路径
        corpus_path: 输出语料文件路径
        merchant_hash_size: 商户哈希空间大小
        chunk_size: 每序列交易数 (~4096 tokens)
        force: 是否强制重新生成

    Returns:
        语料行列表
    """
    from transaction_model.tokenizer import FinancialTokenizerPipeline

    corpus_path = Path(corpus_path)
    parquet_path = Path(parquet_path)

    if corpus_path.exists() and not force:
        with open(corpus_path) as f:
            n_lines = sum(1 for _ in f)
        print(f"[{split_name}] Corpus already exists: {n_lines:,} sequences")
        with open(corpus_path) as f:
            return [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Generating decoder corpus for {split_name}")
    print(f"{'='*60}")

    t0 = time.time()
    gdf = load_parquet(parquet_path)
    print(f"  Loaded {len(gdf):,} rows in {time.time()-t0:.1f}s")

    pip = FinancialTokenizerPipeline(merchant_hash_size=merchant_hash_size)
    gdf_proc = pip.preprocess(gdf)
    pip.fit(gdf_proc)
    token_df = pip.transform(gdf_proc)

    # 识别分组列
    group_cols = []
    for col_name in ["user", "User", "cust"]:
        if col_name in gdf_proc.columns:
            group_cols.append(col_name)
            break
    for col_name in ["card", "Card", "card_id"]:
        if col_name in gdf_proc.columns:
            group_cols.append(col_name)
            break
    if not group_cols:
        group_cols = [gdf_proc.columns[0]]

    print(f"  Grouping by: {group_cols}")
    n_tokens_per_txn = 12
    print(f"  Chunk size: {chunk_size} transactions "
          f"(~{chunk_size * n_tokens_per_txn} txn tokens + {chunk_size - 1} <sep>'s)")

    corpus_lines = pip.to_corpus_lines(
        token_df, gdf_proc, group_cols, chunk_size=chunk_size
    )

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    elapsed = time.time() - t0
    print(f"  Generated {len(corpus_lines):,} sequences in {elapsed:.1f}s")
    print(f"  Saved to: {corpus_path}")

    sample = corpus_lines[0]
    n_tokens = len(sample.split())
    n_seps = sample.count("<sep>")
    print(f"  Sample: {n_tokens} tokens, {n_seps+1} transactions")
    print(f"  Preview: {sample[:200]}...")

    return corpus_lines


def generate_all_corpora(
    config_name: str = "tokenizer",
    force: bool = False,
) -> dict[str, list[str]]:
    """生成所有分割的语料库

    Args:
        config_name: 配置文件名（不含 .yaml）
        force: 是否强制重新生成

    Returns:
        {split_name: corpus_lines} 字典
    """
    cfg = load_config(config_name)
    ds_cfg = load_config("dataset")

    corpus_cfg = cfg["corpus"]
    tok_cfg = cfg["tokenizer"]

    splits = [
        ("train", ds_cfg["dataset"]["temporal_split_dir"] + "/train.parquet", corpus_cfg["train"]),
        ("val",   ds_cfg["dataset"]["temporal_split_dir"] + "/val.parquet",   corpus_cfg["val"]),
        ("test",  ds_cfg["dataset"]["temporal_split_dir"] + "/test.parquet",  corpus_cfg["test"]),
    ]

    results = {}
    for split_name, parquet_path, corpus_path in splits:
        results[split_name] = generate_corpus(
            split_name=split_name,
            parquet_path=resolve_path(parquet_path),
            corpus_path=resolve_path(corpus_path),
            merchant_hash_size=tok_cfg["merchant_hash_size"],
            chunk_size=tok_cfg["chunk_size"],
            force=force,
        )

    # 打印总结
    print("\nCorpus Summary:")
    print("=" * 60)
    for name in ["train", "val", "test"]:
        path = resolve_path(corpus_cfg[name])
        if path.exists():
            n_lines = len(results[name])
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name:6s}: {n_lines:>8,} sequences  ({size_mb:>7.1f} MB)  {path.name}")

    return results
