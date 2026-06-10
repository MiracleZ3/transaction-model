"""银联 NDJSON → tokenized corpus（路线 A 核心入口）。

把「用户级聚合」的 NDJSON 转成与现有 ``corpus/generate.py`` 同结构的
``<bos> txn1 <sep> txn2 <sep> ... <eos>`` 文本语料，让现有 ``training/clm_data.py``
的 ``build_financial_clm_dataset`` 可以无差别消费。

流程：
  1. NDJSON → 行级 DataFrame（``data/ndjson_loader``）
  2. YLPipeline.preprocess + fit + transform（``tokenizer/yl_pipeline``）
  3. 按 cert_sm3 分组，chunk_size 切窗，拼成 corpus 行（``Pipeline.to_corpus_lines``）
  4. 同时保存 sidecar 元数据（每行 = 哪个 cert_sm3 / label / chunk_id）

输出 corpus 文件**与 TabFormer 路线完全同格式**，可直接喂进 NeMo recipe。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Union

from transaction_model.config import load_config, resolve_path
from transaction_model.constants import YL_LABEL_KEY, YL_USER_KEY
from transaction_model.data.ndjson_loader import (
    load_ndjson,
    temporal_split_ndjson,
)

PathLike = Union[str, Path]


def generate_corpus_from_ndjson(
    split_name: str,
    ndjson_path_or_dir: PathLike,
    corpus_path: PathLike,
    merchant_hash_size: int = 4000,
    merch_name_hash_size: Optional[int] = None,
    amount_strategy: str = "quantile",
    amount_bins: int = 20,
    amount_thresholds: Optional[list] = None,
    include_time_delta: bool = True,
    time_delta_bins: int = 32,
    chunk_size: int = 315,
    tokenizer_state_path: Optional[PathLike] = None,
    temporal_split_first: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    force: bool = False,
    save_sidecar: bool = True,
) -> tuple[list[str], "YLTabularTokenizer"]:
    """把一个 NDJSON 源转成 corpus 文本 + YL tokenizer state。

    Args:
        split_name: 分割名（"train" / "val" / "test"）— 仅用于日志
        ndjson_path_or_dir: NDJSON 单文件或目录
        corpus_path: 输出 corpus 文本路径
        merchant_hash_size / merch_name_hash_size / amount_strategy / ... :
            YLPipeline 参数
        chunk_size: 每序列最大交易数（与 TabFormer 一致 ~315 fits 4096 tokens）
        tokenizer_state_path: 保存 YLTabularTokenizer vocab 的 JSON 路径
            （None 则不落盘）。第一次 fit 后保存；transform 时可复用。
        temporal_split_first: 是否先按 unix_timestap 时间切分再 fit/transform。
            True 时本 split 只取切分后对应的部分 fit + transform。
        train_ratio / val_ratio: 时间切分比例
        force: 是否覆盖已有 corpus
        save_sidecar: 是否同时输出 ``<corpus>.meta.jsonl``（行级 cert_sm3+label）

    Returns:
        (corpus_lines, yl_tokenizer)
    """
    from transaction_model.tokenizer import YLTabularTokenizer

    corpus_path = Path(corpus_path)
    ndjson_path_or_dir = Path(ndjson_path_or_dir)

    if corpus_path.exists() and not force and tokenizer_state_path and Path(tokenizer_state_path).exists():
        with open(corpus_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        tok = YLTabularTokenizer.from_file(tokenizer_state_path)
        print(f"[{split_name}] Corpus already exists: {len(lines):,} sequences")
        return lines, tok

    print(f"\n{'='*60}")
    print(f"Generating YL corpus for {split_name}")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. 加载 NDJSON → 行级 DataFrame
    df = load_ndjson(ndjson_path_or_dir, use_gpu=True, drop_time_cols=False)
    print(f"  Loaded {len(df):,} txn rows in {time.time()-t0:.1f}s")

    # 2. 时间切分（按 unix_timestap 分位数）
    if temporal_split_first:
        if split_name == "train":
            df, _, _ = temporal_split_ndjson(
                df, train_ratio=train_ratio, val_ratio=val_ratio
            )
        elif split_name == "val":
            _, df, _ = temporal_split_ndjson(
                df, train_ratio=train_ratio, val_ratio=val_ratio
            )
        elif split_name == "test":
            _, _, df = temporal_split_ndjson(
                df, train_ratio=train_ratio, val_ratio=val_ratio
            )
        # 其它自定义 split_name 保留全量

    # 3. tokenizer：尝试加载已 fit 的 state，否则 fit 并保存
    if tokenizer_state_path and Path(tokenizer_state_path).exists() and not force:
        tok = YLTabularTokenizer.from_file(tokenizer_state_path)
        print(f"  Loaded YL tokenizer from {tokenizer_state_path} "
              f"(vocab_size={tok.get_vocab_size():,})")
        pipeline = tok._pipeline
        gdf_proc = pipeline.preprocess(df)
    else:
        tok = YLTabularTokenizer(
            merchant_hash_size=merchant_hash_size,
            merch_name_hash_size=merch_name_hash_size,
            amount_strategy=amount_strategy,
            amount_bins=amount_bins,
            amount_thresholds=amount_thresholds,
            include_time_delta=include_time_delta,
            time_delta_bins=time_delta_bins,
        )
        pipeline = tok._pipeline
        gdf_proc = pipeline.preprocess(df)
        tok.fit(gdf_proc)
        print(f"  Fitted YL tokenizer: vocab_size={tok.get_vocab_size():,}")
        if tokenizer_state_path:
            tok.save(tokenizer_state_path)
            print(f"  Saved tokenizer state → {tokenizer_state_path}")

    # 4. transform → token DataFrame
    token_df = pipeline.transform(gdf_proc)

    # 5. 按 cert_sm3 分组，chunk 切窗
    group_cols = [YL_USER_KEY]
    print(f"  Grouping by: {group_cols}; chunk_size={chunk_size}")

    corpus_lines, sidecar = _to_corpus_lines_with_meta(
        token_df, gdf_proc, group_cols, chunk_size=chunk_size
    )

    # 6. 写出 corpus
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    if save_sidecar:
        sidecar_path = corpus_path.with_suffix(corpus_path.suffix + ".meta.jsonl")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            for row in sidecar:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Sidecar (cert_sm3+label per line) → {sidecar_path}")

    elapsed = time.time() - t0
    print(f"  Generated {len(corpus_lines):,} sequences in {elapsed:.1f}s")
    print(f"  Saved to: {corpus_path}")

    if corpus_lines:
        sample = corpus_lines[0]
        n_tokens = len(sample.split())
        n_seps = sample.count("<sep>")
        print(f"  Sample: {n_tokens} tokens, {n_seps+1} transactions")
        print(f"  Preview: {sample[:200]}...")

    return corpus_lines, tok


def _to_corpus_lines_with_meta(
    token_df,
    df_meta,
    group_cols: list[str],
    chunk_size: int = 315,
) -> tuple[list[str], list[dict]]:
    """与 ``Pipeline.to_corpus_lines`` 等价，但额外返回行级 sidecar。

    sidecar 每行 = 一条 corpus 序列的元数据::

        {"line_idx": 0, "cert_sm3": "u1", "chunk_id": 0,
         "label": 0 | null, "n_txns": 315}

    label 取该分组内 label 的众数（用户级标签广播到行后应一致）。
    """
    token_cols = list(token_df.columns)
    txn_text = token_df[token_cols[0]].str.cat(
        [token_df[c] for c in token_cols[1:]], sep=" "
    )

    work = df_meta[group_cols].copy()
    work["_txn_text"] = txn_text
    if YL_LABEL_KEY in df_meta.columns:
        work[YL_LABEL_KEY] = df_meta[YL_LABEL_KEY].values

    work["_seq_id"] = work.groupby(group_cols).cumcount()
    work["_chunk_id"] = (work["_seq_id"] // chunk_size).astype("int32")

    group_keys = group_cols + ["_chunk_id"]
    grouped = work.groupby(group_keys)["_txn_text"].agg(list)
    if hasattr(grouped, "to_pandas"):
        grouped = grouped.to_pandas()
    else:
        grouped = grouped

    # 同时取每组的 label（用户级）
    label_series = None
    if YL_LABEL_KEY in work.columns:
        # group_keys[:-1] = group_cols（去掉 _chunk_id）
        label_grp = (
            work.groupby(group_cols)[YL_LABEL_KEY]
            .agg(lambda s: s.iloc[0] if hasattr(s, "iloc") else list(s)[0])
        )
        if hasattr(label_grp, "to_pandas"):
            label_grp = label_grp.to_pandas()
        label_series = label_grp

    lines: list[str] = []
    sidecar: list[dict] = []
    for line_idx, ((group_vals), txn_list) in enumerate(grouped.items()):
        # group_vals 可能是 tuple（多 group col）或 scalar（单 group col）
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        cert = group_vals[0]
        chunk_id = group_vals[-1] if len(group_vals) > 1 else 0

        line = "<bos> " + " <sep> ".join(txn_list) + " <eos>"
        lines.append(line)

        meta = {
            "line_idx": line_idx,
            YL_USER_KEY: str(cert),
            "chunk_id": int(chunk_id),
            "n_txns": len(txn_list),
            "label": (
                int(label_series.loc[cert])
                if label_series is not None and cert in label_series.index
                else None
            ),
        }
        sidecar.append(meta)
    return lines, sidecar


def generate_all_yl_corpora(
    config_name: str = "dataset_yl",
    force: bool = False,
) -> dict:
    """从 ``configs/dataset_yl.yaml`` 驱动，生成 train/val/test 三份语料。

    Returns:
        {split_name: corpus_lines}
    """
    cfg = load_config(config_name)
    ds_cfg = cfg["dataset"]
    tok_cfg = cfg["tokenizer"]
    corpus_cfg = cfg["corpus"]

    ndjson_dir = resolve_path(ds_cfg["ndjson_dir"])
    tokenizer_state = resolve_path(tok_cfg.get("state_path", "data/yl/yl_tokenizer.json"))

    results = {}
    splits = [
        ("train", corpus_cfg["train"]),
        ("val", corpus_cfg["val"]),
        ("test", corpus_cfg["test"]),
    ]

    # train 先 fit + 保存 tokenizer；val/test 复用 train 的 tokenizer state
    first_split = True
    for split_name, corpus_path in splits:
        lines, _ = generate_corpus_from_ndjson(
            split_name=split_name,
            ndjson_path_or_dir=ndjson_dir,
            corpus_path=resolve_path(corpus_path),
            merchant_hash_size=tok_cfg["merchant_hash_size"],
            merch_name_hash_size=tok_cfg.get("merch_name_hash_size"),
            amount_strategy=tok_cfg.get("amount_strategy", "quantile"),
            amount_bins=tok_cfg.get("amount_bins", 20),
            amount_thresholds=tok_cfg.get("amount_thresholds"),
            include_time_delta=tok_cfg.get("include_time_delta", True),
            time_delta_bins=tok_cfg.get("time_delta_bins", 32),
            chunk_size=tok_cfg.get("chunk_size", 315),
            tokenizer_state_path=tokenizer_state,
            temporal_split_first=True,
            train_ratio=cfg["split"]["train_ratio"],
            val_ratio=cfg["split"]["val_ratio"],
            force=force or first_split,  # 第一次必须 fit
            save_sidecar=True,
        )
        results[split_name] = lines
        first_split = False

    # 总结
    print("\nYL Corpus Summary:")
    print("=" * 60)
    for name in ["train", "val", "test"]:
        path = resolve_path(corpus_cfg[name])
        if path.exists():
            n_lines = len(results[name])
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name:6s}: {n_lines:>8,} sequences  "
                  f"({size_mb:>7.1f} MB)  {path.name}")

    return results
