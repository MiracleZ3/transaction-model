"""Step 02b: 从银联 NDJSON 生成 tokenized 语料库

路线 A 的 Step 2：调用 ``corpus/generate_ndjson.generate_all_yl_corpora``，
把展开后的行级 parquet（含 cups_* 列）经 YLPipeline 处理成与 decoder CLM
同结构的 ``<bos> ... <sep> ... <eos>`` 文本语料，并保存 YLTabularTokenizer state。

用法:
    python scripts/step_02_tokenize_ndjson.py [--config dataset_yl] [--force]
"""
from __future__ import annotations

import argparse

from transaction_model.corpus.generate_ndjson import generate_all_yl_corpora


def main():
    parser = argparse.ArgumentParser(description="Step 02b: Tokenize YL NDJSON corpus")
    parser.add_argument(
        "--config", default="dataset_yl",
        help="数据集配置名（不含 .yaml）",
    )
    parser.add_argument(
        "--force", action="store_true", help="强制重新生成（含重新 fit tokenizer）"
    )
    args = parser.parse_args()

    generate_all_yl_corpora(config_name=args.config, force=args.force)
    print("\nYL corpus generation complete!")


if __name__ == "__main__":
    main()
