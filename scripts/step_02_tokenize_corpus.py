"""Step 02: 生成 tokenized 语料库"""
from __future__ import annotations

import argparse

from transaction_model.corpus.generate import generate_all_corpora


def main():
    parser = argparse.ArgumentParser(description="Step 02: Tokenize Corpus")
    parser.add_argument("--force", action="store_true", help="强制重新生成")
    args = parser.parse_args()

    generate_all_corpora(force=args.force)
    print("\nCorpus generation complete!")


if __name__ == "__main__":
    main()
