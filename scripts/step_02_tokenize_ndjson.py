"""Step 02b: 从银联 NDJSON 生成 tokenized 语料库

路线 A 的 Step 2：调用 ``corpus/generate_ndjson.generate_all_yl_corpora``，
把展开后的行级 parquet（含 cups_* 列）经 YLPipeline 处理成与 decoder CLM
同结构的 ``<bos> ... <sep> ... <eos>`` 文本语料，并保存 YLTabularTokenizer state。

完成后会自动读取已 fit 的 tokenizer 实际 vocab_size，同步回写
``configs/training_yl.yaml::model.config.vocab_size``，避免占位值（26000）
与真实词表不一致导致 IndexError 或参数浪费。

用法:
    python scripts/step_02_tokenize_ndjson.py [--config dataset_yl] [--force]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from transaction_model.corpus.generate_ndjson import generate_all_yl_corpora
from transaction_model.config import CONFIG_DIR, load_config
from transaction_model.tokenizer.yl_tokenizer import YLTabularTokenizer


def _sync_vocab_size(dataset_config_name: str) -> None:
    """读 dataset_yl.yaml 里的 tokenizer state，把真实 vocab_size 写回
    configs/training_yl.yaml 的 model.config.vocab_size。

    仅在二者不一致时改写，且只动那一行，保留 YAML 其余结构。
    """
    cfg = load_config(dataset_config_name)
    state_path = cfg["tokenizer"].get("state_path")
    if not state_path:
        return
    state_full = Path(state_path)
    if not state_full.is_absolute():
        state_full = (CONFIG_DIR.parent / state_full).resolve()
    if not state_full.exists():
        return

    real_vocab = YLTabularTokenizer.from_file(str(state_full)).get_vocab_size()

    train_cfg_path = CONFIG_DIR / "training_yl.yaml"
    if not train_cfg_path.exists():
        return

    # 按行处理：找到缩进的「vocab_size: <数字> [# ...]」行，只重写数字、
    # 保留缩进、键名和行内注释。yaml.dump 会丢注释，所以不用它。
    # 形如:  "    vocab_size: 26000          # 占位 ..." → "    vocab_size: <real> ..."
    line_pat = re.compile(r"^(\s*vocab_size:\s*)(\d+)(\s*)(#.*)?$", re.MULTILINE)

    current = None
    def _grab(m):
        nonlocal current
        current = int(m.group(2))
        return m.group(0)
    line_pat.sub(_grab, train_cfg_path.read_text(encoding="utf-8"), count=1)

    if current is None:
        print(f"  [skip] training_yl.yaml 未找到 vocab_size，请手动改为 {real_vocab}")
        return
    if current == real_vocab:
        return

    def _repl(m):
        indent_key, _old, ws, comment = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"{indent_key}{real_vocab}{ws}{comment or ''}"

    new_text = line_pat.sub(_repl, train_cfg_path.read_text(encoding="utf-8"), count=1)
    train_cfg_path.write_text(new_text, encoding="utf-8")
    print(f"  [sync] training_yl.yaml vocab_size: {current} → {real_vocab}")


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
    _sync_vocab_size(args.config)
    print("\nYL corpus generation complete!")


if __name__ == "__main__":
    main()
