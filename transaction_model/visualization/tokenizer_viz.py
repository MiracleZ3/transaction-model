"""Tokenizer 对比可视化"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def compare_tokenizers(
    financial_tokens: list[str],
    raw_tabular_text: str,
    context_window: int = 4096,
    save_path: str | None = None,
) -> dict:
    """对比 Financial Tokenizer 和 GPT-2 BPE Tokenizer

    Args:
        financial_tokens: 金融分词器输出的 token 列表
        raw_tabular_text: 原始表格文本
        context_window: 上下文窗口大小
        save_path: 图片保存路径

    Returns:
        对比结果字典
    """
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tokens = gpt2_tokenizer.tokenize(raw_tabular_text)

    fin_count = len(financial_tokens)
    gpt2_count = len(gpt2_tokens)

    fin_per_txn = fin_count - 2  # 减去 <bos>/<eos>
    fin_txns_per_seq = context_window // (fin_per_txn + 1)
    gpt2_txns_per_seq = context_window // (gpt2_count + 1)

    # 打印对比
    print("=" * 70)
    print("TOKENIZER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Tokenizer':<20} {'Tokens/Txn':>12} {'Compression':>14} {'Txns in ' + str(context_window):>15}")
    print("-" * 70)
    print(f"{'Financial':<20} {fin_per_txn:>12} {'1x (baseline)':>14} {f'~{fin_txns_per_seq}':>15}")
    print(f"{'GPT-2 (BPE)':<20} {gpt2_count:>12} {f'{gpt2_count/fin_per_txn:.1f}x more':>14} {f'~{gpt2_txns_per_seq}':>15}")

    # 绘制对比图
    if save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Tokens per transaction
        ax1.bar(['Financial', 'GPT-2 (BPE)'], [fin_per_txn, gpt2_count],
                color=['#2ca02c', '#1f77b4'], edgecolor='black')
        ax1.set_ylabel('Tokens per Transaction')
        ax1.set_title('Token Efficiency')

        # Transactions per sequence
        ax2.bar(['Financial', 'GPT-2 (BPE)'], [fin_txns_per_seq, gpt2_txns_per_seq],
                color=['#2ca02c', '#1f77b4'], edgecolor='black')
        ax2.set_ylabel('Transactions per Sequence')
        ax2.set_title(f'Sequence Capacity (window={context_window})')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()

    return {
        "financial_tokens_per_txn": fin_per_txn,
        "gpt2_tokens_per_txn": gpt2_count,
        "financial_txns_per_seq": fin_txns_per_seq,
        "gpt2_txns_per_seq": gpt2_txns_per_seq,
        "compression_ratio": gpt2_count / fin_per_txn,
    }
