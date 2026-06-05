"""数据探索可视化"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from transaction_model.constants import MCC_INDUSTRY_RANGES


def mcc_label(code: int) -> str:
    """将 MCC 代码映射为行业标签"""
    code = int(code)
    for lo, hi, name in MCC_INDUSTRY_RANGES:
        if lo <= code <= hi:
            return f"{name} ({code})"
    return str(code)


def plot_data_overview(
    train_gdf,
    all_gdf,
    train_cutoff,
    test_cutoff,
    save_path: str | None = None,
) -> None:
    """绘制数据概览 4 子图

    图1: 欺诈分布 (对数尺度)
    图2: 各 MCC 欺诈率 (水平条形图)
    图3: 每用户交易数分布
    图4: 时间趋势 + 分割边界

    Args:
        train_gdf: 训练集 DataFrame
        all_gdf: 全量 DataFrame (含 date 列)
        train_cutoff: 训练/验证分割日期
        test_cutoff: 验证/测试分割日期
        save_path: 图片保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # 1. 欺诈分布
    ax1 = axes[0, 0]
    if hasattr(train_gdf['Is Fraud?'], 'to_pandas'):
        fraud_counts_pd = train_gdf['Is Fraud?'].value_counts().to_pandas()
    else:
        fraud_counts_pd = train_gdf['Is Fraud?'].value_counts()
    colors = ['#2ca02c', '#d62728']
    bars = ax1.bar(fraud_counts_pd.index, fraud_counts_pd.values, color=colors, edgecolor='black')
    ax1.set_title('Fraud Distribution (Training Set)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, fraud_counts_pd.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylabel('Transaction Count (log scale)')

    # 2. MCC 欺诈率
    ax4 = axes[0, 1]
    _fraud_flag = (train_gdf['Is Fraud?'].str.lower() == 'yes').astype('int32')
    mcc_fraud_sum = _fraud_flag.groupby(train_gdf['MCC']).sum()
    mcc_total = _fraud_flag.groupby(train_gdf['MCC']).count()
    mcc_fraud_rate = (mcc_fraud_sum / mcc_total * 100)
    min_txns = 500
    mask = mcc_total >= min_txns
    top_mccs = mcc_fraud_rate[mask].sort_values(ascending=False).head(15)
    if hasattr(top_mccs.index, 'values_host'):
        mcc_codes = top_mccs.index.values_host
        rates = top_mccs.values_host
    else:
        mcc_codes = top_mccs.index.values
        rates = top_mccs.values
    labels = [mcc_label(c) for c in mcc_codes]
    bar_colors = ['#d62728' if r > 1.0 else '#ff7f0e' for r in rates]
    bars = ax4.barh(range(len(labels)), rates, color=bar_colors, edgecolor='black', alpha=0.85)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel('Fraud Rate (%)')
    ax4.set_title(f'Fraud Rate by Top MCC (min {min_txns} txns)', fontsize=13, fontweight='bold')
    for bar, rate in zip(bars, rates):
        ax4.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{rate:.2f}%', va='center', fontsize=9)

    # 3. 每用户交易数
    ax2 = axes[1, 0]
    txn_per_user = train_gdf.groupby('User')['User'].count()
    if hasattr(txn_per_user, 'values_host'):
        txn_per_user = txn_per_user.values_host
    else:
        txn_per_user = txn_per_user.values
    ax2.hist(txn_per_user, bins=50, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax2.set_title('Transactions per User (Training Set)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Number of Transactions')
    ax2.set_ylabel('Number of Users')
    median_val = int(np.median(txn_per_user))
    ax2.axvline(x=median_val, color='red', linestyle='--', linewidth=1.5,
                label=f'Median: {median_val:,}')
    ax2.legend(fontsize=10)

    # 4. 时间趋势
    ax3 = axes[1, 1]
    all_gdf['_year'] = all_gdf['Year'].astype(int)
    all_gdf['_month'] = all_gdf['Month'].astype(int)
    monthly = all_gdf.groupby(['_year', '_month']).size().reset_index(name='count')
    monthly['period'] = monthly['_year'].astype(str) + '-' + monthly['_month'].astype(str).str.zfill(2)
    monthly = monthly.sort_values('period').reset_index(drop=True)
    if hasattr(monthly, 'to_pandas'):
        monthly = monthly.to_pandas()
    ax3.plot(range(len(monthly)), monthly['count'].values, color='#ff7f0e', linewidth=1.5)

    train_cutoff_period = f"{train_cutoff.year}-{train_cutoff.month:02d}"
    test_cutoff_period = f"{test_cutoff.year}-{test_cutoff.month:02d}"
    tc_idx = monthly.index[monthly['period'] == train_cutoff_period]
    vc_idx = monthly.index[monthly['period'] == test_cutoff_period]
    tc_x = tc_idx[0] if len(tc_idx) > 0 else None
    vc_x = vc_idx[0] if len(vc_idx) > 0 else None

    if tc_x is not None and vc_x is not None:
        ax3.axvline(x=tc_x, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax3.axvline(x=vc_x, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax3.axvspan(0, tc_x, alpha=0.07, color='green', label='Train')
        ax3.axvspan(tc_x, vc_x, alpha=0.07, color='blue', label='Val')
        ax3.axvspan(vc_x, len(monthly)-1, alpha=0.07, color='red', label='Test')
        ax3.legend(loc='upper left', fontsize=9)

    tick_positions = range(0, len(monthly), 12)
    tick_labels_list = [monthly['period'].iloc[i] for i in tick_positions]
    ax3.set_xticks(list(tick_positions))
    ax3.set_xticklabels(tick_labels_list, rotation=45)
    ax3.set_title('Transactions Over Time (with Temporal Splits)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Monthly Transaction Count')
    ax3.set_xlabel('Year-Month')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_baseline_metrics(test_auc: float, test_ap: float, save_path: str | None = None) -> None:
    """绘制基线指标柱状图"""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Test AUROC', 'Test AUPRC']
    values = [test_auc, test_ap]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('XGBoost Baseline Metrics (Test Set)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
