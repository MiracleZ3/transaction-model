"""TabFormer 数据集下载与解压"""
from __future__ import annotations

import tarfile
from pathlib import Path
from urllib.request import urlretrieve

from transaction_model.config import load_config, resolve_path


def download_tabformer(
    download_url: str | None = None,
    data_dir: str | None = None,
) -> Path:
    """下载并解压 TabFormer 数据集

    Args:
        download_url: 下载链接 (默认从 dataset.yaml 读取)
        data_dir: 数据目录 (默认从 dataset.yaml 读取)

    Returns:
        解压后的 CSV 文件路径
    """
    cfg = load_config("dataset")["dataset"]
    url = download_url or cfg["download_url"]
    raw_dir = resolve_path(data_dir or cfg["raw_csv"]).parent

    raw_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = raw_dir.parent / "transactions.tgz"
    csv_path = raw_dir / "card_transaction.v1.csv"

    if not tgz_path.exists():
        print("Downloading transactions.tgz from IBM Box...")
        urlretrieve(url, tgz_path)

    if not csv_path.exists():
        print("Extracting transactions.tgz...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=raw_dir, filter='data')

    print(f"Dataset ready: {csv_path}")
    return csv_path
