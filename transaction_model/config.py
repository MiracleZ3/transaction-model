"""统一配置加载器，从 YAML 文件读取参数"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict[str, Any]:
    """加载指定配置文件 (不含 .yaml 后缀)

    Args:
        name: 配置名，如 "dataset", "tokenizer", "training", "xgboost"

    Returns:
        配置字典
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    """将配置中的相对路径解析为基于项目根的绝对路径"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return get_project_root() / p
