"""Checkpoint 路径解析工具：自动从 NeMo AutoModel FSDP2 训练产物中
找到 HuggingFace 兼容目录（含 config.json + *.safetensors）。

NeMo 训练的典型目录结构::

    <output_dir>/                         # 如 models/decoder-yl/
    └── checkpoints/
        ├── step_X/mp_rank_0/model.pt      # FSDP2 分片 ckpt（Route C / Step 4 都不直接吃）
        └── consolidated/safetensors/      # save_consolidated=true 时由 NeMo 导出
            ├── config.json
            └── model.safetensors          # 或 sharded model-0000X-of-0000N.safetensors

下游推理（inference/extract.py）和 Route C 微调（finetune/models/llama_encoder.py）
都需要一个 HF 目录。原代码要求用户手工 mv/symlink `consolidated/safetensors/`
到 `<output_dir>` 根——很易踩坑（如服务器报 OSError: no file named model.safetensors）。

本 helper 让传入的 model_path 可以是任意一种：

  1. ``models/decoder-yl``                          → 自动展开成下方的 HF 子目录
  2. ``models/decoder-yl/checkpoints``              → 同上
  3. ``models/decoder-yl/checkpoints/consolidated/safetensors`` → 直接命中
  4. 已经是一个合法 HF 目录（含 config.json + *.safetensors）→ 直接返回

无加权重的 config.json 目录会返回（让 transformers 用更清晰的错误信息报）；
完全找不到时返回 None，调用方自行决定怎么报错。
"""
from __future__ import annotations

from pathlib import Path


def resolve_hf_model_dir(model_path: str | Path) -> Path:
    """从可能的 NeMo 训练产出路径里解出 HF 兼容目录。

    Args:
        model_path: 用户传入的路径（可能是根目录、checkpoints/ 或 consolidated/）。

    Returns:
        含 config.json + safetensors 的目录 Path。

    Raises:
        FileNotFoundError: 无法在树中找到 HF 目录。
    """
    p = Path(model_path)
    found = _find_consolidated_model(p)
    if found is None:
        raise FileNotFoundError(
            f"No HuggingFace model dir (config.json + *.safetensors) found under {p}. "
            f"Did step_03 finish with save_consolidated=true? Look under "
            f"{p}/checkpoints/consolidated/safetensors/ and either point llama.model_path "
            f"there or run `ln -s {p}/checkpoints/consolidated/safetensors/* {p}/`."
        )
    return found


def _find_consolidated_model(ckpt_dir: Path) -> Path | None:
    """在 ckpt_dir 树中查找含 config.json 的 HuggingFace 兼容目录。

    优先级：
      1. ckpt_dir 自身合法（含 config.json + safetensors）。
      2. 直接子目录合法（如 consolidated/、safetensors/、<model_repo_id>/）。
      3. checkpoints/<name>/ 二层嵌套（NeMo 默认 consolidated/safetensors/）。
      4. rglob 兜底——但只接受同时含权重文件的目录，无助避免误命中仅 config 的子项目。
    """
    # 1. 自身：含 config + 权重 → 直接 OK
    if (ckpt_dir / "config.json").exists() and _has_weights(ckpt_dir):
        return ckpt_dir

    # 把 ckpt_dir 当成根，也允许直接搜它下面的 checkpoints/ 子树
    candidates: list[Path] = []
    if (ckpt_dir / "checkpoints").is_dir():
        candidates.append(ckpt_dir / "checkpoints")
    candidates.append(ckpt_dir)

    for root in candidates:
        # 2. 直接子目录（如 .../consolidated/、.../safetensors/、.../yl-financial-decoder/）
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "config.json").exists() and _has_weights(child):
                return child

        # 3. 二层嵌套 root/<a>/<b>/config.json（NeMo: checkpoints/consolidated/safetensors/）
        for a in sorted(root.iterdir()):
            if not a.is_dir():
                continue
            for b in sorted(a.iterdir()):
                if b.is_dir() and (b / "config.json").exists() and _has_weights(b):
                    return b

        # 4. rglob 兜底：任意深度，要求同目录下有权重文件
        for cf in root.rglob("config.json"):
            parent = cf.parent
            if _has_weights(parent):
                return parent

    return None


def _has_weights(d: Path) -> bool:
    return any(d.glob("*.safetensors")) or bool(list(d.glob("pytorch_model*.bin")))
