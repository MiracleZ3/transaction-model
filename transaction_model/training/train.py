"""Decoder 基础模型训练"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transaction_model.config import get_project_root, load_config, resolve_path


def get_python_executable() -> str:
    """获取 Python 解释器路径"""
    return sys.executable


def get_torchrun_cmd(
    nproc: int = 8,
    nnodes: int | None = None,
    node_rank: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> list[str]:
    """生成 torchrun 命令前缀（支持多机多卡）。

    单机多卡（默认）：`torchrun --nproc-per-node=N`
    多机多卡：传 `nnodes, node_rank, master_addr, master_port` 即可：
        `torchrun --nproc-per-node=N --nnodes=M --node-rank=R
                 --master-addr=A --master-port=P`
    """
    cmd = ["torchrun", f"--nproc-per-node={nproc}"]
    if nnodes is not None:
        cmd.append(f"--nnodes={nnodes}")
    if node_rank is not None:
        cmd.append(f"--node-rank={node_rank}")
    if master_addr is not None:
        cmd.append(f"--master-addr={master_addr}")
    if master_port is not None:
        cmd.append(f"--master-port={master_port}")
    return cmd


def build_train_command(
    config_path: str | Path,
    train_corpus: str | Path,
    val_corpus: str | Path,
    output_dir: str | Path | None = None,
    max_steps: int | None = None,
    global_batch_size: int | None = None,
    local_batch_size: int | None = None,
    nproc: int = 1,
) -> list[str]:
    """构建训练命令

    Args:
        config_path: YAML 配置文件路径
        train_corpus: 训练语料路径
        val_corpus: 验证语料路径
        output_dir: 检查点输出目录
        max_steps: 最大训练步数 (覆盖配置)
        global_batch_size: 全局批次大小 (覆盖配置)
        local_batch_size: 本地批次大小 (覆盖配置)
        nproc: GPU 数量 (1 = 单卡，>1 = torchrun)

    Returns:
        命令列表
    """
    project_root = get_project_root()
    train_script = project_root / "transaction_model" / "training" / "run_training.py"

    if nproc > 1:
        cmd = get_torchrun_cmd(nproc)
    else:
        cmd = [get_python_executable()]

    cmd.extend([
        str(train_script),
        "-c", str(config_path),
        "--dataset.data_path", str(train_corpus),
        "--validation_dataset.data_path", str(val_corpus),
    ])

    if max_steps is not None:
        cmd.extend(["--step_scheduler.max_steps", str(max_steps)])
    if global_batch_size is not None:
        cmd.extend(["--step_scheduler.global_batch_size", str(global_batch_size)])
    if local_batch_size is not None:
        cmd.extend(["--step_scheduler.local_batch_size", str(local_batch_size)])
    if output_dir is not None:
        cmd.extend(["--checkpoint.checkpoint_dir", str(output_dir)])

    return cmd


def launch_training(
    config_name: str = "training",
    max_steps: int | None = None,
    nproc: int = 1,
    demo: bool = False,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """启动训练

    Args:
        config_name: 配置文件名 (不含 .yaml)
        max_steps: 覆盖最大步数
        nproc: GPU 数量
        demo: 是否 demo 模式 (30步)
        capture_output: 是否捕获输出

    Returns:
        subprocess 结果
    """
    cfg = load_config(config_name)
    paths = cfg.get("paths", {})

    config_path = resolve_path(f"configs/{config_name}.yaml")
    train_corpus = resolve_path(paths.get("train_corpus", "data/decoder_corpus/train_corpus.txt"))
    val_corpus = resolve_path(paths.get("val_corpus", "data/decoder_corpus/val_corpus.txt"))

    # 前置检查
    assert train_corpus.exists(), f"Training corpus not found: {train_corpus} (run step_02 first)"
    assert val_corpus.exists(), f"Validation corpus not found: {val_corpus} (run step_02 first)"

    output_dir = None
    if demo:
        output_dir = resolve_path("models/decoder-demo/checkpoints")
        output_dir.mkdir(parents=True, exist_ok=True)
        max_steps = max_steps or 30

    cmd = build_train_command(
        config_path=config_path,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        output_dir=output_dir,
        max_steps=max_steps,
        nproc=nproc,
    )

    print("Launching training:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
    )

    if result.returncode != 0:
        print(f"\nTraining exited with code {result.returncode}")
    else:
        print("\nTraining complete!")

    return result
