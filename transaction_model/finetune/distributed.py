"""Route C 分布式训练工具（torch native DDP，无 DeepSpeed 依赖）。

参考 risk_control_2/utils/setup.py + engine/strategy.py::DDPStrategy 的模式。

核心 API：
  - setup_distributed()      读 env → init_process_group → 返回 (local_rank, rank, world_size)
  - cleanup_distributed()    销毁 process_group
  - is_main_process()        rank==0 或未初始化
  - reduce_tensor(t, op)     跨 rank 平均/求和（用于日志同步）
  - gather_predictions(...)  预测结果拼到 rank 0（分布式评估）

向后兼容：未设 LOCAL_RANK 环境变量时全部退化为单卡（local_rank=0, rank=0, world_size=1）。
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 60,
) -> Tuple[int, int, int]:
    """初始化分布式训练。

    完全依赖 torchrun/deepspeed 注入的环境变量，不自己解析 argparse：
      LOCAL_RANK     本地 GPU id（每节点内）
      RANK           全局 rank
      WORLD_SIZE     全局进程数
      (NCCL_SOCKET_IFNAME / MASTER_ADDR / MASTER_PORT 由 torchrun 自动设)

    未设 LOCAL_RANK 时判定为单卡模式，跳过 init_process_group。

    Returns:
        (local_rank, rank, world_size)。单卡时返回 (0, 0, 1)。
    """
    if "LOCAL_RANK" not in os.environ:
        logger.info("LOCAL_RANK not set → single-process mode")
        return 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        # HIGHEST_PRIORITY_LAST 让 nccl 在某些卡上做 allreduce 时少一份显存复制
        logger.info(
            f"init_process_group(backend={backend}) rank={rank}/{world_size} "
            f"local_rank={local_rank}"
        )
    return local_rank, rank, world_size


def cleanup_distributed() -> None:
    """销毁 process group，单卡时 no-op。"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    """rank==0 或未初始化（单卡）。仅用于日志/写文件守卫。"""
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank()


def reduce_tensor(
    tensor: torch.Tensor,
    op: str = "mean",
) -> torch.Tensor:
    """跨 rank 同步一个标量或张量，返回平均值（op='mean'）或求和（op='sum'）。

    单卡时直接返回原张量；非分布式 init 时同样 no-op。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if op == "mean":
        rt /= dist.get_world_size()
    return rt


def gather_predictions(
    logits: torch.Tensor,   # [local_n, num_class]
    labels: torch.Tensor,   # [local_n]
    users: List[str],       # len=local_n
    amounts: torch.Tensor,  # [local_n]
    to_rank: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str], Optional[torch.Tensor]]:
    """把每个 rank 的预测拼到 to_rank。

    不同 rank 的 batch 大小可能不等长 → 先 all_gather 各 rank 的本地 N，
    pad 到 max(N_rank)，gather，rank 0 trim 回原长度。

    Returns:
        在 to_rank：(gathered_logits, gathered_labels, gathered_users, gathered_amounts)
        其它 rank：(None, None, [], None)（节省内存）
        单卡时原样返回。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return logits, labels, users, amounts

    world_size = dist.get_world_size()
    rank = get_rank()
    local_n = logits.size(0)
    num_class = logits.size(-1)

    # Step 1: 拿到所有 rank 的 local_n
    n_tensor = torch.tensor([local_n], device=logits.device, dtype=torch.long)
    n_list = [torch.zeros(1, dtype=torch.long, device=logits.device) for _ in range(world_size)]
    dist.all_gather(n_list, n_tensor)
    n_list = [int(t.item()) for t in n_list]
    max_n = max(n_list)

    if max_n == 0:
        # 所有 rank 都空
        if rank == to_rank:
            return (
                torch.zeros((0, num_class), device=logits.device, dtype=logits.dtype),
                torch.zeros(0, device=logits.device, dtype=labels.dtype),
                [],
                torch.zeros(0, device=logits.device, dtype=amounts.dtype),
            )
        return None, None, [], None

    # Step 2: pad 到 max_n 后 gather
    def _pad(t: torch.Tensor) -> torch.Tensor:
        if t.size(0) < max_n:
            pad_shape = list(t.shape)
            pad_shape[0] = max_n - t.size(0)
            pad_t = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
            return torch.cat([t, pad_t], dim=0)
        return t

    logits_p = _pad(logits)
    labels_p = _pad(labels)
    amounts_p = _pad(amounts)

    def _gather(t: torch.Tensor):
        out = [torch.zeros_like(t) for _ in range(world_size)]
        dist.gather(t, gather_list=out if rank == to_rank else None, dst=to_rank)
        return out

    logits_gathered = _gather(logits_p)
    labels_gathered = _gather(labels_p)
    amounts_gathered = _gather(amounts_p)

    if rank != to_rank:
        return None, None, [], None

    # Step 3: rank 0 trim 回各 rank 的真实 n。users 是 list-of-str，需手动同步
    out_logits, out_labels, out_amounts = [], [], []
    for i in range(world_size):
        n = n_list[i]
        out_logits.append(logits_gathered[i][:n])
        out_labels.append(labels_gathered[i][:n])
        out_amounts.append(amounts_gathered[i][:n])

    # Step 4: 字符串列表用 all_gather_object 同步
    users_per_rank = [None] * world_size
    dist.all_gather_object(users_per_rank, users)
    out_users = users_per_rank if rank == to_rank else []
    flat_users: List[str] = []
    if rank == to_rank:
        # 各 rank 的 users 顺序与 logits_gathered 一一对应
        for i in range(world_size):
            flat_users.extend(users_per_rank[i])

    return (
        torch.cat(out_logits, dim=0),
        torch.cat(out_labels, dim=0),
        flat_users,
        torch.cat(out_amounts, dim=0),
    )


def barrier() -> None:
    """屏障；非分布式时 no-op。"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
