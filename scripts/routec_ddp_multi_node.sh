#!/usr/bin/env bash
# scripts/routec_ddp_multi_node.sh
#
# 多机多卡 Route C 微调（torchrun 多节点）。
#
# 用法（每台机器各自起一份，NNODES/NODE_RANK/MASTER_ADDR 不同）：
#
#   # node 0 (master)：
#   NNODES=2 NNODE_RANK=0 NGPUS=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash scripts/routec_ddp_multi_node.sh
#
#   # node 1：
#   NNODES=2 NNODE_RANK=1 NGPUS=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash scripts/routec_ddp_multi_node.sh
#
# 关键约束：
#   - 所有节点的 NNODES / MASTER_ADDR / MASTER_PORT 必须一致
#   - 每个节点的 NNODE_RANK 不同（0..NNODES-1）
#   - 模型 ckpt / 数据 path 必须在每个节点上 path-reachable（NFS / rsync / cp）
#   - python/torch/transformers/peft 版本必须在每个节点一致
#
# 等价于 risk_control_2 的 multi-node 用法，但用 torchrun 代替 deepspeed launcher。

set -euo pipefail

cd "$(dirname "$0")/.."

NNODES="${NNODES:-1}"
NNODE_RANK="${NNODE_RANK:-0}"
NGPUS="${NGPUS:-8}"
CONFIG="${CONFIG:-configs/routec/default_multinode.json}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-}"
AUTO_LOAD="${AUTO_LOAD:-}"

EXTRA_ARGS=()
[ -n "$MAX_STEPS" ] && EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
[ -n "$AUTO_LOAD" ] && EXTRA_ARGS+=(--auto-load)

echo "================================================================"
echo "Route C: 多机多卡 DDP"
echo "  nnodes:       $NNODES"
echo "  node_rank:    $NNODE_RANK"
echo "  gpus/node:    $NGPUS"
echo "  total gpus:   $((NNODES * NGPUS))"
echo "  master:       $MASTER_ADDR:$MASTER_PORT"
echo "  config:       $CONFIG"
echo "================================================================"

torchrun \
    --nproc-per-node="${NGPUS}" \
    --nnodes="${NNODES}" \
    --node-rank="${NNODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    scripts/step_06_finetune_routec.py \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"
