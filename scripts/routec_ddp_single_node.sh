#!/usr/bin/env bash
# scripts/routec_ddp_single_node.sh
#
# 单机多卡 Route C 微调（torchrun 默认）。
#
# 用法：
#   NGPUS=8 bash scripts/routec_ddp_single_node.sh
#   NGPUS=4 MAX_STEPS=5000 CONFIG=configs/routec/default_multinode.json bash scripts/routec_ddp_single_node.sh
#
# 等价于你 risk_control_2 项目中的 Hu_start_ddp_05B.sh。
#
# 环境变量（均有默认值）：
#   NGPUS         本机参与训练的 GPU 数（必填，默认 8）
#   CONFIG        Route C JSON 配置路径（默认 configs/routec/default_multinode.json）
#   MAX_STEPS     覆盖 config.max_steps
#   MASTER_PORT   torchrun 端口（默认 29500），与其他训练任务冲突时改

set -euo pipefail

# 切到项目根目录（兼容从任意位置调用）
cd "$(dirname "$0")/.."

NGPUS="${NGPUS:-8}"
CONFIG="${CONFIG:-configs/routec/default_multinode.json}"
MAX_STEPS="${MAX_STEPS:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
AUTO_LOAD="${AUTO_LOAD:-}"

EXTRA_ARGS=()
[ -n "$MAX_STEPS" ] && EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
[ -n "$AUTO_LOAD" ] && EXTRA_ARGS+=(--auto-load)

echo "================================================================"
echo "Route C: 单机 ${NGPUS} 卡 DDP 微调"
echo "  config:     $CONFIG"
echo "  master_port:$MASTER_PORT"
echo "================================================================"

torchrun --nproc-per-node="${NGPUS}" \
    --master-port="${MASTER_PORT}" \
    scripts/step_06_finetune_routec.py \
    --config "$CONFIG" \
    "${EXTRA_ARGS[@]}"
