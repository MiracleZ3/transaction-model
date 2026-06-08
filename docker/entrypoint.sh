#!/usr/bin/env bash
# docker/entrypoint.sh
#
# 统一容器入口：把 docker run IMAGE CMD 映射到对应 step 脚本。
# 所有命令都在 /workspace 下执行（已由 Dockerfile WORKDIR 设置）。

set -euo pipefail

cd /workspace

subcommand="${1:-help}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

print_help() {
    cat <<'EOF'
Transaction Model container entrypoint

Usage:
    docker run <image> <command> [args...]

Commands:
    pipeline [--steps S1,S2,...] [--demo]   Full pipeline (default: data,tokenize,extract,detect)
    data      [--skip-download] [--skip-baseline]   Step 1: dataset + baseline
    tokenize  [--force]                       Step 2: corpus generation (GPU)
    train     [--demo] [--nproc N]            Step 3: decoder training (GPU)
    extract   [--force]                       Step 4: embedding extraction (GPU)
    detect    [--no-plot]                     Step 5: fraud detection comparison
    test      [pytest args...]                Run pytest
    notebook  (optional)                      Start Jupyter Lab on port 8888
    bash      [args...]                       Drop into bash shell

Examples:
    # CPU: data step on existing parquet
    docker run --rm -v $(pwd)/data:/workspace/data tm-cpu:latest \
        data --skip-download

    # GPU: demo training (30 steps, 1 GPU)
    docker run --rm --gpus all \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/models:/workspace/models \
        tm-gpu:latest train --demo

    # Pipeline with profile-selected steps
    docker run --rm tm-cpu:latest pipeline --steps data,detect

    # Run pytest
    docker run --rm tm-cpu:latest test

Environment variables:
    EXTRA_ARGS_APPEND   Append extra args to every command
EOF
}

case "$subcommand" in
    --help|-h|help)
        print_help
        ;;

    pipeline)
        exec python scripts/run_pipeline.py "$@"
        ;;

    data)
        exec python scripts/step_01_dataset_baseline.py "$@"
        ;;

    tokenize)
        exec python scripts/step_02_tokenize_corpus.py "$@"
        ;;

    train)
        # 支持 train --nproc N：转成 torchrun
        nproc=1
        if [[ "${1:-}" == "--nproc" ]]; then
            nproc="${2:-1}"
            shift 2
        fi
        if [[ "${1:-}" == "--demo" ]]; then
            # demo 模式走 scripts/step_03 入口（内部用 python，不 torchrun）
            exec python scripts/step_03_train_model.py --demo
        fi
        # 非 demo 模式直接 torchrun
        exec torchrun --nproc-per-node="$nproc" \
            transaction_model/training/run_training.py "$@"
        ;;

    extract)
        exec python scripts/step_04_extract_embeddings.py "$@"
        ;;

    detect)
        exec python scripts/step_05_fraud_detection.py "$@"
        ;;

    test)
        exec pytest tests/ -v "$@"
        ;;

    notebook)
        # 需要在镜像里 pip install jupyterlab；默认未装
        if ! command -v jupyter >/dev/null 2>&1; then
            echo "jupyter not installed in this image." >&2
            echo "Rebuild with: pip install jupyterlab" >&2
            exit 1
        fi
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser "$@"
        ;;

    bash|sh)
        exec /bin/bash "$@"
        ;;

    python|python3)
        exec python "$@"
        ;;

    *)
        echo "Unknown command: $subcommand" >&2
        echo "Run with --help to see available commands." >&2
        exit 2
        ;;
esac
