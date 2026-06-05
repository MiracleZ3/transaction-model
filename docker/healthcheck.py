"""容器健康检查

Docker HEALTHCHECK 使用：
    HEALTHCHECK CMD python /workspace/docker/healthcheck.py || exit 1

退出码：
    0 = 健康
    1 = 不健康（message 写到 stderr）
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

WORKSPACE = Path("/workspace")
DATA_DIR = WORKSPACE / "data"


def _check_workspace_writable() -> str | None:
    """检查 /workspace 是否可写（在一些只读 mount 场景会失败）。"""
    try:
        with tempfile.NamedTemporaryFile(
            dir=WORKSPACE, prefix=".healthcheck-", suffix=".tmp", delete=True
        ):
            pass
    except Exception as e:  # pragma: no cover - depends on environment
        return f"workspace not writable: {e}"
    return None


def _check_import() -> str | None:
    """核心包能否 import。"""
    try:
        import transaction_model  # noqa: F401
    except Exception as e:  # pragma: no cover
        return f"import transaction_model failed: {e}"
    return None


def _check_data_mount() -> str | None:
    """温柔的提醒：data 目录若未挂载，pipeline 会失败；但不算致命。"""
    if not DATA_DIR.is_dir():
        return (
            f"warn: {DATA_DIR} not present; "
            "pipeline will need it mounted to access splits/corpus/outputs"
        )
    return None


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for check in (_check_workspace_writable, _check_import):
        msg = check()
        if msg:
            errors.append(msg)

    msg = _check_data_mount()
    if msg:
        warnings.append(msg)

    if errors:
        print("UNHEALTHY:", "; ".join(errors), file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        return 1

    if warnings:
        print("HEALTHY (with warnings):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("HEALTHY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
