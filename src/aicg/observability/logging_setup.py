"""统一日志。

对应需求：NFR-O4（运行日志）
规则（《编码规范.md》§3.4）：禁止 print；禁止在日志中输出 API Key 或完整请求头。
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO", force: bool = False) -> None:
    """初始化根日志配置。多次调用幂等。

    Args:
        level: 日志级别名，如 ``DEBUG`` / ``INFO``。
        force: 为 True 时强制重新配置（供测试使用）。
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("aicg")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取命名日志器，统一挂在 ``aicg`` 命名空间下。"""
    if not name.startswith("aicg"):
        name = f"aicg.{name}"
    return logging.getLogger(name)
