"""命令行接口（CLI）。

对应文档：《目录结构.md》、《开发计划.md》
对应需求：全流程可运行（NFR-M4）

入口：``aicg`` 或 ``python -m aicg.cli``。子命令实现见 ``__main__.py``。
"""

from __future__ import annotations

from .__main__ import build_parser, main

__all__ = ["build_parser", "main"]
