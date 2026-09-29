"""可视化层：把结构化快照渲染成可看的画面。

对应需求：FR-05（可观测性）
对应文档：《技术方案.md》§1、交付形态「Web + 录屏双轨」

本层是**只读消费者**——它依赖 ``schemas``，但不被任何算法层依赖。
这保证了"关掉可视化不影响算法"，也符合《目录结构.md》的单向依赖规则。
"""

from __future__ import annotations

from .render import SnapshotRenderer, render_snapshot

__all__ = ["SnapshotRenderer", "render_snapshot"]
