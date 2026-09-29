"""后处理层：滤镜推荐（FR-08）与修图（FR-11，P2 未落地）。

对应文档：《技术方案.md》§2.5 后处理层

**本层不在实时回路内**：它发生在"用户按下快门之后"，对延迟不敏感，
因此可以引入较重的图像处理。这与语言层同理——分清哪些能力必须实时、
哪些可以异步，是本项目"60% 实时工程"的具体体现。
"""

from __future__ import annotations

from .filter_recommend import (
    FilterRecommendation,
    FilterRecommender,
    build_filter_recommender,
)

__all__ = [
    "FilterRecommendation",
    "FilterRecommender",
    "build_filter_recommender",
]
