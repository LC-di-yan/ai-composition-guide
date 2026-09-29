"""构图决策层：从感知结果推导"该怎么拍"。

对应需求：FR-01（距离校准）、FR-03（构图评估）、FR-04（动作指令）
对应文档：《技术方案.md》§1 构图决策层

本层是项目的**核心**：调研指出"神奇感"中 30% 来自构图评分模型的选点，
而真正的护城河在于这部分输出是否稳定、可解释、可执行。

本层只依赖 ``schemas`` 的数据类型，不依赖 ``perception`` 的任何实现
（《目录结构.md》§4 分层依赖规则）。
"""

from .candidate import Candidate, CandidateGenerator
from .differ import CommandDiffer, DeltaResult
from .distance import DistanceEstimator
from .rules import (
    ThirdsAlignment,
    balance_score,
    classify_pattern,
    describe_pattern,
    describe_shot_size,
    headroom_score,
    lead_room_score,
    saliency_center_score,
    subject_center_score,
    thirds_alignment,
)
from .scorer import HeuristicCompositionScorer, SAMPNetScorer

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "CommandDiffer",
    "DeltaResult",
    "DistanceEstimator",
    "HeuristicCompositionScorer",
    "SAMPNetScorer",
    "ThirdsAlignment",
    "balance_score",
    "classify_pattern",
    "describe_pattern",
    "describe_shot_size",
    "headroom_score",
    "lead_room_score",
    "saliency_center_score",
    "subject_center_score",
    "thirds_alignment",
    "build_scorer",
]


def build_scorer(settings=None):
    """按配置构造构图评分器。

    当前恒返回 :class:`HeuristicCompositionScorer`。SAMPNet 接入桩存在但
    未启用（权重依赖未满足），详见 :class:`SAMPNetScorer` 的说明。
    """
    if settings is None:
        from ..settings import get_settings

        settings = get_settings()

    ccfg = settings.composition
    generator = CandidateGenerator(
        min_area_ratio=ccfg.candidate.min_area_ratio,
        max_area_ratio=ccfg.candidate.max_area_ratio,
        grid_x=ccfg.candidate.grid_x,
        grid_y=ccfg.candidate.grid_y,
        min_aspect=ccfg.candidate.min_aspect,
        max_aspect=ccfg.candidate.max_aspect,
    )
    return HeuristicCompositionScorer(
        scoring=ccfg.scoring,
        candidate_gen=generator,
        top_k=ccfg.candidate.top_k,
    )
