"""构图评分器。

对应需求：FR-03（构图评估）、NFR-O2（建议可溯源）
对应文档：《技术方案.md》§2.3 构图决策层选型

**重要的设计说明（诚实标注）：**

调研给出的首选方案是接入 SAMPNet + CADB 预训练权重。本项目在实现时
遇到的实际约束是：CADB/SAMPNet 权重需从作者仓库单独获取，且其训练
数据以通用场景为主、人像域适配不明。因此本模块采取**两段式**设计：

1. :class:`HeuristicCompositionScorer` —— 基于构图学规则的加权评分，
   当前**默认启用**。它的优势是每一项分数都可解释、可追溯到具体规则
   （直接服务 NFR-O2），且无需权重即可运行。
2. :class:`SAMPNetScorer` —— 预留的模型接入桩，在权重可用时可直接替换，
   接口完全一致（满足 NFR-M1 可替换性）。

这个取舍会在《开发计划.md》§6 简化处清单中如实登记，不伪装成已接入模型。
"""

from __future__ import annotations

import time

import numpy as np

from ..observability import get_logger
from ..schemas import (
    CandidateScore,
    CompositionPattern,
    CompositionResult,
    RuleName,
    RuleViolation,
    Severity,
)
from ..schemas.perception import BBox
from ..settings import ScoringConfig
from ..utils.image import bbox_center, bbox_height, bbox_width
from .candidate import Candidate, CandidateGenerator
from .rules import (
    balance_score,
    classify_pattern,
    describe_pattern,
    describe_shot_size,
    headroom_score,
    lead_room_score,
    saliency_center_score,
    subject_center_score,
    subject_scale_score,
    thirds_alignment,
)

log = get_logger("composition.scorer")


class HeuristicCompositionScorer:
    """基于构图学规则的构图评分器（当前默认实现）。

    评分构成（权重来自 ``configs/default.yaml``）：

    ====================  ==========================================
    子项                  衡量内容
    ====================  ==========================================
    ``thirds``            主体是否贴近三分点
    ``balance``           主体在画面中是否左右/上下严重失衡
    ``headroom``          头顶留白是否合适
    ``lead_room``         朝向侧是否留出空间
    ``subject_scale``     主体大小是否合适（过小 = 该走近）
    ``saliency_center``   视觉重点是否落在框内
    ``subject_center``    主体是否被合理包含（门控项）
    ====================  ==========================================

    最终分为加权和 × 100，取值域 0~100。
    """

    def __init__(
        self,
        scoring: ScoringConfig | None = None,
        candidate_gen: CandidateGenerator | None = None,
        top_k: int = 5,
    ) -> None:
        self.cfg = scoring or ScoringConfig()
        self.generator = candidate_gen or CandidateGenerator()
        self.top_k = top_k

    # ------------------------------------------------------------------
    def score_frame(
        self,
        subject_bbox: BBox | None,
        saliency: np.ndarray | None = None,
        face_yaw: float | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> CompositionResult:
        """对一帧做完整构图评估。

        Args:
            subject_bbox: 主体框；None 表示无主体，走降级路径。
            saliency: 0~1 显著图（可选）。
            face_yaw: 人脸偏航角（可选），用于前方留白判断。
            frame_shape: ``(h, w)``，用于推算候选框宽高比。

        Returns:
            构图评估结果。**保证返回**（无主体时返回三等分通用建议）。
        """
        t0 = time.perf_counter()

        frame_aspect = 3.0 / 4.0
        if frame_shape and frame_shape[0] > 0:
            frame_aspect = frame_shape[1] / float(frame_shape[0])

        degraded = False
        if subject_bbox is None:
            # 降级：无主体时给出通用三分法建议（NFR-R2）
            degraded = True
            subject_bbox = (0.34, 0.30, 0.66, 0.86)

        candidates = self.generator.generate(subject_bbox, frame_aspect)
        if not candidates:
            degraded = True
            candidates = [Candidate(bbox=(0.0, 0.0, 1.0, 1.0), origin="fallback")]

        # ``subject_overflow`` 是**信号**而非**建议**：它表示"主体已大到任何
        # 候选画框都装不下"，此时唯一有意义的动作是后退，而不是"改成全画幅"。
        # 因此把它从"最优框"候选中剔除——否则会出现"最优评分 (7.9) 低于
        # 当前评分 (33.3)"的荒谬结果，让整个建议体系失去可信度。
        scorable = [c for c in candidates if c.origin != "subject_overflow"]

        scored: list[tuple[float, Candidate, dict[str, float]]] = []
        for cand in scorable:
            total, subs = self._score_candidate(
                cand.bbox, subject_bbox, saliency, face_yaw
            )
            scored.append((total, cand, subs))

        scored.sort(key=lambda t: t[0], reverse=True)

        # 当前画面的实际构图（用主体框本身评估，而非最优框）
        current_total, current_subs = self._score_candidate(
            subject_bbox, subject_bbox, saliency, face_yaw
        )

        if scored:
            best_total, best_cand, best_subs = scored[0]
        else:
            # 无任何可评分候选（主体溢出）→ 诚实地把"当前构图"作为最优，
            # 并向上层透传溢出信号，由 differ 依据占比判定"需要后退"。
            best_total, best_cand = current_total, Candidate(bbox=subject_bbox, origin="current")
            best_subs = current_subs
            degraded = True

        # 自洽性保护：最优框的评分不应低于当前构图。若发生（因候选空间受限
        # 或评分函数的非单调性），则以当前构图为准——给用户一个"更差"的
        # 建议是产品事故，宁可说"就这样挺好"。
        if best_total < current_total - 1e-9:
            best_total, best_cand, best_subs = current_total, Candidate(bbox=subject_bbox, origin="current"), current_subs

        pattern_name, pattern_conf = classify_pattern(subject_bbox, saliency, frame_shape)
        try:
            pattern = CompositionPattern(pattern_name)
        except ValueError:
            pattern = CompositionPattern.UNKNOWN

        violations = self._collect_violations(subject_bbox, best_cand.bbox, current_subs)

        return CompositionResult(
            composition_score=round(current_total * 100.0, 2),
            sub_scores={k: round(v, 4) for k, v in current_subs.items()},
            pattern=pattern,
            pattern_label=describe_pattern(pattern_name, subject_bbox),
            shot_size_label=describe_shot_size(subject_bbox),
            best_bbox=best_cand.bbox,
            best_score=round(best_total * 100.0, 2),
            candidate_count=len(candidates),
            rule_violations=violations,
            decision_ms=(time.perf_counter() - t0) * 1000.0,
            degraded=degraded,
            top_candidates=[
                CandidateScore(bbox=c.bbox, score=round(s * 100.0, 2), sub_scores={k: round(v, 4) for k, v in sub.items()})
                for s, c, sub in scored[: self.top_k]
            ],
        )

    # ------------------------------------------------------------------
    def _score_candidate(
        self,
        candidate_bbox: BBox,
        subject_bbox: BBox,
        saliency: np.ndarray | None,
        face_yaw: float | None,
    ) -> tuple[float, dict[str, float]]:
        """对单个候选框评分，返回 ``(加权总分 0~1, 各子项得分)``。

        **参照系说明（真实缺陷修正）**：``balance`` 现在以**画面**为参照
        系。若仍以 ``candidate_bbox`` 为参照，则评估"当前构图"
        （``subject == candidate``）时边距恒为 0，balance 被恒定置零。
        """
        cfg = self.cfg

        thirds = thirds_alignment(subject_bbox, cfg.thirds_tolerance)
        sub = {
            "thirds": thirds.score,
            "balance": balance_score(subject_bbox, candidate_bbox),
            "headroom": headroom_score(subject_bbox, candidate_bbox, cfg.ideal_headroom_ratio),
            "lead_room": lead_room_score(subject_bbox, candidate_bbox, face_yaw),
            "subject_scale": subject_scale_score(subject_bbox, cfg.ideal_subject_height),
            "subject_center": subject_center_score(
                subject_bbox, candidate_bbox, cfg.subject_center_tolerance
            ),
            "saliency_center": (
                saliency_center_score(saliency, candidate_bbox) if saliency is not None else 0.5
            ),
        }

        weights = {
            "thirds": cfg.weight_rule_of_thirds,
            "balance": cfg.weight_balance,
            "headroom": cfg.weight_headroom,
            "lead_room": cfg.weight_lead_room,
            "saliency_center": cfg.weight_saliency_center,
            "subject_scale": cfg.weight_subject_scale,
            # subject_center 作为门控项而非加权项：它是"有效性"而非"优劣"，
            # 因此用乘性惩罚避免框跑偏的候选拿到高分。
        }
        total_w = sum(weights.values()) or 1.0
        weighted = sum(sub[k] * w for k, w in weights.items()) / total_w

        # 门控：主体偏离候选框中心过多则整体降权
        gate = 0.35 + 0.65 * sub["subject_center"]
        return max(0.0, min(1.0, weighted * gate)), sub

    def _collect_violations(
        self, subject_bbox: BBox, best_bbox: BBox, current_subs: dict[str, float]
    ) -> list[RuleViolation]:
        """把当前画面的问题项转成结构化规则违反记录（NFR-O2）。"""
        out: list[RuleViolation] = []

        thirds = thirds_alignment(subject_bbox, self.cfg.thirds_tolerance)
        if thirds.horizontal == "off" and thirds.vertical == "off":
            out.append(
                RuleViolation(
                    rule=RuleName.THIRDS_ALIGNMENT,
                    severity=Severity.INFO,
                    detail="主体既不在三分点也不在中心",
                )
            )

        if current_subs.get("headroom", 1.0) < 0.35:
            out.append(
                RuleViolation(
                    rule=RuleName.MARGIN_RATIO,
                    severity=Severity.WARN,
                    detail="头顶留白不足或过大",
                )
            )

        if current_subs.get("balance", 1.0) < 0.30:
            out.append(
                RuleViolation(
                    rule=RuleName.SUBJECT_PLACEMENT,
                    severity=Severity.WARN,
                    detail="主体过于贴边，画面失衡",
                )
            )

        # 主体被候选框裁切
        if (
            subject_bbox[0] < best_bbox[0] - 0.01
            or subject_bbox[1] < best_bbox[1] - 0.01
            or subject_bbox[2] > best_bbox[2] + 0.01
            or subject_bbox[3] > best_bbox[3] + 0.01
        ):
            out.append(
                RuleViolation(
                    rule=RuleName.SUBJECT_CUTOFF,
                    severity=Severity.ERROR,
                    detail="主体超出建议画框，可能被裁切",
                )
            )
        return out


class SAMPNetScorer:
    """SAMPNet + CADB 构图评分模型接入桩（当前未启用）。

    预留原因与状态（诚实标注）：
        - 调研将 SAMPNet 列为**最贴合需求的模型**（"明确支持给出改进建议"）；
        - 但权重需从作者仓库单独获取，且本机离线环境未验证其人像域效果；
        - 因此当前**不接入**，仅保留接口以证明架构可替换（NFR-M1）。

    启用方式：实现 :meth:`score_frame` 并让工厂返回本类。
    权重就绪后应做的验证：
        1. 在 CADB 测试集上复现其报告的 SRCC；
        2. 在人像测试集上对比本类与 Heuristic 版本的排序一致性；
        3. 确认单帧推理耗时满足 NFR-P1。
    """

    name = "sampnet"

    def __init__(self, weights_path: str | None = None, device: str = "auto") -> None:
        self.weights_path = weights_path
        self.device = device
        self._available = False

    @property
    def available(self) -> bool:
        """权重是否就绪。当前恒为 False。"""
        return self._available

    def score_frame(self, subject_bbox, saliency=None, face_yaw=None, frame_shape=None) -> CompositionResult:
        raise NotImplementedError(
            "SAMPNet 评分后端尚未接入。"
            "当前请使用 HeuristicCompositionScorer；"
            "接入前需先获取 CADB/SAMPNet 权重并验证其人像域效果。"
        )
