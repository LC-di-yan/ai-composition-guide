"""单帧处理单元 —— 把五层粘合成一个 ``FrameSnapshot``。

对应需求：FR-05（实时引导循环）、NFR-O1（结构化中间表示）、NFR-P1（延迟）
对应文档：《数据模型与接口.md》§2.6

职责边界（严格）::

    取帧 → 感知 → 构图决策 → 差分量 → 防抖 → 组装快照

本模块**只做编排与计时**，不做任何算法。所有算法都在各自层内。
这样做的收益：

1. **可替换**：换感知后端、换评分器都不需要动流水线；
2. **可测量**：每个阶段独立计时，延迟瓶颈一目了然（NFR-O4）；
3. **可测试**：用桩件替换任意一层即可做契约测试。

**降级是常态而非异常**：任何一层失败都返回带 ``degraded=True`` 的结果，
而不是抛异常。理由：实时引导场景下"崩掉"比"给个差建议"糟糕得多
（NFR-R1 / NFR-R2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..camera.base import Frame
from ..composition.candidate import CandidateGenerator
from ..composition.differ import CommandDiffer
from ..composition.rules import describe_pattern, describe_shot_size
from ..composition.scorer import HeuristicCompositionScorer
from ..observability import StageTimer, get_logger
from ..perception.base import BasePerception
from ..schemas import (
    ActionCommand,
    ActionType,
    CompositionPattern,
    CompositionResult,
    DegradationReason,
    FrameSnapshot,
    LatencyBreakdown,
    PerceptionResult,
    RuleViolation,
    Subject,
    SubjectSource,
    Urgency,
)
from ..settings import Settings
from ..stabilization import CommandDebouncer

log = get_logger("pipeline.frame")


@dataclass
class FrameContext:
    """逐帧处理的会话级状态容器。

    与 :class:`FrameProcessor` 分离，是为了让"无状态的处理逻辑"与
    "有状态的会话上下文"边界清晰——同一处理器可在多个并发会话中复用。
    """

    session_id: str = "default"
    calibration_distance_m: float | None = None
    """FR-01 距离校准结果；存在时会给指令附加绝对米数。"""

    subject_override_bbox: tuple[float, float, float, float] | None = None
    """FR-02 用户手动指定的主体框；存在时优先于自动检测。"""

    frame_counter: int = 0
    snapshots: list[FrameSnapshot] = field(default_factory=list)


class FrameProcessor:
    """单帧处理器：无状态的五层编排。

    注意：本类的所有依赖都通过构造注入，因此可以直接用桩件替换任一层
    做契约测试（见 ``tests/test_pipeline.py``）。
    """

    def __init__(
        self,
        perception: BasePerception,
        settings: Settings,
        *,
        generator: CandidateGenerator | None = None,
        scorer: HeuristicCompositionScorer | None = None,
        differ: CommandDiffer | None = None,
        debouncer: CommandDebouncer | None = None,
    ) -> None:
        self.perception = perception
        self.settings = settings
        comp = settings.composition
        self.generator = generator or CandidateGenerator(comp.candidate)
        self.scorer = scorer or HeuristicCompositionScorer(comp.scoring)
        # [D-10 修复 2026-09-29] 差分层的目标占比与容差**必须**与评分层同源：
        # 二者都在回答"理想的构图长什么样"，如果各存一份字面量，就会再次
        # 分裂出"评分器认为 0.55 最好、差分层却按 0.68 判断距离"的矛盾。
        # 因此这里显式从 comp.scoring 派生，而不是让 differ 用模块常量兜底。
        self.differ = differ or CommandDiffer(
            assumed_person_height_m=comp.distance.assumed_person_height_m,
            focal_constant=comp.distance.focal_constant,
            sensor_height_mm=comp.distance.sensor_height_mm,
            target_occupancy=comp.scoring.ideal_subject_height,
            occupancy_tolerance=comp.scoring.derived_occupancy_tolerance,
        )
        self.debouncer = debouncer or CommandDebouncer(settings.stabilization)

    # ------------------------------------------------------------------
    def warmup(self, image_shape: tuple[int, int] | None = None) -> None:
        """预热感知模型，避免首帧延迟尖刺污染指标（NFR-P1）。

        Args:
            image_shape: 实际推理尺寸 ``(高, 宽)``。引导循环内部尺寸一致，
                可不传；**单帧场景（CLI/单图打分）应传真实尺寸**，
                否则推理引擎会因尺寸变化重做一次规划（实测差 7 倍）。
        """
        try:
            self.perception.warmup(image_shape)
        except Exception as e:  # noqa: BLE001
            log.warning("感知预热失败（不影响运行）: %s", e)

    def reset(self) -> None:
        """重置会话级状态。"""
        self.debouncer.reset()

    # ------------------------------------------------------------------
    def process(self, frame: Frame, ctx: FrameContext) -> FrameSnapshot:
        """处理单帧，返回核心契约对象。

        Args:
            frame: 取帧层产出的帧。
            ctx: 会话上下文（携带校准值、主体覆盖等）。

        Returns:
            ``FrameSnapshot``。**保证返回**，任何阶段失败都走降级路径。
        """
        ctx.frame_counter += 1
        degraded = False
        reason: DegradationReason | None = None

        timer = StageTimer()
        image = frame.image

        # ---------- 第 1 层：感知（带降级）----------
        with timer.stage("perception"):
            perception = self._run_perception(image, frame, ctx)
        if perception.degraded:
            degraded = True
            reason = (
                DegradationReason.SUBJECT_DETECTION_FAILED
                if perception.primary_subject is None
                else DegradationReason.WEIGHTS_MISSING
            )

        # ---------- 第 2 层：构图决策 ----------
        with timer.stage("decision"):
            composition, raw_command = self._decide(image, perception)
        if composition.degraded and reason is None:
            degraded = True
            reason = DegradationReason.COMPOSITION_MODEL_UNAVAILABLE

        # ---------- 第 3 层：防抖 ----------
        with timer.stage("stabilization"):
            stabilized = self.debouncer.stabilize(raw_command, frame.timestamp_ms)

        # ---------- 组装快照 ----------
        latency: LatencyBreakdown = timer.to_breakdown()
        latency.capture_ms = float(frame.capture_ms)
        latency.recompute_total()

        snapshot = FrameSnapshot(
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            frame_size=frame.size,
            perception=perception,
            composition=composition,
            command=stabilized,
            latency=latency,
            degraded=degraded,
            degradation_reason=reason,
        )
        ctx.snapshots.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    def _run_perception(self, image: np.ndarray, frame: Frame, ctx: FrameContext) -> PerceptionResult:
        """执行感知，失败时返回降级结果。"""
        try:
            result = self.perception.infer(image, frame.frame_id, frame.timestamp_ms)
        except Exception as e:  # noqa: BLE001 - 感知失败不得中断回路
            log.error("感知阶段异常，返回降级结果: %s", e)
            return PerceptionResult(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                frame_size=frame.size,
                subjects=[],
                backend=self.perception.name,
                degraded=True,
            )

        # FR-02：手动主体覆盖优先
        if ctx.subject_override_bbox is not None:
            result = self._apply_subject_override(result, ctx.subject_override_bbox)
        return result

    def _apply_subject_override(self, result: PerceptionResult, bbox) -> PerceptionResult:
        """把用户手动圈选的主体注入感知结果（FR-02）。"""
        manual = Subject(
            subject_id="manual-0",
            label="subject(manual)",
            bbox=tuple(bbox),  # type: ignore[arg-type]
            confidence=1.0,
            is_primary=True,
            source=SubjectSource.MANUAL,
        )
        others = [s.model_copy(update={"is_primary": False}) for s in result.subjects]
        return result.model_copy(update={"subjects": [manual, *others], "degraded": False})

    # ------------------------------------------------------------------
    def _decide(
        self, image: np.ndarray, perception: PerceptionResult
    ) -> tuple[CompositionResult, ActionCommand]:
        """构图决策：生成候选 → 评分 → 差分量。

        Returns:
            ``(CompositionResult, ActionCommand)``。无主体时返回降级的
            "保持"指令，绝不抛异常。
        """
        subject = perception.primary_subject
        if subject is None:
            return self._empty_composition(perception, "未识别到可靠主体"), _hold_command()

        # 复用评分器的完整评估流程（内部已完成候选生成 + 评分 + 最优选择）
        saliency = perception.extras.get("saliency_map")
        if saliency is not None and not isinstance(saliency, np.ndarray):
            saliency = None
        h, w = image.shape[:2]

        composition = self.scorer.score_frame(
            subject_bbox=subject.bbox,
            saliency=saliency,
            face_yaw=perception.faces[0].yaw_deg if perception.faces else None,
            frame_shape=(h, w),
        )

        best_bbox = composition.best_bbox
        if best_bbox is None:
            return composition, _hold_command()

        command = self.differ.to_command(
            current_bbox=subject.bbox,
            best_bbox=best_bbox,
            confidence=float(composition.best_score or 0.0) / 100.0,
        )
        return composition, command

    # ------------------------------------------------------------------
    def _empty_composition(self, perception: PerceptionResult, note: str) -> CompositionResult:
        """构造降级的空构图结果（NFR-R2：无主体时给通用引导）。"""
        subj_bbox = (
            perception.primary_subject.bbox if perception.primary_subject else (0.34, 0.30, 0.66, 0.86)
        )
        default_bbox = (0.34, 0.30, 0.66, 0.86)
        return CompositionResult(
            composition_score=0.0,
            sub_scores={},
            pattern=CompositionPattern.UNKNOWN,
            pattern_label=describe_pattern("unknown", default_bbox),
            shot_size_label=describe_shot_size(default_bbox),
            best_bbox=default_bbox,
            best_score=0.0,
            candidate_count=0,
            rule_violations=[
                RuleViolation(
                    rule="thirds_alignment",  # type: ignore[arg-type]
                    severity="info",  # type: ignore[arg-type]
                    detail=note,
                )
            ],
            degraded=True,
        )


def _hold_command() -> ActionCommand:
    """降级用「保持」指令。"""
    return ActionCommand(
        action=ActionType.HOLD,
        magnitude_text="保持",
        magnitude_raw=0.0,
        urgency=Urgency.LOW,
        is_hold=True,
        confidence=0.0,
        can_skip=True,
    )
