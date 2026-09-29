"""主体确认流程（FR-02）。

对应需求：FR-02（主体确认）
对应文档：《数据模型与接口.md》§3.3 ``POST /v1/session/{id}/subject``

**为什么需要有这一步**：自动主体选择在多主体场景（合影、街拍带路人）
下必然出错。若不给用户"指定谁"的能力，Agent 会持续给出针对错误对象的
建议——用户会觉得"这 AI 傻"。因此 FR-02 是**产品的可信度闸门**。

**实现策略**：主体选择逻辑（排序规则）在 ``perception.base.SubjectSelector``，
本模块只负责流程编排与状态访问：

1. 自动模式：跑感知 → 选择器挑主主体；
2. 手动模式：用户给出框（或从候选里点选）→ 直接采用，置信度置 1.0。

**诚实标注**：M3 阶段的手动指定仅支持"传入 bbox"，**不包含**点击选中
的交互（那属于前端职责）。接口层负责把点击坐标转成 bbox。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..observability import get_logger
from ..perception.base import BasePerception
from ..schemas import Subject, SubjectSource
from ..settings import Settings
from ..utils.image import bbox_iou

log = get_logger("pipeline.subject")


def _now_ms() -> int:
    """当前 Unix 毫秒时间戳。"""
    return int(time.time() * 1000)


@dataclass
class SubjectOutcome:
    """主体确认产出。"""

    subjects: list[Subject] = field(default_factory=list)
    primary: Subject | None = None
    needs_user_input: bool = False
    """是否建议用户手动指定（多主体且置信度接近时置 True）。"""
    backend: str = "unknown"
    degraded: bool = False


class SubjectPipeline:
    """主体确认流程。"""

    # 当 top-1 与 top-2 的"可选性分值"过于接近时，主动请求用户确认
    _AMBIGUITY_GAP = 0.12

    def __init__(self, perception: BasePerception, settings: Settings) -> None:
        self.perception = perception
        self.settings = settings

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray, *, frame_id: int = 0) -> SubjectOutcome:
        """自动检测主体，必要时标记"建议人工确认"。

        Returns:
            主体确认产出。**保证返回**（无主体时 ``primary=None``）。
        """
        try:
            result = self.perception.infer(image, frame_id, _now_ms())
        except Exception as e:  # noqa: BLE001
            log.warning("主体检测失败: %s", e)
            return SubjectOutcome(subjects=[], primary=None, backend=self.perception.name, degraded=True)

        subjects = list(result.subjects)
        if not subjects:
            return SubjectOutcome(
                subjects=[], primary=None, backend=result.backend, degraded=True
            )

        primary = result.primary_subject
        needs_input = self._is_ambiguous(subjects)
        return SubjectOutcome(
            subjects=subjects,
            primary=primary,
            needs_user_input=needs_input,
            backend=result.backend,
            degraded=result.degraded,
        )

    def override(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        *,
        frame_id: int = 0,
        current_subjects: list[Subject] | None = None,
    ) -> SubjectOutcome:
        """用户手动指定主体（FR-02）。

        Args:
            image: 当前帧（用于在未提供候选时重新检测，以便保留其他主体）。
            bbox: 用户指定框（归一化）。
            frame_id: 帧序号。
            current_subjects: 已知的主体列表，避免重复检测。

        Returns:
            以手动主体为 primary 的产出。
        """
        subjects = list(current_subjects or [])
        if not subjects:
            try:
                subjects = list(self.perception.infer(image, frame_id, _now_ms()).subjects)
            except Exception as e:  # noqa: BLE001
                log.warning("手动指定时重新检测失败: %s", e)
                subjects = []

        # 若用户框与已有主体高度重叠，则"接管"该主体而非新增
        merged: list[Subject] = []
        claimed = False
        for s in subjects:
            if not claimed and bbox_iou(s.bbox, bbox) > 0.5:
                merged.append(
                    s.model_copy(update={"bbox": bbox, "is_primary": True, "source": SubjectSource.MANUAL})
                )
                claimed = True
            else:
                merged.append(s.model_copy(update={"is_primary": False}))

        if not claimed:
            merged.insert(
                0,
                Subject(
                    subject_id="manual-0",
                    label="subject(manual)",
                    bbox=bbox,
                    confidence=1.0,
                    is_primary=True,
                    source=SubjectSource.MANUAL,
                ),
            )

        return SubjectOutcome(
            subjects=merged,
            primary=next((s for s in merged if s.is_primary), merged[0] if merged else None),
            needs_user_input=False,
            backend=self.perception.name,
            degraded=False,
        )

    # ------------------------------------------------------------------
    def _is_ambiguous(self, subjects: list[Subject]) -> bool:
        """判断主体选择是否含糊：存在多个同类别、面积接近的候选。

        理由：合影场景下"谁是主角"是**语义问题**，不是算法能可靠解决的。
        与其猜错，不如早问——这比事后被用户投诉"选错人"成本低得多。
        """
        persons = [s for s in subjects if s.label == "person"]
        if len(persons) < 2:
            return False
        persons.sort(key=lambda s: s.area, reverse=True)
        top1, top2 = persons[0], persons[1]
        if top1.area <= 0:
            return False
        gap = (top1.area - top2.area) / top1.area
        return gap < self._AMBIGUITY_GAP
