"""距离校准流程（FR-01）。

对应需求：FR-01（距离校准）
对应文档：《数据模型与接口.md》§3.2 ``POST /v1/session/{id}/calibrate``

**流程语义**：用户进入拍摄模式后，系统先做一次"距离体检"——
告诉用户"现在约 2.1 米，落在建议区间内/外"。这一步之所以独立于逐帧
引导，是因为距离建议**变化慢**（人不会每帧移动半米），独立成一次
调用可以避免把噪声抖进滑条。

**诚实标注**：本流程输出的是**估算**而非测量，误差 ±25%。因此接口
返回 ``error_margin``，UI 必须如实呈现"约"字与误差范围，不能伪装成
精确测距。这是本模块最重要的产品约束。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..composition.distance import DistanceEstimator
from ..observability import get_logger
from ..perception.base import BasePerception
from ..schemas import CalibrationResult
from ..settings import Settings

log = get_logger("pipeline.calibration")


def _now_ms() -> int:
    """当前 Unix 毫秒时间戳。"""
    return int(time.time() * 1000)


@dataclass
class CalibrationOutcome:
    """一次校准的完整产出。"""

    result: CalibrationResult
    subject_bbox: tuple[float, float, float, float] | None = None
    subject_label: str | None = None
    backend: str = "unknown"
    """实际使用的感知后端，保证可复现（NFR-O3）。"""
    degraded: bool = False


class CalibrationPipeline:
    """距离校准流程：取帧 → 感知主体 → 估算距离。"""

    def __init__(
        self,
        perception: BasePerception,
        settings: Settings,
        estimator: DistanceEstimator | None = None,
    ) -> None:
        self.perception = perception
        self.settings = settings
        self.estimator = estimator or DistanceEstimator(settings.composition.distance)

    # ------------------------------------------------------------------
    def calibrate(
        self,
        image: np.ndarray,
        *,
        frame_id: int = 0,
        assumed_height_m: float | None = None,
        nominal_focal_mm: float | None = None,
        subject_bbox: tuple[float, float, float, float] | None = None,
    ) -> CalibrationOutcome:
        """执行一次距离校准。

        Args:
            image: 输入帧（BGR）。
            frame_id: 帧序号。
            assumed_height_m: 主体真实身高假设；None 用配置默认。
            nominal_focal_mm: 等效焦距；None 用配置默认。
            subject_bbox: 已知主体框（FR-02 手动指定时传入）。提供时
                跳过感知，直接估算。

        Returns:
            校准产出。**保证返回**——无主体时返回降级结果而非异常。
        """
        backend = self.perception.name
        label: str | None = None

        if subject_bbox is None:
            try:
                perception = self.perception.infer(image, frame_id, _now_ms())
                subject = perception.primary_subject
                if subject is None:
                    log.info("校准未找到主体，返回降级结果")
                    return CalibrationOutcome(
                        result=self.estimator.estimate(None),
                        backend=backend,
                        degraded=True,
                    )
                subject_bbox = subject.bbox
                label = subject.label
            except Exception as e:  # noqa: BLE001
                log.warning("校准感知失败，返回降级结果: %s", e)
                return CalibrationOutcome(
                    result=self.estimator.estimate(None),
                    backend=backend,
                    degraded=True,
                )

        result = self.estimator.estimate(
            subject_bbox,
            assumed_height_m=assumed_height_m,
            nominal_focal_mm=nominal_focal_mm,
        )
        return CalibrationOutcome(
            result=result,
            subject_bbox=subject_bbox,
            subject_label=label,
            backend=backend,
            degraded=result.degraded,
        )
