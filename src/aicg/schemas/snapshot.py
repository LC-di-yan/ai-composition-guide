"""核心契约：FrameSnapshot（结构化中间表示）与拍后报告。

对应需求：NFR-O1（结构化中间表示）、NFR-O2（建议可溯源）、NFR-P1（延迟）
对应文档：《数据模型与接口.md》§2.6 / §2.7

``FrameSnapshot`` 是**全项目最核心的契约**：它串联感知、决策、防抖三层的
输出，并携带分阶段耗时，是"每一句建议都能追溯到具体模型输出字段"的载体。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .composition import (
    CalibrationResult,
    CompositionPattern,
    CompositionResult,
    StabilizedCommand,
)
from .perception import BBox, PerceptionResult

# 契约版本号。任何字段增删改都必须同步递增并记入 CHANGELOG。
SCHEMA_VERSION = "0.1"


class LatencyBreakdown(BaseModel):
    """分阶段耗时（NFR-P1 / NFR-O4）。"""

    capture_ms: float = Field(default=0.0, description="取帧耗时")
    perception_ms: float = Field(default=0.0, description="感知耗时")
    decision_ms: float = Field(default=0.0, description="决策耗时")
    stabilization_ms: float = Field(default=0.0, description="防抖耗时")
    language_ms: float | None = Field(
        default=None, description="语言层耗时；非每帧调用，未调用时为 None"
    )
    total_ms: float = Field(default=0.0, description="端到端总耗时")

    def recompute_total(self) -> "LatencyBreakdown":
        """按各阶段重新汇总 total_ms。"""
        self.total_ms = (
            self.capture_ms
            + self.perception_ms
            + self.decision_ms
            + self.stabilization_ms
        )
        return self


class DegradationReason(str, Enum):
    """降级原因（NFR-R1 / NFR-R2），接口以 200 返回但字段标记降级。"""

    SUBJECT_DETECTION_FAILED = "subject_detection_failed"
    """主体检测失败 → 提示长按指定主体。"""

    COMPOSITION_MODEL_UNAVAILABLE = "composition_model_unavailable"
    """构图模型不可用 → 降级为通用三分法引导。"""

    DISTANCE_ESTIMATION_FAILED = "distance_estimation_failed"
    """距离无法估算 → 仅给方向不给距离。"""

    WEIGHTS_MISSING = "weights_missing"
    """模型权重缺失 → 感知层降级为规则版。"""

    TIMEOUT = "timeout"
    """单帧推理超时。"""

    UNKNOWN = "unknown"


class FrameSnapshot(BaseModel):
    """单帧的完整结构化中间表示 —— 本项目的核心契约。"""

    schema_version: str = Field(default=SCHEMA_VERSION, description="契约版本号")
    frame_id: int
    timestamp_ms: int
    frame_size: tuple[int, int] = Field(description="[宽, 高]，单位 px")

    perception: PerceptionResult
    composition: CompositionResult
    command: StabilizedCommand
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

    calibration: CalibrationResult | None = Field(
        default=None, description="距离校准结果；进入引导后按需附带"
    )

    degraded: bool = Field(default=False, description="本次是否走了降级路径")
    degradation_reason: DegradationReason | None = Field(default=None)

    def overview(self) -> str:
        """单行摘要，用于 CLI 与日志输出。"""
        subj = self.perception.primary_subject
        subj_txt = f"{subj.label}({subj.confidence:.2f})" if subj else "无主体"
        return (
            f"#{self.frame_id:<4d} "
            f"主体={subj_txt:<16s} "
            f"构图={self.composition.composition_score:5.1f} "
            f"模式={self.composition.pattern.value:<14s} "
            f"指令={self.command.command.action.value:<12s} "
            f"{self.command.command.magnitude_text:<12s} "
            f"用时={self.latency.total_ms:6.1f}ms"
            + ("  [降级]" if self.degraded else "")
        )


class TokenUsage(BaseModel):
    """VLM Token 消耗，用于成本指标（NFR-E3）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ShotReport(BaseModel):
    """拍后解说报告（FR-07 / FR-08）。"""

    shot_id: str
    composition_narration: str = Field(description="构图理念解说")
    filter_name: str | None = Field(default=None, description="推荐滤镜名")
    filter_reason: str | None = Field(default=None, description="滤镜推荐理由")
    input_snapshot: FrameSnapshot | None = Field(
        default=None, description="生成解说所依据的快照，保证可溯源（NFR-O2）"
    )
    prompt_version: str = Field(default="v1")
    vlm_model: str = Field(default="mock", description="所用 VLM 名称；mock 表示模板兜底")
    is_fallback: bool = Field(
        default=False, description="是否走了模板兜底（无 API Key 或调用失败）"
    )
    token_usage: TokenUsage | None = None
    tts_audio_ref: str | None = None


__all__ = [
    "SCHEMA_VERSION",
    "BBox",
    "CompositionPattern",
    "DegradationReason",
    "FrameSnapshot",
    "LatencyBreakdown",
    "ShotReport",
    "TokenUsage",
]
