"""感知层数据契约。

对应需求：FR-02（主体确认）
对应文档：《数据模型与接口.md》§2.1

坐标系约定（原文档标记为待确认，此处明确）：
    bbox 统一采用 **归一化坐标**，格式 ``[x1, y1, x2, y2]``，取值 0.0~1.0，
    其中 x 相对图像宽度、y 相对图像高度。
    选择归一化坐标的原因：候选框搜索、距离估算、规则校验全部在比例空间
    进行，与具体分辨率解耦；仅在可视化时乘回像素尺寸。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# 归一化 bbox：[x1, y1, x2, y2]，取值 0.0~1.0
BBox = tuple[float, float, float, float]


class SubjectSource(str, Enum):
    """主体来源，对应 FR-02 的人工兜底设计。"""

    AUTO = "auto"
    """自动检测选中。"""

    MANUAL = "manual"
    """人工长按指定。"""


class Subject(BaseModel):
    """画面中的候选拍摄主体。"""

    subject_id: str = Field(description="帧内唯一标识")
    label: str = Field(description="类别名，如 person / dog")
    bbox: BBox = Field(description="主体框，归一化 [x1,y1,x2,y2]")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0~1")
    is_primary: bool = Field(default=False, description="是否为当前选定主体")
    source: SubjectSource = Field(
        default=SubjectSource.AUTO, description="主体来源：自动 / 人工指定"
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: BBox) -> BBox:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox 非法，要求 x2>x1 且 y2>y1，实际为 {v}")
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError(f"bbox 必须为归一化坐标（0~1），实际为 {v}")
        return v

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class FaceInfo(BaseModel):
    """人脸与朝向信息，用于判断正脸、构图与景别。"""

    bbox: BBox = Field(description="人脸框，归一化")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    yaw_deg: float | None = Field(default=None, description="偏航角，正值为面向画面右侧")
    pitch_deg: float | None = Field(default=None, description="俯仰角，正值为抬头")
    landmarks_ref: str | None = Field(default=None, description="关键点存储引用（不内联数组）")


class PerceptionResult(BaseModel):
    """感知层单帧输出。"""

    frame_id: int = Field(description="帧序号")
    timestamp_ms: int = Field(description="帧时间戳（Unix ms）")
    frame_size: tuple[int, int] = Field(description="[宽, 高]，单位 px")
    subjects: list[Subject] = Field(default_factory=list, description="检测到的主体，空列表表示未检测到")
    faces: list[FaceInfo] = Field(default_factory=list)
    saliency_map_ref: str | None = Field(default=None, description="显著性图引用")
    depth_map_ref: str | None = Field(default=None, description="深度图引用（可选能力）")
    perception_ms: float = Field(default=0.0, description="本层耗时（ms）")
    backend: str = Field(default="rule", description="实际生效的感知后端：yolo / rule")
    degraded: bool = Field(default=False, description="是否走了降级路径")
    model_versions: dict[str, str] = Field(default_factory=dict, description="各模型版本，用于可复现")
    extras: dict[str, Any] = Field(default_factory=dict, description="后端特有的附加中间结果")

    @property
    def primary_subject(self) -> Subject | None:
        """返回当前选定主体；无主体时返回 None。"""
        for s in self.subjects:
            if s.is_primary:
                return s
        return None
