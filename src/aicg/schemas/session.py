"""会话与接口请求/响应契约。

对应文档：《数据模型与接口.md》§3.2
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .perception import BBox


class SessionInfo(BaseModel):
    """拍摄会话。"""

    session_id: str
    created_at_ms: int
    frame_count: int = 0
    shot_count: int = 0
    language_calls: int = Field(default=0, description="语言层已调用次数（成本控制）")


class CalibrateRequest(BaseModel):
    """距离校准请求（API-03 / FR-01）。"""

    session_id: str | None = None
    image_ref: str = Field(description="图像引用：本地路径或 base64 data URI")


class SubjectOverrideRequest(BaseModel):
    """人工指定主体请求（FR-02 兜底路径）。"""

    session_id: str
    image_ref: str
    # 长按点选位置（归一化坐标，用户点在哪里）
    point: tuple[float, float] | None = Field(
        default=None, description="点选位置，归一化 [x, y]"
    )
    bbox: BBox | None = Field(default=None, description="或直接给出框，归一化")


class FrameRequest(BaseModel):
    """单帧引导推理请求（API-05 / 核心接口）。"""

    session_id: str
    frame_id: int
    timestamp_ms: int
    image_ref: str = Field(description="图像引用：本地路径或 base64 data URI")
    subject_override: BBox | None = Field(
        default=None, description="人工指定主体框（归一化），覆盖自动检测"
    )
    enable_language: bool = Field(
        default=False, description="是否触发语言层（成本控制，NFR-E3）"
    )
    persist_visual: bool = Field(
        default=False, description="是否输出该帧的标注可视化图"
    )


class ShotReportRequest(BaseModel):
    """拍后解说请求（API-07 / FR-07、FR-08）。"""

    session_id: str
    image_ref: str
    # 可选：传入该次拍摄的快照，用于溯源；不传则重新推理一次
    snapshot_frame_id: int | None = None


class CaseSearchRequest(BaseModel):
    """案例检索请求（API-08 / FR-09）。"""

    image_ref: str | None = Field(default=None, description="以图搜图")
    query_text: str | None = Field(default=None, description="以文搜图")
    top_k: int = Field(default=5, ge=1, le=50)
    pattern: str | None = Field(default=None, description="按构图模式过滤")


class CaseSearchResult(BaseModel):
    """单条案例检索结果。"""

    case_id: str
    image_ref: str
    similarity: float = Field(ge=0.0, le=1.0)
    pattern: str = "unknown"
    scene_tags: list[str] = Field(default_factory=list)
    description: str | None = None


class ErrorDetail(BaseModel):
    """统一错误响应体。"""

    code: str
    message: str
    detail: object | None = None


class ErrorResponse(BaseModel):
    """统一错误响应（《数据模型与接口.md》§3.3）。"""

    error: ErrorDetail
