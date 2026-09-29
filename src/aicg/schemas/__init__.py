"""数据契约包。

所有层间数据交换**只能**通过本包定义的类型（《数据模型与接口.md》§1 契约总原则）。

契约版本：``SCHEMA_VERSION``，任何字段增删改都必须同步递增并记入 CHANGELOG。
"""

from .composition import (
    ActionCommand,
    ActionType,
    CalibrationResult,
    CandidateScore,
    CompositionPattern,
    CompositionResult,
    EstimationMethod,
    RuleName,
    RuleViolation,
    Severity,
    StabilizedCommand,
    SuppressReason,
    Urgency,
)
from .perception import (
    BBox,
    FaceInfo,
    PerceptionResult,
    Subject,
    SubjectSource,
)
from .session import (
    CalibrateRequest,
    CaseSearchRequest,
    CaseSearchResult,
    ErrorDetail,
    ErrorResponse,
    FrameRequest,
    SessionInfo,
    ShotReportRequest,
    SubjectOverrideRequest,
)
from .snapshot import (
    SCHEMA_VERSION,
    DegradationReason,
    FrameSnapshot,
    LatencyBreakdown,
    ShotReport,
    TokenUsage,
)

__all__ = [
    # perception
    "BBox",
    "FaceInfo",
    "PerceptionResult",
    "Subject",
    "SubjectSource",
    # composition
    "ActionCommand",
    "ActionType",
    "CalibrationResult",
    "CandidateScore",
    "CompositionPattern",
    "CompositionResult",
    "EstimationMethod",
    "RuleName",
    "RuleViolation",
    "Severity",
    "StabilizedCommand",
    "SuppressReason",
    "Urgency",
    # snapshot
    "SCHEMA_VERSION",
    "DegradationReason",
    "FrameSnapshot",
    "LatencyBreakdown",
    "ShotReport",
    "TokenUsage",
    # session
    "CalibrateRequest",
    "CaseSearchRequest",
    "CaseSearchResult",
    "ErrorDetail",
    "ErrorResponse",
    "FrameRequest",
    "SessionInfo",
    "ShotReportRequest",
    "SubjectOverrideRequest",
]
