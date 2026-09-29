"""构图决策层数据契约。

对应需求：FR-01（距离校准）、FR-03（构图评估）、FR-04（动作指令）
对应文档：《数据模型与接口.md》§2.2 / §2.3 / §2.4
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .perception import BBox

# 三分点在归一化坐标下的位置
THIRDS_X: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
THIRDS_Y: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)


# --------------------------------------------------------------------------
# FR-01 距离校准
# --------------------------------------------------------------------------
class EstimationMethod(str, Enum):
    """距离估算手段，决定误差量级。"""

    BBOX_HEIGHT_RATIO = "bbox_height_ratio"
    """框高占比反推，误差约 ±20%~30%（当期实现）。"""

    DEPTH_MAP = "depth_map"
    """深度图估算（未实现，属生产环境增强项）。"""


class CalibrationResult(BaseModel):
    """距离校准结果（FR-01）。"""

    current_distance_m: float | None = Field(
        default=None, description="当前估距（m）；无法估算时为 None"
    )
    min_distance_m: float = Field(description="建议最近距离（m）")
    max_distance_m: float = Field(description="建议最远距离（m）")
    position_ratio: float = Field(
        ge=0.0, le=1.0, description="当前距离在区间内的位置 0~1，用于滑条可视化"
    )
    is_in_range: bool = Field(description="是否落在建议区间内")
    advice_text: str = Field(description="面向用户的提示文案")
    estimation_method: EstimationMethod = Field(
        default=EstimationMethod.BBOX_HEIGHT_RATIO
    )
    error_margin: float | None = Field(
        default=None, description="估算误差范围（±m 或 ±比例），当前实现未标定"
    )
    degraded: bool = Field(default=False, description="是否因无法估算而降级")

    @field_validator("max_distance_m")
    @classmethod
    def _check_range(cls, v: float, info) -> float:
        mn = info.data.get("min_distance_m")
        if mn is not None and v <= mn:
            raise ValueError(f"max_distance_m({v}) 必须大于 min_distance_m({mn})")
        return v


# --------------------------------------------------------------------------
# FR-03 构图评估
# --------------------------------------------------------------------------
class CompositionPattern(str, Enum):
    """构图模式分类结果（FR-03）。"""

    RULE_OF_THIRDS = "rule_of_thirds"
    CENTER = "center"
    SYMMETRIC = "symmetric"
    DIAGONAL = "diagonal"
    FRAMING = "framing"
    UNKNOWN = "unknown"


class RuleName(str, Enum):
    """规则校验项名称。"""

    THIRDS_ALIGNMENT = "thirds_alignment"
    MARGIN_RATIO = "margin_ratio"
    SUBJECT_PLACEMENT = "subject_placement"
    SUBJECT_CUTOFF = "subject_cutoff"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class RuleViolation(BaseModel):
    """单条规则校验未通过项。"""

    rule: RuleName
    severity: Severity
    detail: str = Field(description="人可读说明")


class CandidateScore(BaseModel):
    """候选画框及其评分明细，是"建议可溯源"（NFR-O2）的载体。"""

    bbox: BBox
    score: float = Field(description="总分")
    sub_scores: dict[str, float] = Field(
        default_factory=dict, description="各子项得分，键为子项名"
    )


class CompositionResult(BaseModel):
    """构图评估结果（FR-03）。"""

    composition_score: float = Field(
        description="构图总分，取值域 0~100（越高越好）"
    )
    sub_scores: dict[str, float] = Field(
        default_factory=dict, description="子属性分"
    )
    pattern: CompositionPattern = Field(default=CompositionPattern.UNKNOWN)
    pattern_label: str = Field(
        default="", description="构图模式中文术语，如「右三分」。供语言层直接引用"
    )
    shot_size_label: str = Field(
        default="", description="景别中文术语，如「七分身」。由主体占比推定"
    )
    best_bbox: BBox | None = Field(default=None, description="候选搜索得到的最优画框")
    best_score: float | None = Field(default=None, description="最优框对应分数")
    candidate_count: int = Field(default=0, description="评估过的候选框数量")
    rule_violations: list[RuleViolation] = Field(default_factory=list)
    decision_ms: float = Field(default=0.0, description="本层耗时（ms）")
    degraded: bool = Field(default=False)
    top_candidates: list[CandidateScore] = Field(
        default_factory=list, description="Top-K 候选，供「切换构图方案」使用"
    )


# --------------------------------------------------------------------------
# FR-04 动作指令
# --------------------------------------------------------------------------
class ActionType(str, Enum):
    """动作类型（FR-04）。"""

    MOVE_CLOSER = "move_closer"
    MOVE_BACK = "move_back"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    HOLD = "hold"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionCommand(BaseModel):
    """动作指令（FR-04）。"""

    action: ActionType
    magnitude_text: str = Field(description='差分量文本，如"约 1.7 米"')
    magnitude_raw: float | None = Field(
        default=None, description="差分原始数值（归一化坐标系下的位移量）"
    )
    urgency: Urgency = Field(default=Urgency.LOW)
    is_hold: bool = Field(default=False, description="是否为「保持不动」")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    can_skip: bool = Field(
        default=True, description="是否允许「就这样构图」跳过（FR-12）"
    )


# --------------------------------------------------------------------------
# FR-06 防抖后指令
# --------------------------------------------------------------------------
class SuppressReason(str, Enum):
    """指令被抑制（未切换）的原因，用于指标归因。"""

    EMA = "ema"
    N_FRAME_CONSENSUS = "n_frame_consensus"
    MIN_INTERVAL = "min_interval"


class StabilizedCommand(BaseModel):
    """防抖后的最终指令（FR-06）。"""

    command: ActionCommand = Field(description="防抖后实际下发的指令")
    raw_command: ActionCommand = Field(description="防抖前的原始指令")
    is_changed: bool = Field(description="本次是否发生切换")
    suppressed_by: SuppressReason | None = Field(
        default=None, description="被抑制的原因；发生切换时为 None"
    )
    consecutive_frames: int = Field(
        default=1, description="当前候选指令已连续命中的帧数"
    )
    smoothing: float | None = Field(
        default=None, description="EMA 平滑后的差分原始值"
    )
    since_last_change_ms: float = Field(
        default=0.0, description="距上次切换的毫秒数"
    )
