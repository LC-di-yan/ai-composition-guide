"""数据契约测试。

对应文档：《数据模型与接口.md》§5 契约变更流程

**为什么契约要单独测**：``FrameSnapshot`` 是跨层、跨进程（API）流转的
核心结构。它的序列化稳定性直接决定前端能否解析。这里锁定三类性质：

1. **坐标系不变量**：所有 bbox 必须落在 0~1 且 x2>x1、y2>y1；
2. **序列化往返**：``model_dump`` → ``model_validate`` 必须无损；
3. **枚举穷尽**：新增动作/降级原因时必须有对应处理（防止静默漏分支）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aicg.schemas import (
    ActionCommand,
    ActionType,
    CalibrationResult,
    CompositionPattern,
    CompositionResult,
    DegradationReason,
    LatencyBreakdown,
    PerceptionResult,
    RuleName,
    RuleViolation,
    Severity,
    ShotReport,
    StabilizedCommand,
    Subject,
    SubjectSource,
    SuppressReason,
    TokenUsage,
)


def _subject(bbox=(0.3, 0.2, 0.6, 0.9)) -> Subject:
    return Subject(
        subject_id="s0", label="person", bbox=bbox, confidence=0.9,
        is_primary=True, source=SubjectSource.AUTO,
    )


def _perception() -> PerceptionResult:
    return PerceptionResult(
        frame_id=1, timestamp_ms=1000, frame_size=(480, 640),
        subjects=[_subject()], backend="stub",
    )


def _composition() -> CompositionResult:
    return CompositionResult(
        composition_score=72.5, best_score=88.0,
        best_bbox=(0.1, 0.05, 0.9, 0.95), current_bbox=(0.3, 0.2, 0.6, 0.9),
        sub_scores={"thirds": 0.8}, pattern=CompositionPattern.RULE_OF_THIRDS,
        pattern_label="右三分", shot_size_label="七分身",
    )


def _command() -> StabilizedCommand:
    raw = ActionCommand(action=ActionType.MOVE_CLOSER, magnitude_text="约 1.7 米", magnitude_raw=0.2)
    return StabilizedCommand(command=raw, raw_command=raw, is_changed=True)


# ---------------------------------------------------------------------------
# bbox 校验
# ---------------------------------------------------------------------------
class TestBBoxValidation:
    def test_valid_bbox_accepted(self):
        s = _subject((0.0, 0.0, 1.0, 1.0))
        assert s.bbox == (0.0, 0.0, 1.0, 1.0)

    @pytest.mark.parametrize(
        "bad",
        [
            (0.6, 0.2, 0.3, 0.9),   # x2 < x1
            (0.3, 0.9, 0.6, 0.2),   # y2 < y1
            (-0.1, 0.2, 0.6, 0.9),  # 越界（负）
            (0.3, 0.2, 1.2, 0.9),   # 越界（>1）
        ],
    )
    def test_invalid_bbox_rejected(self, bad):
        """非法 bbox 必须在构造时失败，不能流入下游。"""
        with pytest.raises(ValidationError):
            _subject(bad)

    def test_degenerate_zero_area_rejected(self):
        with pytest.raises(ValidationError):
            _subject((0.5, 0.5, 0.5, 0.5))


class TestSubjectDerivedProps:
    def test_geometry_properties(self):
        s = _subject((0.2, 0.1, 0.6, 0.9))
        assert s.width == pytest.approx(0.4)
        assert s.height == pytest.approx(0.8)
        assert s.area == pytest.approx(0.32)
        assert s.center == pytest.approx((0.4, 0.5))


# ---------------------------------------------------------------------------
# 序列化往返
# ---------------------------------------------------------------------------
class TestSerializationRoundTrip:
    def test_perception_roundtrip(self):
        p = _perception()
        assert PerceptionResult.model_validate(p.model_dump(mode="json")) == p

    def test_composition_roundtrip(self):
        c = _composition()
        assert CompositionResult.model_validate(c.model_dump(mode="json")) == c

    def test_command_roundtrip(self):
        c = _command()
        assert StabilizedCommand.model_validate(c.model_dump(mode="json")) == c

    def test_latency_roundtrip(self):
        lat = LatencyBreakdown(capture_ms=1.0, perception_ms=2.0, decision_ms=3.0, stabilization_ms=0.5)
        lat.recompute_total()
        assert LatencyBreakdown.model_validate(lat.model_dump(mode="json")).total_ms == 6.5

    def test_calibration_roundtrip(self):
        c = CalibrationResult(
            current_distance_m=2.1, min_distance_m=2.0, max_distance_m=5.0,
            position_ratio=0.3, is_in_range=True, advice_text="距离合适",
        )
        assert CalibrationResult.model_validate(c.model_dump(mode="json")) == c

    def test_shot_report_roundtrip(self):
        r = ShotReport(shot_id="s1", composition_narration="解说文本", filter_name="PROVIA")
        assert ShotReport.model_validate(r.model_dump(mode="json")) == r

    def test_no_numpy_types_in_payload(self):
        """序列化结果不得含 numpy 类型（JSON 无法编码）。"""
        import json

        payload = _composition().model_dump(mode="json")
        json.dumps(payload)  # 不抛异常即通过


# ---------------------------------------------------------------------------
# LatencyBreakdown 自洽
# ---------------------------------------------------------------------------
class TestLatencyBreakdown:
    def test_total_excludes_language(self):
        """语言层不在实时回路内，不计入 total（否则延迟指标会失真）。"""
        lat = LatencyBreakdown(
            capture_ms=1.0, perception_ms=2.0, decision_ms=3.0,
            stabilization_ms=1.0, language_ms=500.0,
        )
        lat.recompute_total()
        assert lat.total_ms == pytest.approx(7.0)
        assert lat.language_ms == 500.0

    def test_language_none_by_default(self):
        """未调用语言层时该字段应为 None，明确表达"未发生"。"""
        assert LatencyBreakdown().language_ms is None


# ---------------------------------------------------------------------------
# 枚举穷尽性
# ---------------------------------------------------------------------------
class TestEnumExhaustiveness:
    def test_action_type_complete(self):
        assert {a.value for a in ActionType} == {
            "move_closer", "move_back", "move_left", "move_right",
            "tilt_up", "tilt_down", "hold",
        }

    def test_suppress_reason_complete(self):
        assert {s.value for s in SuppressReason} == {"ema", "n_frame_consensus", "min_interval"}

    def test_degradation_reason_complete(self):
        """降级原因必须覆盖 NFR-R1/R2 定义的所有场景。"""
        vals = {d.value for d in DegradationReason}
        assert "subject_detection_failed" in vals
        assert "composition_model_unavailable" in vals
        assert "weights_missing" in vals
        assert "timeout" in vals

    def test_rule_and_severity(self):
        assert len(RuleName) == 4
        assert {s.value for s in Severity} == {"info", "warn", "error"}

    def test_pattern_values(self):
        assert len(CompositionPattern) == 6


# ---------------------------------------------------------------------------
# 边界与缺省
# ---------------------------------------------------------------------------
class TestDefaultsAndBoundaries:
    def test_primary_subject_none_when_empty(self):
        p = PerceptionResult(frame_id=0, timestamp_ms=0, frame_size=(1, 1), subjects=[])
        assert p.primary_subject is None

    def test_primary_subject_selected_by_flag(self):
        a = _subject((0.1, 0.1, 0.2, 0.2)).model_copy(update={"is_primary": False})
        b = _subject((0.5, 0.5, 0.9, 0.9))
        p = PerceptionResult(frame_id=0, timestamp_ms=0, frame_size=(1, 1), subjects=[a, b])
        assert p.primary_subject.bbox == b.bbox

    def test_token_usage_defaults_zero(self):
        t = TokenUsage()
        assert t.total_tokens == 0

    def test_composition_optional_fields(self):
        c = CompositionResult(composition_score=50.0)
        assert c.best_bbox is None
        assert c.shot_size_label == ""
        assert c.rule_violations == []

    def test_rule_violation_construction(self):
        v = RuleViolation(rule=RuleName.MARGIN_RATIO, severity=Severity.WARN, detail="头顶留白不足")
        assert v.severity is Severity.WARN
