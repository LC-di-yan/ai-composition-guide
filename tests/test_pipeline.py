"""端到端流水线集成测试（FR-05）与降级路径测试。

对应文档：《测试与验收.md》§3.2 集成测试、§5 降级与鲁棒性

**本文件的测试目标**：验证"引导回路在单帧失败时不会崩、会如实标降级"。
实时引导场景下，"崩掉"比"给个差建议"糟糕得多（NFR-R1 / NFR-R2），
因此这里大量使用**故障注入**（抛异常的桩件）来验证降级契约。
"""

from __future__ import annotations

import numpy as np
import pytest

from aicg.camera.base import Frame, FrameSource
from aicg.perception.base import BasePerception
from aicg.pipeline import FrameContext, FrameProcessor, GuidingLoop
from aicg.pipeline.guiding_loop import _ListSource
from aicg.schemas import (
    ActionType,
    FrameSnapshot,
    PerceptionResult,
    Subject,
    SubjectSource,
)


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class StubPerception(BasePerception):
    """固定输出的感知桩件：围绕给定框生成一个 person 主体。"""

    name = "stub"

    def __init__(self, bbox=(0.35, 0.15, 0.65, 0.90), confidence: float = 0.9):
        self._bbox = bbox
        self._conf = confidence

    def model_versions(self):
        return {"stub": "1.0"}

    def infer(self, image: np.ndarray, frame_id: int, timestamp_ms: int) -> PerceptionResult:
        h, w = image.shape[:2]
        return PerceptionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            frame_size=(w, h),
            subjects=[
                Subject(
                    subject_id="s0",
                    label="person",
                    bbox=self._bbox,
                    confidence=self._conf,
                    is_primary=True,
                    source=SubjectSource.AUTO,
                )
            ],
            backend=self.name,
        )

    def warmup(self):
        return None


class FailingPerception(BasePerception):
    """总是抛异常的感知桩件，用于验证降级路径。"""

    name = "failing"

    def infer(self, image, frame_id, timestamp_ms):
        raise RuntimeError("注入的感知故障")


class EmptyPerception(BasePerception):
    """返回空主体列表的桩件。"""

    name = "empty"

    def infer(self, image, frame_id, timestamp_ms):
        h, w = image.shape[:2]
        return PerceptionResult(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            frame_size=(w, h),
            subjects=[],
            backend=self.name,
        )


def _frame(i: int, w: int = 320, h: int = 240) -> Frame:
    img = np.full((h, w, 3), 120, np.uint8)
    return Frame(frame_id=i, image=img, timestamp_ms=i * 333, capture_ms=0.5)


# ---------------------------------------------------------------------------
# 单帧处理
# ---------------------------------------------------------------------------
class TestFrameProcessor:
    def test_produces_valid_snapshot(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())

        assert isinstance(snap, FrameSnapshot)
        assert snap.frame_id == 0
        assert snap.schema_version == "0.1"
        assert snap.degraded is False
        assert snap.perception.primary_subject is not None
        assert snap.composition.best_bbox is not None
        assert snap.command.command.action in set(ActionType)

    def test_latency_breakdown_populated(self, settings_factory):
        """延迟分解各阶段必须被填充，且 total 为各阶段之和。"""
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())

        lat = snap.latency
        assert lat.perception_ms > 0
        assert lat.decision_ms > 0
        expected = lat.capture_ms + lat.perception_ms + lat.decision_ms + lat.stabilization_ms
        assert lat.total_ms == pytest.approx(expected, rel=1e-6)

    def test_contract_serializable(self, settings_factory):
        """核心契约必须可 JSON 序列化（接口层依赖此性质）。"""
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())
        payload = snap.model_dump(mode="json")
        assert payload["frame_id"] == 0
        assert "composition" in payload and "command" in payload

    def test_overview_runnable(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())
        assert "#0" in snap.overview()


# ---------------------------------------------------------------------------
# 降级路径（故障注入）
# ---------------------------------------------------------------------------
class TestDegradation:
    def test_perception_exception_degrades_not_raises(self, settings_factory):
        """感知抛异常 → 必须返回降级快照，而不是向上抛。"""
        cfg = settings_factory()
        proc = FrameProcessor(FailingPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())

        assert snap.degraded is True
        assert snap.perception.primary_subject is None
        assert snap.perception.degraded is True
        assert snap.command.command.action is ActionType.HOLD

    def test_no_subject_degrades(self, settings_factory):
        """无主体 → 降级但给"保持"指令，不报错。"""
        cfg = settings_factory()
        proc = FrameProcessor(EmptyPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())

        assert snap.degraded is True
        assert snap.command.command.action is ActionType.HOLD
        assert snap.command.command.is_hold is True

    def test_degradation_reason_recorded(self, settings_factory):
        """降级必须带原因，供接口层告知前端（NFR-R1）。"""
        cfg = settings_factory()
        proc = FrameProcessor(FailingPerception(), cfg)
        snap = proc.process(_frame(0), FrameContext())
        assert snap.degradation_reason is not None

    def test_loop_survives_all_failing_frames(self, settings_factory):
        """全程感知失败时循环仍需跑完所有帧。"""
        cfg = settings_factory()
        proc = FrameProcessor(FailingPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        frames = [_frame(i) for i in range(10)]
        result = loop.run(_ListSource(frames))

        assert result.frames_processed == 10
        assert result.stopped_reason == "completed"
        assert all(s.degraded for s in result.snapshots)


# ---------------------------------------------------------------------------
# 会话上下文
# ---------------------------------------------------------------------------
class TestFrameContext:
    def test_subject_override_takes_priority(self, settings_factory):
        """FR-02：手动指定主体应覆盖自动检测结果。"""
        cfg = settings_factory()
        proc = FrameProcessor(EmptyPerception(), cfg)
        ctx = FrameContext(subject_override_bbox=(0.2, 0.1, 0.5, 0.8))
        snap = proc.process(_frame(0), ctx)

        subj = snap.perception.primary_subject
        assert subj is not None
        assert subj.source is SubjectSource.MANUAL
        assert subj.bbox == pytest.approx((0.2, 0.1, 0.5, 0.8))

    def test_override_not_flagged_degraded(self, settings_factory):
        """手动指定后不应因"自动检测失败"而标记降级。"""
        cfg = settings_factory()
        proc = FrameProcessor(EmptyPerception(), cfg)
        ctx = FrameContext(subject_override_bbox=(0.2, 0.1, 0.5, 0.8))
        snap = proc.process(_frame(0), ctx)
        assert snap.perception.degraded is False

    def test_snapshots_accumulate(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        ctx = FrameContext()
        for i in range(5):
            proc.process(_frame(i), ctx)
        assert len(ctx.snapshots) == 5
        assert ctx.frame_counter == 5


# ---------------------------------------------------------------------------
# 引导循环
# ---------------------------------------------------------------------------
class TestGuidingLoop:
    def test_full_run_produces_metrics(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        frames = [_frame(i) for i in range(30)]
        result = loop.run(_ListSource(frames))

        assert result.frames_processed == 30
        assert result.duration_s > 0
        assert "stages" in result.latency_report
        assert "total" in result.latency_report["stages"]
        assert "switches_per_minute" in result.switch_stats

    def test_max_frames_respected(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        frames = [_frame(i) for i in range(50)]
        result = loop.run(_ListSource(frames), max_frames=7)

        assert result.frames_processed == 7
        assert result.stopped_reason == "max_frames"

    def test_on_frame_callback_invoked(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        seen: list[int] = []
        loop.run(_ListSource([_frame(i) for i in range(5)]), on_frame=lambda s: seen.append(s.frame_id))
        assert seen == [0, 1, 2, 3, 4]

    def test_stream_yields_snapshots(self, settings_factory):
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        out = list(loop.stream(_ListSource([_frame(i) for i in range(6)])))
        assert len(out) == 6
        assert all(isinstance(s, FrameSnapshot) for s in out)

    def test_source_closed_after_run(self, settings_factory):
        """循环结束后必须关闭帧源（防止摄像头句柄泄漏）。"""
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        src = _ListSource([_frame(i) for i in range(3)])
        loop.run(src)
        assert src._closed is True


# ---------------------------------------------------------------------------
# 稳定性验收（对齐 NFR-P2）
# ---------------------------------------------------------------------------
class TestStabilityAcceptance:
    def test_stable_scene_no_command_flapping(self, settings_factory):
        """主体位置固定 → 循环全程不应出现指令翻转。"""
        cfg = settings_factory()
        proc = FrameProcessor(StubPerception(), cfg)
        loop = GuidingLoop(proc, cfg)
        frames = [_frame(i) for i in range(60)]
        result = loop.run(_ListSource(frames))

        actions = {s.command.command.action for s in result.snapshots}
        assert len(actions) == 1, f"稳定场景出现了 {len(actions)} 种指令: {actions}"
        assert result.switch_stats["switches_per_minute"] < 30.0
