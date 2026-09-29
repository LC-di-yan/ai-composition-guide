"""响应映射：``FrameSnapshot`` → API 响应结构。

对应文档：《数据模型与接口.md》§3.2 核心接口详述

**为什么需要单独的映射层**：

内部契约（``FrameSnapshot``）追求**完整可溯源**（NFR-O1），字段多、嵌套深；
而对外的 API 响应追求**稳定且精简**——前端不需要知道 `sub_scores` 的每一项，
也不该被内部字段改名牵连。

把两者解耦的价值：内部重构不会造成 API 破坏性变更；反之亦然。
若直接把 `FrameSnapshot` 当响应体返回，等于把内部实现锁进公开契约。

**关于 `magnitude_text` 的中文**：
CLI / 录屏轨因 OpenCV 字体限制只能渲染 ASCII，但 API 是 JSON、
经 UTF-8 传输，**不受该限制**。因此这里保留中文文案——中文用户看的
就是 Web 轨，理应给中文。
"""

from __future__ import annotations

from typing import Any

from ..schemas import ActionType, FrameSnapshot

# ActionType → 方向词。前端据此做箭头朝向。
_DIRECTION: dict[ActionType, str] = {
    ActionType.MOVE_CLOSER: "forward",
    ActionType.MOVE_BACK: "backward",
    ActionType.MOVE_LEFT: "left",
    ActionType.MOVE_RIGHT: "right",
    ActionType.TILT_UP: "up",
    ActionType.TILT_DOWN: "down",
    ActionType.HOLD: "hold",
}


def action_to_payload(snapshot: FrameSnapshot) -> dict[str, Any]:
    """把稳定化指令映射为对外结构。

    同时带上 ``raw_action`` 与抑制信息——前端可据此做"AI 正在犹豫"的
    可视化（例如把被抑制的抖动指令显示成灰字）。这既是调试价值，
    也是防抖机制的可视化证明。
    """
    cmd = snapshot.command
    inner = cmd.command
    raw = cmd.raw_command

    payload: dict[str, Any] = {
        "action": inner.action.value,
        "direction": _DIRECTION.get(inner.action, "hold"),
        "magnitude_text": inner.magnitude_text,
        "magnitude_raw": inner.magnitude_raw,
        "urgency": inner.urgency.value,
        "is_hold": inner.is_hold,
        "confidence": inner.confidence,
        "can_skip": inner.can_skip,
        # 防抖可观测性
        "is_changed": cmd.is_changed,
        "raw_action": raw.action.value,
        "suppressed_by": cmd.suppressed_by.value if cmd.suppressed_by else None,
        "consecutive_frames": cmd.consecutive_frames,
        "since_last_change_ms": cmd.since_last_change_ms,
    }
    if cmd.smoothing is not None:
        payload["smoothing"] = cmd.smoothing
    return payload


def composition_to_payload(snapshot: FrameSnapshot) -> dict[str, Any]:
    """把构图结果映射为对外结构（含人话标签）。"""
    comp = snapshot.composition
    return {
        "composition_score": comp.composition_score,
        "pattern": comp.pattern.value,
        "pattern_label": comp.pattern_label,
        "shot_size_label": comp.shot_size_label,
        "best_bbox": list(comp.best_bbox) if comp.best_bbox else None,
        "best_score": comp.best_score,
        "candidate_count": comp.candidate_count,
        "sub_scores": comp.sub_scores,
        "rule_violations": [
            {"rule": v.rule, "severity": v.severity, "detail": v.detail}
            for v in comp.rule_violations
        ],
        "degraded": comp.degraded,
    }


def subject_to_payload(snapshot: FrameSnapshot) -> dict[str, Any] | None:
    """主体摘要。无主体时返回 None（前端据此提示"长按指定主体"）。"""
    subj = snapshot.perception.primary_subject
    if subj is None:
        return None
    return {
        "subject_id": subj.subject_id,
        "label": subj.label,
        "bbox": list(subj.bbox),
        "confidence": subj.confidence,
        "source": subj.source.value,
        "height_ratio": subj.height,
        "area_ratio": subj.area,
    }


def latency_to_payload(snapshot: FrameSnapshot) -> dict[str, Any]:
    lat = snapshot.latency
    return {
        "capture_ms": lat.capture_ms,
        "perception_ms": lat.perception_ms,
        "decision_ms": lat.decision_ms,
        "stabilization_ms": lat.stabilization_ms,
        "total_ms": lat.total_ms,
    }


def snapshot_to_response(snapshot: FrameSnapshot) -> dict[str, Any]:
    """完整映射：一把把一份快照转成 API 响应体。"""
    payload: dict[str, Any] = {
        "schema_version": snapshot.schema_version,
        "frame_id": snapshot.frame_id,
        "timestamp_ms": snapshot.timestamp_ms,
        "frame_size": list(snapshot.frame_size),
        "command": action_to_payload(snapshot),
        "composition": composition_to_payload(snapshot),
        "subject": subject_to_payload(snapshot),
        "latency": latency_to_payload(snapshot),
        "degraded": snapshot.degraded,
        "degradation_reason": (
            snapshot.degradation_reason.value if snapshot.degradation_reason else None
        ),
        "backend": snapshot.perception.backend,
    }
    if snapshot.calibration is not None:
        cal = snapshot.calibration
        payload["calibration"] = {
            "method": cal.method.value if hasattr(cal.method, "value") else str(cal.method),
            "current_distance_m": getattr(cal, "current_distance_m", None),
            "ideal_distance_m": getattr(cal, "ideal_distance_m", None),
            "advice": getattr(cal, "advice", None),
            "confidence": getattr(cal, "confidence", None),
            "degraded": getattr(cal, "degraded", False),
        }
    return payload


__all__ = [
    "action_to_payload",
    "composition_to_payload",
    "latency_to_payload",
    "snapshot_to_response",
    "subject_to_payload",
]
