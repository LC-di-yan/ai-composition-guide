"""分阶段耗时打点与延迟统计。

对应需求：NFR-P1（引导延迟）、NFR-O3（指标可复现）、NFR-O4（运行日志）
对应文档：《数据模型与接口.md》§2.6 ``LatencyBreakdown``

设计约束（《编码规范.md》§3.5）：打点路径禁止做同步 IO，只做内存累加。
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..schemas.snapshot import LatencyBreakdown


class StageTimer:
    """单帧的分阶段计时器。

    用法::

        t = StageTimer()
        with t.stage("perception"):
            ...
        with t.stage("decision"):
            ...
        print(t.to_breakdown().total_ms)
    """

    _STAGE_FIELDS = {
        "capture": "capture_ms",
        "perception": "perception_ms",
        "decision": "decision_ms",
        "stabilization": "stabilization_ms",
        "language": "language_ms",
    }

    def __init__(self) -> None:
        self._acc: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """计时上下文。同名 stage 多次进入会累加。"""
        if name not in self._STAGE_FIELDS:
            raise KeyError(f"未知阶段 {name!r}，可选：{list(self._STAGE_FIELDS)}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._acc[name] = self._acc.get(name, 0.0) + elapsed_ms

    def mark(self, name: str, elapsed_ms: float) -> None:
        """直接累加一个已测得的耗时（用于测量发生在别处的场景）。"""
        self._acc[name] = self._acc.get(name, 0.0) + elapsed_ms

    def to_breakdown(self) -> LatencyBreakdown:
        """导出为契约对象。"""
        kw = {field: self._acc.get(stage, 0.0) for stage, field in self._STAGE_FIELDS.items()}
        bd = LatencyBreakdown(**kw)
        return bd.recompute_total()


def _percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数，q 取 0~1。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class LatencyTracker:
    """滑动窗口延迟统计，用于产出指标报告（NFR-P1）。"""

    def __init__(self, window: int = 300) -> None:
        self.window = window
        self._samples: deque[LatencyBreakdown] = deque(maxlen=window)

    def add(self, breakdown: LatencyBreakdown) -> None:
        self._samples.append(breakdown)

    def __len__(self) -> int:
        return len(self._samples)

    def stats(self) -> dict[str, dict[str, float]]:
        """按阶段返回 mean / p50 / p95 / p99 / max。"""
        out: dict[str, dict[str, float]] = {}
        for stage in ("capture", "perception", "decision", "stabilization", "total"):
            field = f"{stage}_ms"
            vals = sorted(float(getattr(b, field)) for b in self._samples if getattr(b, field) is not None)
            if not vals:
                continue
            out[stage] = {
                "n": float(len(vals)),
                "mean": round(statistics.fmean(vals), 2),
                "p50": round(_percentile(vals, 0.50), 2),
                "p95": round(_percentile(vals, 0.95), 2),
                "p99": round(_percentile(vals, 0.99), 2),
                "max": round(vals[-1], 2),
            }
        return out

    def to_report(self, meta: dict | None = None) -> dict:
        """生成完整报告字典（含元信息，保证可复现，NFR-O3）。"""
        return {
            "kind": "latency_report",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "window": self.window,
            "sample_count": len(self._samples),
            "meta": meta or {},
            "stages": self.stats(),
        }

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        """报告落盘为 JSON。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_report(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p


class CommandSwitchTracker:
    """指令切换频率统计。

    这是 FR-06 防抖效果的核心指标（NFR-P2）：切换次数越少、平均指令
    持续时间越长，说明引导越稳定。
    """

    def __init__(self) -> None:
        self.switch_count = 0
        self.total_frames = 0
        self._current_action: str | None = None
        self._current_run_len = 0
        self._run_lengths: list[int] = []
        self._first_ts_ms: int | None = None
        self._last_ts_ms: int | None = None

    def add(self, action: str, timestamp_ms: int) -> bool:
        """记录一帧的最终动作。

        Returns:
            本帧是否发生了动作切换。
        """
        self.total_frames += 1
        self._last_ts_ms = timestamp_ms
        if self._first_ts_ms is None:
            self._first_ts_ms = timestamp_ms

        changed = action != self._current_action
        if changed:
            if self._current_action is not None:
                self.switch_count += 1
                self._run_lengths.append(self._current_run_len)
            self._current_action = action
            self._current_run_len = 1
        else:
            self._current_run_len += 1
        return changed

    @property
    def duration_s(self) -> float:
        if self._first_ts_ms is None or self._last_ts_ms is None:
            return 0.0
        return max(1e-6, (self._last_ts_ms - self._first_ts_ms) / 1000.0)

    @property
    def switches_per_minute(self) -> float:
        """指令切换频率（次/分钟）—— 防抖对比的核心数值。"""
        return self.switch_count / self.duration_s * 60.0 if self.duration_s > 0 else 0.0

    @property
    def mean_run_frames(self) -> float:
        """平均每条指令持续的帧数。"""
        runs = list(self._run_lengths)
        if self._current_run_len > 0:
            runs.append(self._current_run_len)
        return statistics.fmean(runs) if runs else 0.0

    def stats(self) -> dict[str, float]:
        return {
            "total_frames": float(self.total_frames),
            "switch_count": float(self.switch_count),
            "duration_s": round(self.duration_s, 2),
            "switches_per_minute": round(self.switches_per_minute, 2),
            "mean_run_frames": round(self.mean_run_frames, 2),
        }
