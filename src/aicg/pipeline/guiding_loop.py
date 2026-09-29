"""实时引导循环（FR-05）—— 端到端串联。

对应需求：FR-05（实时引导循环）、FR-06（防抖）、NFR-P1（延迟）、NFR-P2（稳定性）
对应文档：《技术方案.md》§1 五层架构

**这是全项目的"主循环"**，也是 Demo 录屏时跑的东西::

    for frame in source.frames():
        snapshot = processor.process(frame, ctx)
        yield snapshot            # 上层（CLI / API / 可视化）消费

关键设计决策：

1. **生成器而非回调**：以生成器产出 ``FrameSnapshot``，调用方自行决定
   是"画框上屏"「打日志」还是"丢进 WebSocket"。这样同一循环同时服务
   录屏轨道与 Web 轨道，无需改算法层。

2. **指标内建而非外挂**：循环内部维护 ``LatencyTracker`` 与
   ``CommandSwitchTracker``。防抖效果（切换频率下降幅度）是简历上最
   有说服力的数字，因此**默认采集**，而不是事后补脚本。

3. **双轨防抖对照**：``compare_debounce=True`` 时，同一批帧会分别经过
   "关闭防抖"与"开启防抖"两条支路，直接产出对比指标。这是自证
   FR-06 价值的唯一诚实方式——**用同一份输入证明差异**。

4. **热路径无 IO**：《编码规范.md》§3.5 硬约束——循环内不打日志、
   不写盘。可视化与落盘由调用方在循环外做。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from ..camera.base import FrameSource
from ..observability import (
    CommandSwitchTracker,
    LatencyTracker,
    get_logger,
)
from ..schemas import ActionType, FrameSnapshot
from ..settings import Settings, get_settings
from ..stabilization import CommandDebouncer
from .frame_processor import FrameContext, FrameProcessor

log = get_logger("pipeline.loop")


@dataclass
class LoopResult:
    """一次引导循环的汇总结果。"""

    frames_processed: int = 0
    duration_s: float = 0.0
    snapshots: list[FrameSnapshot] = field(default_factory=list)
    latency_report: dict = field(default_factory=dict)
    switch_stats: dict = field(default_factory=dict)
    stopped_reason: str = "completed"
    """停止原因：``completed`` / ``max_frames`` / ``error``。"""


class GuidingLoop:
    """实时引导主循环。

    与 :class:`FrameProcessor` 的分工：
    - ``FrameProcessor``：**单帧** 的算法编排（无会话状态）；
    - ``GuidingLoop``：**多帧** 的会话管理 + 指标采集 + 两条防抖支路。
    """

    def __init__(
        self,
        processor: FrameProcessor,
        settings: Settings | None = None,
        *,
        track_switch: bool = True,
    ) -> None:
        self.processor = processor
        self.settings = settings or get_settings()
        self.latency = LatencyTracker(window=self.settings.observability.latency_window)
        self._switch_tracker = CommandSwitchTracker() if track_switch else None
        self.context = FrameContext()

    # ------------------------------------------------------------------
    def run(
        self,
        source: FrameSource,
        *,
        max_frames: int | None = None,
        on_frame=None,
    ) -> LoopResult:
        """执行引导循环。

        Args:
            source: 帧源（视频文件 / 摄像头 / 图片序列）。
            max_frames: 最大处理帧数；None 表示跑完整个源。
            on_frame: 可选回调 ``(snapshot) -> None``，在每帧处理后调用。
                **不要在此回调里做重 IO**——它位于主循环路径上。

        Returns:
            循环汇总结果，含完整快照序列与指标报告。
        """
        result = LoopResult()
        t_start = time.perf_counter()

        self.processor.warmup()
        log.info(
            "引导循环启动: source=%s fps=%.1f max_frames=%s",
            type(source).__name__,
            self.settings.pipeline.target_fps,
            max_frames if max_frames is not None else "∞",
        )

        try:
            for frame in source.frames():
                snapshot = self.processor.process(frame, self.context)
                result.snapshots.append(snapshot)
                self.latency.add(snapshot.latency)
                if self._switch_tracker is not None:
                    self._switch_tracker.add(snapshot.command.command.action.value, frame.timestamp_ms)

                if on_frame is not None:
                    on_frame(snapshot)

                result.frames_processed += 1
                if max_frames is not None and result.frames_processed >= max_frames:
                    result.stopped_reason = "max_frames"
                    break
        except Exception as e:  # noqa: BLE001 - 循环不得因单点异常整体崩掉
            log.error("引导循环异常中断: %s", e)
            result.stopped_reason = "error"
        finally:
            source.close()

        result.duration_s = time.perf_counter() - t_start
        result.latency_report = self.latency.to_report(
            meta={
                "target_fps": self.settings.pipeline.target_fps,
                "perception_backend": self.settings.perception.effective_backend(),
                "downsample_width": self.settings.pipeline.frame_downsample_width,
                "frames": result.frames_processed,
            }
        )
        if self._switch_tracker is not None:
            result.switch_stats = self._switch_tracker.stats()

        log.info(
            "引导循环结束: frames=%d 用时=%.2fs 切换=%.2f次/分",
            result.frames_processed,
            result.duration_s,
            result.switch_stats.get("switches_per_minute", 0.0),
        )
        return result

    # ------------------------------------------------------------------
    def stream(self, source: FrameSource, *, max_frames: int | None = None) -> Iterator[FrameSnapshot]:
        """流式产出快照，供 WebSocket / 可视化逐帧消费。

        与 :meth:`run` 的差异：不收集全部快照（避免长视频占用内存），
        只产出。指标仍需在调用方自行汇总。
        """
        self.processor.warmup()
        count = 0
        try:
            for frame in source.frames():
                snapshot = self.processor.process(frame, self.context)
                self.latency.add(snapshot.latency)
                if self._switch_tracker is not None:
                    self._switch_tracker.add(snapshot.command.command.action.value, frame.timestamp_ms)
                yield snapshot
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            source.close()

    @property
    def switch_tracker(self) -> CommandSwitchTracker | None:
        return self._switch_tracker


# ----------------------------------------------------------------------
def compare_debounce(
    source_frames: list,
    settings: Settings,
    *,
    perception=None,
) -> dict:
    """防抖开关对照实验：同一批帧分别跑两条支路。

    这是**自证 FR-06 价值的唯一诚实方式**——用同一份输入、同一套感知与
    评分，只切换防抖开关，直接观测指令切换频率的差异。任何"我们做了
    防抖所以更好"的说法，都必须能拿出这张对照表。

    Args:
        source_frames: 已读入内存的帧列表（需可重复遍历，故不能用迭代器）。
        settings: 全局配置。**后端取自该配置**（``perception.backend``），
            而不是固定选 auto —— 否则调用方钉死的后端会被这里悄悄改写，
            导致对照实验跑在与预期不同的后端上（实测踩过的坑）。
        perception: 可选的共享感知实例；None 时按配置构建。

    Returns:
        含两条支路统计的对照字典。
    """
    from ..perception.factory import perception_from_settings

    # 用工厂读配置，尊重调用方钉死的 backend（rule / yolo / auto）
    shared = perception or perception_from_settings(settings)

    def _run(debounce_enabled: bool) -> dict:
        cfg = settings.model_copy(deep=True)
        cfg.stabilization.debounce.enabled = debounce_enabled
        # 关闭防抖时同时关闭 EMA，否则测的是"EMA 的贡献"而非"防抖总贡献"
        cfg.stabilization.ema.enabled = debounce_enabled

        debouncer = CommandDebouncer(cfg.stabilization)
        proc = FrameProcessor(shared, cfg, debouncer=debouncer)
        loop = GuidingLoop(proc, cfg)
        result = loop.run(_ListSource(source_frames, settings.pipeline.frame_downsample_width))
        return {
            "debounce_enabled": debounce_enabled,
            "switch_stats": result.switch_stats,
            "latency": result.latency_report.get("stages", {}),
        }

    off = _run(False)
    on = _run(True)

    off_spm = off["switch_stats"].get("switches_per_minute", 0.0)
    on_spm = on["switch_stats"].get("switches_per_minute", 0.0)
    reduction = (1.0 - on_spm / off_spm) if off_spm > 0 else 0.0

    return {
        "kind": "debounce_comparison",
        "frames": len(source_frames),
        "off": off,
        "on": on,
        "switch_reduction_pct": round(reduction * 100.0, 2),
        "mean_run_frames_before": off["switch_stats"].get("mean_run_frames", 0.0),
        "mean_run_frames_after": on["switch_stats"].get("mean_run_frames", 0.0),
    }


class _ListSource(FrameSource):
    """把内存中的帧列表包装成帧源（供对照实验重复遍历）。"""

    def __init__(self, frames: list, downsample_width: int | None = None) -> None:
        super().__init__(downsample_width=downsample_width)
        self._frames = list(frames)

    def _iter_raw(self):
        for f in self._frames:
            yield f

    @property
    def total_frames(self) -> int | None:
        return len(self._frames)


def build_loop(settings: Settings | None = None, *, max_frames: int | None = None) -> GuidingLoop:
    """按配置构建引导循环（含自动感知后端选择）。

    Args:
        settings: 全局配置；None 时读取单例。
        max_frames: 仅用于日志标注，不在此处生效。
    """
    from ..perception.factory import perception_from_settings

    cfg = settings or get_settings()
    # 注意：必须用 perception_from_settings() 而不是 build_perception()。
    # 后者的 backend 参数是**裸字符串**，不会读取配置中的权重路径、阈值与
    # device 设置，会静默退回默认规则后端——从而出现"配置写着 yolo，
    # 实际跑着 rule"的静默降级，指标与结论都会失真。
    perception = perception_from_settings(cfg)
    processor = FrameProcessor(perception, cfg)
    return GuidingLoop(processor, cfg)
