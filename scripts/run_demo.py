#!/usr/bin/env python
"""录屏 Demo：跑一遍引导循环并输出带标注的视频 + 指标报告。

对应需求：FR-05、FR-06、NFR-P1、NFR-P2
对应文档：《开发计划.md》M3 出口条件 —— "交付一个能看的录屏引导 Demo"

这是 M3 里程碑的**主交付脚本**。一条命令产出三样东西：

1. **带标注的引导视频**（``outputs/demo/*.mp4``）—— 录屏素材；
2. **延迟报告**（``outputs/reports/latency_*.json``）—— 证明实时性；
3. **防抖对照报告**（``outputs/reports/debounce_*.json``）—— 证明稳定性。

``--compare`` 会在同一批帧上跑"开关防抖"对照，这是自证 FR-06 价值的
唯一诚实方式。

**关于 ``--backend``（重要，来自实测）**：

本项目的测试素材是**分后端专用**的，原因是 YOLOv8 无法检出任何
"代码画出来的人形"（详见 ``tests/fixtures/README.md``）。因此：

- 合成素材（``walk_towards`` / ``handheld_jitter``）**必须**配 ``rule``；
  用 yolo 会得到"全程降级、0% 降幅"的无意义结果；
- 真实素材（``real_photo_zoom``）**必须**配 ``yolo``。

脚本会依据素材名自动选择默认后端（合成→rule，真实→yolo），
也可用 ``--backend`` 显式覆盖。这是"双源分流"在工具层的落地。

用法::

    # 合成素材：防抖对照（自动选 rule）
    python scripts/run_demo.py --source tests/fixtures/handheld_jitter.mp4 --compare

    # 真实素材：验证 yolo 链路（自动选 yolo）
    python scripts/run_demo.py --source tests/fixtures/real_photo_zoom.mp4 --compare

    # 显式指定后端
    python scripts/run_demo.py --source tests/fixtures/handheld_jitter.mp4 --backend rule

    # 用摄像头
    python scripts/run_demo.py --source camera:0 --max-frames 150
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许直接以脚本方式运行（无需安装包）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aicg.camera import open_source  # noqa: E402
from aicg.observability import setup_logging, get_logger  # noqa: E402
from aicg.perception import (
    infer_backend_for_source,
    perception_from_settings,
)  # noqa: E402
from aicg.pipeline import FrameProcessor, GuidingLoop, compare_debounce  # noqa: E402
from aicg.settings import load_settings, PROJECT_ROOT  # noqa: E402
from aicg.viz import SnapshotRenderer  # noqa: E402

log = get_logger("scripts.demo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI 实时构图指导 —— 录屏 Demo")
    p.add_argument("--source", default="tests/fixtures/walk_towards.mp4",
                   help="帧源：视频路径 / camera:0 / 图片目录")
    p.add_argument("--backend", default=None, choices=["auto", "rule", "yolo"],
                   help="感知后端；默认按素材名自动推荐（合成→rule，真实→yolo）")
    p.add_argument("--output", default=None, help="输出视频路径（默认自动命名）")
    p.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    p.add_argument("--compare", action="store_true", help="运行防抖开关对照实验")
    p.add_argument("--config", default=None, help="自定义配置文件路径")
    p.add_argument("--fps", type=float, default=None, help="覆盖目标 FPS")
    p.add_argument("--no-render", action="store_true", help="跳过视频渲染（仅出指标）")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_backend(args: argparse.Namespace) -> str:
    """确定本次运行使用的感知后端，并说明依据。

    优先级：显式 ``--backend`` > 素材名推荐 > 配置默认。

    为什么要显式提示"自动选择"：若不加说明，使用者看到合成素材自动
    选了 rule，会以为 yolo 坏了；看到真实素材选了 yolo，又会以为
    防抖失效。把决策依据打出来，避免误读指标。
    """
    if args.backend and args.backend != "auto":
        return args.backend
    hint = infer_backend_for_source(args.source)
    if hint:
        log.info("按素材名自动选用后端: %s（依据 tests/fixtures/README.md）", hint)
        return hint
    return "auto"


def main() -> int:
    args = parse_args()
    setup_logging("WARNING" if args.quiet else "INFO")

    overrides = {}
    if args.fps is not None:
        overrides["pipeline.target_fps"] = args.fps

    backend = resolve_backend(args)
    if backend != "auto":
        # 显式把配置的 backend 钉死，避免 effective_backend() 走
        # "权重存在就用 yolo" 的自动分支而与我们选定的后端不一致。
        overrides["perception.backend"] = backend

    cfg = load_settings(args.config, overrides or None)
    log.info("配置摘要: %s", cfg.summary())

    out_dir = PROJECT_ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = cfg.observability.resolved_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    source_name = Path(args.source).stem if not args.source.startswith("camera:") else "webcam"

    # ---------------- 1. 引导循环 + 渲染 ----------------
    # 用工厂函数读取配置里的权重路径/阈值/device；直接用 build_perception()
    # 会忽略这些设置并静默退回规则后端。
    perception = perception_from_settings(cfg)
    processor = FrameProcessor(perception, cfg)
    loop = GuidingLoop(processor, cfg)
    log.info("感知后端: %s (请求: %s)", perception.name, backend)

    # 安全护栏：合成素材 + yolo 后端 = 全程降级、指标无意义。
    # 与其产出误导性数字，不如当场说清。
    if perception.name == "yolo" and infer_backend_for_source(args.source) == "rule":
        print(
            "\n  [警告] 检测到「合成素材 + yolo 后端」的组合。\n"
            "         YOLOv8 无法检出合成人形，所有帧都会走降级兜底返回 hold，\n"
            "         由此得到的切换频率/防抖降幅**没有意义**。\n"
            "         合成素材请用 --backend rule（或去掉 --backend 让其自动选择）。\n"
            "         依据见 tests/fixtures/README.md。\n"
        )

    renderer = SnapshotRenderer(scale=1.0)
    writer = None
    video_path: Path | None = None
    snapshots: list = []
    frames_raw: list = []
    # 手工循环的条件：需要渲染，或需要 frames_raw 做防抖对照。
    # （早期只在渲染分支收集帧，导致 `--compare --no-render` 被静默跳过——
    #   用户以为跑了对照，其实什么都没输出。）
    manual_loop = (not args.no_render) or args.compare

    if manual_loop:
        # 需要原始帧才能渲染 / 做对照，因此这里手写循环而非用 loop.run
        src = open_source(args.source, cfg.pipeline.frame_downsample_width)
        processor.warmup()
        t0 = time.perf_counter()
        count = 0
        try:
            for frame in src.frames():
                snap = processor.process(frame, loop.context)
                loop.latency.add(snap.latency)
                loop.switch_tracker.add(snap.command.command.action.value, frame.timestamp_ms)

                if not args.no_render:
                    canvas = renderer.render(frame.image, snap)
                    if writer is None:
                        h, w = canvas.shape[:2]
                        video_path = out_dir / f"guide_{source_name}_{stamp}.mp4"
                        writer = cv2.VideoWriter(
                            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h)
                        )
                        log.info("输出视频: %s  (%dx%d)", video_path, w, h)
                    writer.write(canvas)

                snapshots.append(snap)
                frames_raw.append(frame)
                count += 1
                if args.max_frames and count >= args.max_frames:
                    break
        finally:
            src.close()
            if writer is not None:
                writer.release()

        duration = time.perf_counter() - t0
        from aicg.pipeline.guiding_loop import LoopResult
        result = LoopResult(
            frames_processed=count,
            duration_s=duration,
            snapshots=snapshots,
            latency_report=loop.latency.to_report(
                meta={
                    "source": args.source,
                    "target_fps": cfg.pipeline.target_fps,
                    "perception_backend": perception.name,
                    "perception_backend_requested": backend,
                }
            ),
            switch_stats=loop.switch_tracker.stats() if loop.switch_tracker else {},
            stopped_reason="max_frames" if args.max_frames and count >= args.max_frames else "completed",
        )
    else:
        src = open_source(args.source, cfg.pipeline.frame_downsample_width)
        result = loop.run(src, max_frames=args.max_frames)

    # ---------------- 2. 指标输出 ----------------
    print("\n" + "=" * 68)
    print("  引导循环结果")
    print("=" * 68)
    print(f"  处理帧数 : {result.frames_processed}")
    print(f"  实际用时 : {result.duration_s:.2f}s")
    print(f"  停止原因 : {result.stopped_reason}")
    print("\n  分阶段延迟 (ms):")
    print(f"  {'阶段':<16s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'max':>9s}")
    for stage, st in result.latency_report.get("stages", {}).items():
        print(f"  {stage:<16s} {st['mean']:>8.2f} {st['p50']:>8.2f} {st['p95']:>8.2f} {st['max']:>9.2f}")
    print("\n  指令稳定性:")
    for k, v in result.switch_stats.items():
        print(f"    {k:<22s} {v}")

    if snapshots:
        print("\n  指令分布（Top-5）:")
        from collections import Counter
        cnt = Counter(s.command.command.action.value for s in snapshots)
        for act, n in cnt.most_common(5):
            print(f"    {act:<14s} {n:>4d} 帧  ({n / len(snapshots) * 100:5.1f}%)")

    # 落盘延迟报告
    lat_path = report_dir / f"latency_{source_name}_{stamp}.json"
    lat_path.write_text(
        json.dumps(result.latency_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  延迟报告: {lat_path}")

    # ---------------- 3. 防抖对照 ----------------
    if args.compare and frames_raw:
        print("\n" + "=" * 68)
        print("  防抖对照实验（同一批帧，仅切换防抖开关）")
        print("=" * 68)
        cmp_result = compare_debounce(frames_raw, cfg, perception=perception)
        off, on = cmp_result["off"]["switch_stats"], cmp_result["on"]["switch_stats"]
        print(f"  {'指标':<24s} {'关闭防抖':>12s} {'开启防抖':>12s}")
        print(f"  {'-'*24} {'-'*12} {'-'*12}")
        print(f"  {'指令切换次数':<22s} {off['switch_count']:>12.0f} {on['switch_count']:>12.0f}")
        print(f"  {'切换频率(次/分)':<21s} {off['switches_per_minute']:>12.2f} {on['switches_per_minute']:>12.2f}")
        print(f"  {'平均指令持续帧数':<21s} {off['mean_run_frames']:>12.2f} {on['mean_run_frames']:>12.2f}")
        print(f"\n  >>> 切换频率下降: {cmp_result['switch_reduction_pct']}%")
        print(f"  >>> 平均指令持续帧数提升: {on['mean_run_frames'] / max(1e-9, off['mean_run_frames']):.1f} 倍")

        cmp_path = report_dir / f"debounce_{source_name}_{stamp}.json"
        cmp_path.write_text(json.dumps(cmp_result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  对照报告: {cmp_path}")

    if video_path is not None:
        print(f"\n  引导视频: {video_path}")
    print()

    # 输出机器可读摘要，便于 CI / 简历数据引用
    summary = {
        "video": str(video_path) if video_path else None,
        "frames": result.frames_processed,
        "duration_s": round(result.duration_s, 3),
        "latency_p95_ms": result.latency_report.get("stages", {}).get("total", {}).get("p95"),
        "latency_mean_ms": result.latency_report.get("stages", {}).get("total", {}).get("mean"),
        "switches_per_minute": result.switch_stats.get("switches_per_minute"),
    }
    print("SUMMARY_JSON " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
