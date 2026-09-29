#!/usr/bin/env python
"""延迟基准测试（NFR-P1）。

对应需求：NFR-P1（引导延迟）、NFR-O4（运行日志）
对应文档：《测试与验收.md》§4 量化验收

**为什么单列一个基准脚本**：Demo 跑一次得到的均值不足以支撑性能结论。
本脚本做的是：

1. **分离冷启动与稳态**——首帧含模型加载，必须单独统计（否则 P95 失真）；
2. **多轮重复取中位数**——抵消系统调度抖动；
3. **输出分阶段分解**——定位瓶颈到具体层，而不只是"总耗时 X ms"。

用法::

    python scripts/benchmark_latency.py                          # 默认 100 帧
    python scripts/benchmark_latency.py --frames 300 --repeat 3
    python scripts/benchmark_latency.py --backend rule           # 对比规则后端
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from aicg.observability import get_logger, setup_logging  # noqa: E402
from aicg.perception import build_perception  # noqa: E402
from aicg.perception.factory import perception_from_settings  # noqa: E402
from aicg.pipeline import FrameContext, FrameProcessor  # noqa: E402
from aicg.settings import PROJECT_ROOT, load_settings  # noqa: E402

log = get_logger("scripts.benchmark")


def _make_frames(n: int, w: int = 480, h: int = 640) -> list[np.ndarray]:
    """生成 n 帧合成人像（确定性，保证可复现）。"""
    frames = []
    for i in range(n):
        t = i / max(1, n - 1)
        img = np.zeros((h, w, 3), np.uint8)
        for y in range(h):
            v = int(200 - 90 * y / h)
            img[y, :] = (v + 30, v + 10, v)
        gy = int(h * 0.72)
        rng = np.random.default_rng(i)
        img[gy:, :] = rng.integers(90, 130, size=(h - gy, w, 3), dtype=np.uint8)

        bh = (0.3 + 0.55 * t) * h
        bw = bh * 0.42
        cx = (0.62 - 0.12 * t) * w
        cy = 0.55 * h
        import cv2

        cv2.ellipse(img, (int(cx), int(cy)), (int(bw / 2), int(bh / 2)), 0, 0, 360, (150, 170, 215), -1)
        frames.append(img)
    return frames


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def benchmark_once(frames: list[np.ndarray], cfg, processor: FrameProcessor) -> dict:
    """单轮基准测试。返回分阶段耗时统计（ms）。"""
    from aicg.camera.base import Frame

    ctx = FrameContext()
    stages: dict[str, list[float]] = {
        "perception": [], "decision": [], "stabilization": [], "total": [],
    }
    first_total: float | None = None

    for i, img in enumerate(frames):
        frame = Frame(frame_id=i, image=img, timestamp_ms=int(i * 1000 / 15), capture_ms=0.0)
        snap = processor.process(frame, ctx)

        lat = snap.latency
        if i == 0:
            first_total = lat.total_ms
            continue  # 首帧计入冷启动，不进稳态统计

        stages["perception"].append(lat.perception_ms)
        stages["decision"].append(lat.decision_ms)
        stages["stabilization"].append(lat.stabilization_ms)
        stages["total"].append(lat.total_ms)

    out: dict = {"first_frame_ms": round(first_total or 0.0, 2), "stages": {}}
    for name, vals in stages.items():
        if vals:
            out["stages"][name] = {
                "mean": round(statistics.fmean(vals), 2),
                "p50": round(_percentile(vals, 0.50), 2),
                "p95": round(_percentile(vals, 0.95), 2),
                "p99": round(_percentile(vals, 0.99), 2),
                "max": round(max(vals), 2),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="引导回路延迟基准测试")
    ap.add_argument("--frames", type=int, default=100, help="每轮帧数")
    ap.add_argument("--repeat", type=int, default=3, help="重复轮数（取中位数）")
    ap.add_argument("--backend", default=None, choices=["yolo", "rule"], help="强制感知后端")
    ap.add_argument("--device", default=None, help="强制推理设备（cpu/cuda）")
    ap.add_argument("--out", default=None, help="报告输出路径")
    args = ap.parse_args()

    setup_logging("WARNING")

    overrides: dict = {}
    if args.backend:
        overrides["perception.backend"] = args.backend
    if args.device:
        overrides["perception.detector.device"] = args.device
    cfg = load_settings(overrides=overrides or None)

    print("=" * 72)
    print("  延迟基准测试")
    print("=" * 72)
    print(f"  感知后端 : {cfg.perception.effective_backend()}  (device={cfg.perception.detector.device})")
    print(f"  每轮帧数 : {args.frames}   重复轮数: {args.repeat}")
    print(f"  FPS 设定 : {cfg.pipeline.target_fps}（延迟预算 {1000 / cfg.pipeline.target_fps:.0f}ms/帧）")

    frames = _make_frames(args.frames)
    # 必须用工厂函数而非 build_perception() 直接调用：后者不会读取配置里
    # 权重路径与阈值，会静默退化成默认规则后端，导致基准测的不是目标后端。
    perception = perception_from_settings(cfg)
    print(f"  感知实例 : {perception.name}")

    # 权重未就绪时明确提示，避免误把 rule 的数字当成 yolo 的成绩
    if cfg.perception.effective_backend() == "yolo" and perception.name != "yolo":
        print("  警告     : 配置要求 yolo，但权重未能加载 → 实际使用规则后端")
        print("             绕过方式：python scripts/download_weights.py")

    # 预热：把模型加载等一次性成本排除在稳态统计之外
    processor = FrameProcessor(perception, cfg)
    t0 = time.perf_counter()
    processor.warmup()
    print(f"  预热耗时 : {(time.perf_counter() - t0) * 1000:.0f}ms")

    runs: list[dict] = []
    for r in range(args.repeat):
        # 每轮重置会话状态，避免防抖余温影响
        processor.reset()
        res = benchmark_once(frames, cfg, processor)
        runs.append(res)
        total = res["stages"]["total"]
        print(
            f"  第 {r + 1} 轮: mean={total['mean']:6.2f}  p50={total['p50']:6.2f}  "
            f"p95={total['p95']:6.2f}  p99={total['p99']:6.2f}"
        )

    # 取各轮中位数作为最终结论
    final: dict = {"stages": {}, "meta": {
        "backend": cfg.perception.effective_backend(),
        "device": cfg.perception.detector.device,
        "frames_per_run": args.frames,
        "repeats": args.repeat,
        "target_fps": cfg.pipeline.target_fps,
    }}
    for stage in ("perception", "decision", "stabilization", "total"):
        for metric in ("mean", "p50", "p95", "p99", "max"):
            vals = [r["stages"][stage][metric] for r in runs if stage in r["stages"]]
            if vals:
                final["stages"].setdefault(stage, {})[metric] = round(statistics.median(vals), 2)

    final["first_frame_ms"] = round(statistics.median([r["first_frame_ms"] for r in runs]), 2)

    print("\n" + "-" * 72)
    print("  稳态延迟（各轮中位数，ms）")
    print("-" * 72)
    print(f"  {'阶段':<16s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'max':>9s}")
    for stage, st in final["stages"].items():
        print(f"  {stage:<16s} {st['mean']:>8.2f} {st['p50']:>8.2f} {st['p95']:>8.2f} {st['p99']:>8.2f} {st['max']:>9.2f}")

    budget = 1000.0 / cfg.pipeline.target_fps
    p95 = final["stages"]["total"]["p95"]
    print(f"\n  首帧（含冷启动）: {final['first_frame_ms']:.1f}ms")
    print(f"  单帧预算         : {budget:.1f}ms")
    print(f"  P95 占用         : {p95 / budget * 100:.1f}%")
    verdict = "通过" if p95 < budget else "未通过"
    print(f"  NFR-P1 判定      : {verdict}")

    # 瓶颈归因
    stage_means = {k: v["mean"] for k, v in final["stages"].items() if k != "total"}
    if stage_means:
        worst = max(stage_means, key=stage_means.get)
        share = stage_means[worst] / max(1e-6, final["stages"]["total"]["mean"]) * 100
        print(f"  主要瓶颈         : {worst}（占比 {share:.0f}%）")

    out_path = Path(args.out) if args.out else (
        cfg.observability.resolved_report_dir() / f"benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存: {out_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
