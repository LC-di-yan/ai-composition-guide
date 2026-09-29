"""命令行入口。

对应文档：《目录结构.md》§CLI、《开发计划.md》M3/M6
对应需求：FR-04、FR-05、全流程可运行（NFR-M4）

设计原则：**CLI 是胶水，不含业务逻辑**。所有子命令都只是调用
``pipeline`` / ``api`` / ``viz`` 中已有的能力。这样做的好处是
"能在 CLI 做的事，在 API 里也一定做得到"，两条交付轨道不会分叉。

子命令一览::

    aicg demo      # 录屏轨：跑引导循环，产出标注视频 + 指标（M3 主交付）
    aicg score     # 单图打分：给一张图打构图分，输出可视化
    aicg serve     # 起 FastAPI 服务（Web 轨）
    aicg make-fixtures  # 生成测试夹具（双源分流）
    aicg bench     # 延迟基准测试（NFR-P1）
    aicg eval      # 构图评分相关性评估
    aicg accept    # 端到端验收报告：延迟 + 抖动 + SRCC 一次跑完（NFR-O3）
    aicg doctor    # 环境自检：依赖 / 权重 / 后端可用性

若未安装为包（``pip install -e .``），也可用::

    python -m aicg.cli <子命令>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证能 import aicg（未安装为包时）。必须在导入 settings 之前完成。
_SRC = Path(__file__).resolve().parents[1].parent  # src/aicg/cli/__main__.py -> src
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 项目根目录。**不要手数 parents[]**——`__main__.py` 比常规模块深一层
# （src/aicg/cli/__main__.py），极易多算或少算。这里直接复用 settings 里
# 已定义好的常量，保证全项目只有一处真源。
from aicg.settings import PROJECT_ROOT  # noqa: E402


def _add_demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="tests/fixtures/handheld_jitter.mp4",
                   help="帧源：视频路径 / camera:0 / 图片目录")
    p.add_argument("--backend", default=None, choices=["auto", "rule", "yolo"],
                   help="感知后端；默认按素材名自动推荐")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--compare", action="store_true", help="跑防抖开关对照实验")
    p.add_argument("--no-render", action="store_true", help="跳过视频渲染")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--config", default=None)


def _add_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("image", help="待评分图像路径")
    p.add_argument("--backend", default=None, choices=["auto", "rule", "yolo"])
    p.add_argument("--output", default=None, help="可视化输出路径（默认 outputs/score/）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出评分详情")
    p.add_argument("--config", default=None)


def _add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="开发模式热重载")
    p.add_argument("--config", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aicg",
        description="AI 实时构图指导 Agent —— 命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="提示：未安装为包时可用 `python -m aicg.cli <子命令>`。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="录屏轨：引导循环 + 标注视频 + 指标")
    _add_demo_args(p_demo)

    p_score = sub.add_parser("score", help="单图构图打分 + 可视化")
    _add_score_args(p_score)

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务（Web 轨）")
    _add_serve_args(p_serve)

    p_fix = sub.add_parser("make-fixtures", help="生成测试夹具（双源分流）")
    p_fix.add_argument("--force", action="store_true")
    p_fix.add_argument("--list", action="store_true")

    p_bench = sub.add_parser("bench", help="延迟基准（NFR-P1）")
    p_bench.add_argument("--frames", type=int, default=100, help="每轮帧数")
    p_bench.add_argument("--repeat", type=int, default=3, help="重复轮数（取中位数）")
    p_bench.add_argument("--backend", default=None, choices=["yolo", "rule"])
    p_bench.add_argument("--device", default=None, help="强制推理设备（cpu/cuda）")
    p_bench.add_argument("--out", default=None, help="报告输出路径")

    p_eval = sub.add_parser("eval", help="构图评分相关性评估")
    p_eval.add_argument("--annotations", default=None, help="标注文件路径（JSON）")
    p_eval.add_argument("--config", default=None)

    p_acc = sub.add_parser("accept", help="端到端验收报告（延迟/抖动/SRCC 一次跑完）")
    p_acc.add_argument("--backend", default="yolo", choices=["yolo", "rule", "auto"])
    p_acc.add_argument("--debounce-source", default="handheld_jitter.mp4",
                       help="抖动对照夹具")
    p_acc.add_argument("--debounce-backend", default=None, choices=["yolo", "rule"])
    p_acc.add_argument("--quick", action="store_true", help="跳过 SRCC")
    p_acc.add_argument("--json", action="store_true", help="额外输出机器可读 JSON")

    sub.add_parser("doctor", help="环境自检：依赖 / 权重 / 后端可用性")

    return parser


# ----------------------------------------------------------------------
def _cmd_demo(args: argparse.Namespace) -> int:
    """转调 scripts/run_demo.py（保持单一实现，避免逻辑分叉）。"""
    import runpy

    argv: list[str] = ["run_demo.py", "--source", args.source]
    if args.backend:
        argv += ["--backend", args.backend]
    if args.max_frames:
        argv += ["--max-frames", str(args.max_frames)]
    if args.compare:
        argv.append("--compare")
    if args.no_render:
        argv.append("--no-render")
    if args.quiet:
        argv.append("--quiet")
    if args.config:
        argv += ["--config", args.config]

    script = PROJECT_ROOT / "scripts" / "run_demo.py"
    saved = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = saved
    return 0


def _cmd_make_fixtures(args: argparse.Namespace) -> int:
    import runpy

    argv = ["make_fixtures.py"]
    if args.force:
        argv.append("--force")
    if args.list:
        argv.append("--list")
    script = PROJECT_ROOT / "scripts" / "make_fixtures.py"
    saved = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = saved
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import runpy

    argv = ["benchmark_latency.py",
            "--frames", str(args.frames),
            "--repeat", str(args.repeat)]
    if args.backend:
        argv += ["--backend", args.backend]
    if args.device:
        argv += ["--device", args.device]
    if args.out:
        argv += ["--out", args.out]
    script = PROJECT_ROOT / "scripts" / "benchmark_latency.py"
    saved = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = saved
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("缺少依赖 uvicorn。请先 pip install -r requirements.txt", file=sys.stderr)
        return 2

    if args.config:
        import os

        os.environ["AICG_CONFIG"] = args.config

    uvicorn.run(
        "aicg.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """单图打分：感知 → 构图 → 差分量，并渲染结果。"""
    import json

    import cv2

    from aicg.camera.base import Frame
    from aicg.perception import infer_backend_for_source, perception_from_settings
    from aicg.pipeline import FrameProcessor
    from aicg.settings import load_settings
    from aicg.viz import SnapshotRenderer

    img_path = args.image
    p = Path(img_path)

    # 友好校验：文件不存在 / 给的是视频，都不该抛 traceback。
    if not p.exists():
        print(f"图像不存在: {img_path}", file=sys.stderr)
        return 2
    if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        print(
            f"这是视频文件（{p.suffix}），score 只处理单张图像。\n"
            f"要看视频的逐帧引导，请用: aicg demo --source {img_path}",
            file=sys.stderr,
        )
        return 2

    import numpy as np

    img = cv2.imread(str(p))
    if img is None:
        # Windows 非 ASCII 路径下 cv2.imread 会失败，用 fromfile 兜底。
        # fromfile 在路径不存在时会抛异常，故先做存在性检查（已在上方完成）。
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError as e:
            print(f"读取图像失败: {img_path} ({e})", file=sys.stderr)
            return 2
    if img is None:
        print(f"无法解码为图像（格式不支持或文件损坏）: {img_path}", file=sys.stderr)
        return 2

    backend = args.backend or infer_backend_for_source(img_path) or "auto"
    cfg = load_settings(args.config, {"perception.backend": backend} if backend != "auto" else None)
    # 与真实引导链路保持一致的降采样：否则 CLI 的耗时/坐标准确度
    # 都不代表线上表现（线上按 frame_downsample_width 缩放后才送模型）。
    from aicg.utils.image import resize_keep_aspect

    original_shape = img.shape[:2]
    img = resize_keep_aspect(img, cfg.pipeline.frame_downsample_width)
    used_shape = img.shape[:2]

    perception = perception_from_settings(cfg)
    processor = FrameProcessor(perception, cfg)

    f = Frame(image=img, frame_id=0, timestamp_ms=0)
    from aicg.pipeline.frame_processor import FrameContext

    # 预热：YOLO 首次推理含 CUDA 上下文建立与算子编译，实测可达 7s。
    # score 是"一次调用一次输出"的命令，若把这 7s 算进报告，用户会
    # 误以为单帧要 7 秒。预热后计时才反映真实的稳态性能。
    #
    # 必须传入真实尺寸：推理引擎按输入尺寸规划，尺寸不匹配会再规划一次
    # （实测 480x640 预热后跑 480x270 仍要 181ms，传对尺寸后降到 ~25ms）。
    processor.warmup(used_shape)

    snap = processor.process(f, FrameContext())

    if args.json:
        from aicg.api.mappers import snapshot_to_response

        print(json.dumps(snapshot_to_response(snap), ensure_ascii=False, indent=2))
    else:
        print("=" * 62)
        print("  构图评分结果")
        print("=" * 62)
        print(f"  感知后端 : {perception.name}")
        print(f"  图像尺寸 : {original_shape[1]}x{original_shape[0]} → "
              f"推理 {used_shape[1]}x{used_shape[0]}（按配置降采样）")
        subj = snap.perception.primary_subject
        if subj is not None:
            print(f"  主体     : {subj.label} ({subj.confidence:.2f})")
        else:
            print("  主体     : 未检出（走降级路径）")
        comp = snap.composition
        print(f"  构图评分 : {comp.composition_score:.1f} / 100")
        print(f"  构图模式 : {comp.pattern_label}")
        print(f"  景别     : {comp.shot_size_label}")
        print(f"  建议     : {snap.command.command.magnitude_text}")
        print(f"  动作     : {snap.command.command.action.value}")
        print("\n  子项得分:")
        for k, v in comp.sub_scores.items():
            bar = "#" * int(v * 24)
            print(f"    {k:<18s} {v:5.3f}  {bar}")
        if comp.rule_violations:
            print("\n  构图问题:")
            for v in comp.rule_violations:
                # 枚举用 .value 输出，避免打印出 "Severity.WARN" 这种 repr
                sev = v.severity.value if hasattr(v.severity, "value") else str(v.severity)
                rule = v.rule.value if hasattr(v.rule, "value") else str(v.rule)
                print(f"    [{sev}] {rule}: {v.detail}")
        print(f"\n  单帧用时 : {snap.latency.total_ms:.1f} ms "
              f"(感知 {snap.latency.perception_ms:.1f} / 决策 {snap.latency.decision_ms:.1f})")
        print("  注：已预热，此处为稳态延迟，不含模型冷启动开销。")

    # 渲染可视化
    out_dir = PROJECT_ROOT / "outputs" / "score"
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = SnapshotRenderer(scale=1.0)
    canvas = renderer.render(img, snap)
    out_path = Path(args.output) if args.output else out_dir / f"{Path(img_path).stem}_scored.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    if not args.json:
        print(f"\n  可视化输出: {out_path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "scripts" / "evaluate_composition.py"
    if not script.exists():
        print(
            "评估脚本尚不存在：scripts/evaluate_composition.py\n"
            "该脚本需要人工标注的构图质量排序数据（SRCC 相关性基线）。\n"
            "当前里程碑（M3）尚未产出该数据集 —— 详见《测试与验收.md》。",
            file=sys.stderr,
        )
        return 2

    import runpy

    argv = ["evaluate_composition.py"]
    if args.annotations:
        argv += ["--annotations", args.annotations]
    if args.config:
        argv += ["--config", args.config]
    saved = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = saved
    return 0


def _cmd_accept(args: argparse.Namespace) -> int:
    """转调 scripts/acceptance_report.py（M3 端到端验收，NFR-O3）。"""
    script = PROJECT_ROOT / "scripts" / "acceptance_report.py"
    if not script.exists():
        print(f"验收脚本不存在：{script}", file=sys.stderr)
        return 2

    import runpy

    argv = ["acceptance_report.py", "--backend", args.backend,
            "--debounce-source", args.debounce_source]
    if args.debounce_backend:
        argv += ["--debounce-backend", args.debounce_backend]
    if args.quick:
        argv.append("--quick")
    if args.json:
        argv.append("--json")
    saved = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        sys.argv = saved
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """环境自检。逐项报告并给出可执行建议，而不是只抛一个异常。"""
    print("=" * 62)
    print("  环境自检")
    print("=" * 62)

    ok = True

    # 1. Python 版本
    v = sys.version_info
    mark = "OK " if v >= (3, 11) else "!! "
    if v < (3, 11):
        ok = False
    print(f"  [{mark}] Python {v.major}.{v.minor}.{v.micro}（要求 >=3.11）")

    # 2. 核心依赖
    deps = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("pydantic", "pydantic"),
        ("yaml", "pyyaml"),
        ("loguru", "loguru"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]
    for mod, pkg in deps:
        try:
            __import__(mod)
            print(f"  [OK ] {pkg}")
        except ImportError:
            print(f"  [!! ] {pkg} 缺失 —— pip install {pkg}")
            ok = False

    # 3. 可选依赖
    for mod, pkg in [("torch", "torch"), ("ultralytics", "ultralytics")]:
        try:
            m = __import__(mod)
            print(f"  [OK ] {pkg} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  [-- ] {pkg} 未安装（YOLO 后端将自动降级为 rule）")

    # 4. YOLO 权重
    weights = PROJECT_ROOT / "models" / "yolov8n-seg.pt"
    if weights.exists():
        size_mb = weights.stat().st_size / 1024 / 1024
        # 校验是否为 zip/pt 格式（防下载到 HTML 错误页）
        with weights.open("rb") as fh:
            magic = fh.read(2)
        valid = magic in (b"PK", b"\x80\x02") or magic[:1] == b"\x80"
        print(f"  [{'OK ' if valid else '!! '}] YOLO 权重 {weights.name} ({size_mb:.1f} MB){'' if valid else ' 格式可疑'}")
        if not valid:
            ok = False
    else:
        print("  [-- ] YOLO 权重不存在 —— 运行 python scripts/download_weights.py")
        print("         （不影响运行：感知会自动降级为 rule 后端）")

    # 5. 感知后端可用性
    try:
        from aicg.perception import perception_from_settings
        from aicg.settings import load_settings

        cfg = load_settings()
        p = perception_from_settings(cfg)
        print(f"  [OK ] 感知后端就绪: {p.name}（配置请求: {cfg.perception.effective_backend()}）")
    except Exception as e:  # noqa: BLE001
        print(f"  [!! ] 感知后端构建失败: {type(e).__name__}: {e}")
        ok = False

    # 6. 显卡
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  [OK ] CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("  [-- ] CUDA 不可用（CPU 推理，延迟会显著上升）")
    except ImportError:
        pass

    # 7. 测试夹具
    fx = PROJECT_ROOT / "tests" / "fixtures"
    missing = [n for n in ("walk_towards.mp4", "handheld_jitter.mp4", "real_photo_zoom.mp4")
               if not (fx / n).exists()]
    if missing:
        print(f"  [-- ] 夹具缺失 {len(missing)} 个 —— 运行 aicg make-fixtures")
    else:
        print("  [OK ] 测试夹具齐备（3/3）")

    print("=" * 62)
    print("  结论: " + ("环境正常，可运行" if ok else "存在问题，请按上方提示修复"))
    return 0 if ok else 1


_DISPATCH = {
    "demo": _cmd_demo,
    "score": _cmd_score,
    "serve": _cmd_serve,
    "make-fixtures": _cmd_make_fixtures,
    "bench": _cmd_bench,
    "eval": _cmd_eval,
    "accept": _cmd_accept,
    "doctor": _cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
