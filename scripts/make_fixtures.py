#!/usr/bin/env python
"""生成测试夹具视频（双源分流）。

对应文档：《测试与验收.md》§3 —— 测试素材分层
对应需求：FR-02、FR-06

**为什么需要"双源分流"？**

这是本项目一个必须诚实交代的工程约束，来自实测而非猜测：

============== ================== ====================================
后端           可检出的素材        原因
============== ================== ====================================
``rule``       合成人形（椭圆/火柴人） 规则后端基于肤色+形状先验，不依赖
                                  学习特征，因此对"代码画出来的人"有效
``yolo``       **仅真实照片**        YOLOv8n-seg 在真实照片上训练，
                                  对合成图元（无论多写实）零响应 —— 实测
                                  纯椭圆、火柴人、带渐变背景+噪声的写实化
                                  人形，全部返回 0 主体
============== ================== ====================================

因此本脚本产出两类素材，各自服务一个后端：

1. **合成运镜序列**（``walk_towards`` / ``handheld_jitter``）
   —— 像素级可控，用于**规则后端**下的防抖对照实验（FR-06 自证）。
   主体用代码绘制，位置/尺寸逐帧精确已知，因此"该不该切换指令"
   有唯一正确答案，能严丝合缝地衡量防抖是否既压抖动又不误杀意图。

2. **真实照片运镜序列**（``real_photo_*``）
   —— 以真实照片为底，通过**裁切+缩放**模拟相机推拉/抖动。
   像素来自真实成像，YOLO 可正常检出，用于**真实模型后端的
   端到端验证**（证明 yolo 链路在真实输入上能跑通全流程）。

两类素材不是互相替代，而是覆盖两条不同链路 —— 这正是"双源分流"的含义。

用法::

    python scripts/make_fixtures.py            # 全部生成（幂等，已存在则跳过）
    python scripts/make_fixtures.py --force    # 强制重建
    python scripts/make_fixtures.py --list     # 只列出现状
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"

W, H = 640, 480
FPS = 12.0


# ==========================================================================
# 一、合成人形绘制（服务 rule 后端）
# ==========================================================================
def _draw_synthetic_person(
    canvas: np.ndarray,
    cx: float,
    height_ratio: float,
    skin: tuple[int, int, int] = (140, 165, 200),
    cloth: tuple[int, int, int] = (70, 90, 160),
) -> tuple[int, int, int, int]:
    """在画布上绘制一个合成人形，返回归一化 bbox。

    为什么用"头+躯干+四肢"而非纯椭圆：
        规则后端靠肤色面积与纵横比筛选主体。纯椭圆缺少纵向结构，
        纵横比接近 1，容易被后端的 aspect 过滤掉。加入四肢后
        bbox 明显偏高瘦，更接近真实人像的形态分布。

    皮肤色取 (140,165,200)（BGR）：这是特意选的 —— 落在规则后端
    的肤色判定区间内（YCbCr 的 Cr 分量偏高），保证能被检出。
    """
    img_h, img_w = canvas.shape[:2]
    bh = int(img_h * height_ratio)
    bw = max(6, int(bh * 0.38))
    cx_px = int(img_w * cx)
    top = int(img_h - bh)  # 脚踩地面，头部随高度上移

    # 躯干（矩形，占身体中段 55%）
    torso_top = top + int(bh * 0.14)
    torso_bot = top + int(bh * 0.58)
    cv2.rectangle(
        canvas,
        (cx_px - bw // 2, torso_top),
        (cx_px + bw // 2, torso_bot),
        cloth,
        -1,
    )

    # 头（肤色圆，半径约为身宽 0.42）
    head_r = max(3, int(bw * 0.42))
    head_cy = top + head_r + int(bh * 0.02)
    cv2.circle(canvas, (cx_px, head_cy), head_r, skin, -1)

    # 双臂（肤色，垂于体侧）
    arm_w = max(2, int(bw * 0.18))
    cv2.rectangle(
        canvas,
        (cx_px - bw // 2 - arm_w, torso_top + int(bh * 0.02)),
        (cx_px - bw // 2, torso_bot),
        skin,
        -1,
    )
    cv2.rectangle(
        canvas,
        (cx_px + bw // 2, torso_top + int(bh * 0.02)),
        (cx_px + bw // 2 + arm_w, torso_bot),
        skin,
        -1,
    )

    # 双腿（深色，占下段 42%）
    leg_w = max(2, int(bw * 0.32))
    cv2.rectangle(
        canvas,
        (cx_px - bw // 2, torso_bot),
        (cx_px - bw // 2 + leg_w, top + bh),
        (45, 50, 70),
        -1,
    )
    cv2.rectangle(
        canvas,
        (cx_px + bw // 2 - leg_w, torso_bot),
        (cx_px + bw // 2, top + bh),
        (45, 50, 70),
        -1,
    )

    # 归一化 bbox（含头部与四肢的完整外接框）
    x1 = (cx_px - bw // 2 - arm_w) / img_w
    x2 = (cx_px + bw // 2 + arm_w) / img_w
    y1 = head_cy - head_r
    y1 = max(0, y1) / img_h
    y2 = min(img_h - 1, top + bh) / img_h
    return (x1, y1, x2, y2)


def _make_background() -> np.ndarray:
    """构造三分法参考背景（天空/地面分割线 + 网格），便于目测构图。"""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    horizon = int(H * 0.62)
    canvas[:horizon] = (150, 140, 120)  # 天空（偏灰蓝）
    canvas[horizon:] = (80, 95, 75)  # 地面（偏绿）
    # 三分线（淡）
    for i in (1, 2):
        y = int(H * i / 3)
        x = int(W * i / 3)
        cv2.line(canvas, (0, y), (W, y), (105, 105, 105), 1)
        cv2.line(canvas, (x, 0), (x, H), (105, 105, 105), 1)
    return canvas


def gen_walk_towards(path: Path, n_frames: int = 90) -> None:
    """单调走近：主体高度 0.28 → 0.92 线性增长，横向居中。

    用途：验证"真实意图变化必须被保留"。指令应沿
    move_closer → hold → move_back 单向演进，且切换次数应较少
    （因为变化是单调的，不存在抖动）。
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    try:
        for i in range(n_frames):
            t = i / max(1, n_frames - 1)
            canvas = _make_background()
            # 高度 0.28 → 0.92
            h_ratio = 0.28 + (0.92 - 0.28) * t
            _draw_synthetic_person(canvas, cx=0.50, height_ratio=h_ratio)
            writer.write(canvas)
    finally:
        writer.release()


def gen_handheld_jitter(path: Path, n_frames: int = 240) -> None:
    """手持抖动：主体在两个"应给建议"的区域之间高频大幅震荡。

    用途：验证防抖价值 —— 这是 FR-06 的**核心自证素材**。

    **为什么要"大幅"而非"小幅"抖动（重要，来自实测）**：

    第一版设计让主体在理想位置 0.68 附近小幅抖动（σ≈0.055），
    结果是 0% 降幅、全程 `hold`。原因在于决策层的容差结构：

    - 尺度：偏离目标占比 0.68 需超过 ``_OCCUPANCY_TOLERANCE=0.08``，
      即主体占高需 <0.60 或 >0.76 才会给"前进/后退"建议；
    - 横向：三处合理落点为 {0.333, 0.5, 0.667}，且需距最近落点
      超过 ``_POSITION_TOLERANCE=0.07`` 才给"左移/右移"建议。
      这使画面中部存在大片"无论怎么动都给 hold"的静默区。

    小幅抖动全部落在静默区内 → 决策层本就无指令可切 → 防抖无事可做。
    这**不是**防抖无效，而是素材没有制造出"需要被压制的抖动指令"。

    因此本版让主体在两个**明确应给建议**的区间之间震荡：

    - 横向在「偏右需左移」(cx≈0.80) 与「偏左需右移」(cx≈0.20)
      之间快速往复 —— 这两处距最近落点均远超容差；
    - 尺度在 0.52（需靠近）与 0.86（需后退）之间快速往复 ——
      两端均远超占比容差。

    这样未开防抖时会因逐帧抖动而大量切换指令，开启防抖后应被吸收。
    同时叠加**缓慢真实漂移**（左移 → 逐渐转为居中/右移），
    确保防抖**不会**把真实趋势一并抹除 —— 这是"压抖动不误杀意图"
    的双向检验。
    """
    rng = np.random.default_rng(20260928)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    try:
        for i in range(n_frames):
            t = i / max(1, n_frames - 1)

            # --- 横向：高频大摆幅（穿越左/右建议区）+ 缓慢真实漂移 ---
            # 摆动周期约 12 帧，幅度 ±0.30 → cx 在 [0.20, 0.80] 震荡
            swing = 0.30 * np.sin(2 * np.pi * i / 12.0)
            jitter_x = rng.normal(0.0, 0.035)
            # 漂移：从 -0.05 线性走到 +0.05（真实意图缓慢转向）
            drift_x = -0.05 + 0.10 * t
            cx = float(np.clip(0.50 + swing + jitter_x + drift_x, 0.12, 0.88))

            # --- 尺度：高频大摆幅（穿越靠近/后退建议区）+ 缓慢真实漂移 ---
            swing_h = 0.17 * np.sin(2 * np.pi * i / 12.0 + 1.1)
            jitter_h = rng.normal(0.0, 0.022)
            drift_h = -0.03 + 0.06 * t
            h_ratio = float(np.clip(0.68 + swing_h + jitter_h + drift_h, 0.40, 0.95))

            canvas = _make_background()
            _draw_synthetic_person(canvas, cx=cx, height_ratio=h_ratio)
            writer.write(canvas)
    finally:
        writer.release()


# ==========================================================================
# 二、真实照片运镜（服务 yolo 后端）
# ==========================================================================
def _find_real_photo() -> Path | None:
    """定位一张含人物的真实照片。

    优先使用 ultralytics 包内自带的 ``zidane.jpg``（Apache-2.0 许可，
    随依赖分发，可离线获取）。该图含 2 名人物，其中一人已近似落在
    三分线，非常适合验证构图评分。

    找不到时返回 None —— 调用方据此跳过真实素材生成而非报错。
    """
    candidates: list[Path] = []
    try:
        import ultralytics

        assets = Path(ultralytics.__file__).parent / "assets"
        candidates += [assets / "zidane.jpg", assets / "bus.jpg"]
    except Exception:  # noqa: BLE001
        pass
    for p in candidates:
        if p.exists():
            return p
    return None


def _crop_scale(
    photo: np.ndarray, zoom: float, pan_x: float, pan_y: float
) -> np.ndarray:
    """在真实照片上模拟"镜头推拉 + 平移"。

    zoom > 1 表示推近（等价于人物在画面中变大）；
    pan_x/pan_y ∈ [-0.5, 0.5] 表示相对可平移范围的横向/纵向偏移。

    实现为**裁切再缩放回原尺寸** —— 像素始终来自真实成像，
    因此 YOLO 的检出不受影响（这正是本素材存在的意义）。
    """
    ph, pw = photo.shape[:2]
    ch = int(ph / zoom)
    cw = int(pw / zoom)
    # 平移范围：裁切窗口可在照片内左右/上下移动的余量
    max_dx = pw - cw
    max_dy = ph - ch
    x0 = int(np.clip(max_dx * (0.5 + pan_x), 0, max_dx))
    y0 = int(np.clip(max_dy * (0.5 + pan_y), 0, max_dy))
    crop = photo[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


def gen_real_photo_zoom(path: Path, n_frames: int = 120) -> tuple[Path, int] | None:
    """真实照片推拉序列：zoom 在 1.0 → 1.9 之间往复。

    用途：真实模型（YOLO）后端的端到端验证。主体由真实照片提供，
    YOLO 可稳定检出，从而让"感知 → 构图 → 指令 → 快照"整条链路
    在真实输入上得到验证。

    设计意图：zoom 往复意味着人物在画面中的占比周期性变大变小，
    指令应在 move_closer / hold / move_back 之间真实往复 ——
    这检验的是**链路连通性**，而非防抖（防抖由合成素材负责）。
    """
    photo_path = _find_real_photo()
    if photo_path is None:
        return None
    photo = cv2.imread(str(photo_path))
    if photo is None:
        return None

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    try:
        for i in range(n_frames):
            t = i / max(1, n_frames - 1)
            # 0→1→0 的三角波，往复两次
            tri = abs(((t * 2.0) % 1.0) * 2.0 - 1.0)
            zoom = 1.0 + 0.9 * tri
            pan_x = 0.18 * np.sin(2 * np.pi * t)
            pan_y = 0.05 * np.cos(2 * np.pi * t)
            frame = _crop_scale(photo, zoom, float(pan_x), float(pan_y))
            writer.write(frame)
    finally:
        writer.release()
    return photo_path, n_frames


# ==========================================================================
# 三、驱动
# ==========================================================================
SPECS = [
    ("walk_towards.mp4", gen_walk_towards, "rule", "单调走近，验证意图变化被保留"),
    ("handheld_jitter.mp4", gen_handheld_jitter, "rule", "手持抖动，验证防抖价值（FR-06 核心自证）"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成测试夹具视频")
    ap.add_argument("--force", action="store_true", help="已存在也重建")
    ap.add_argument("--list", action="store_true", help="只列出现状")
    args = ap.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print(f"夹具目录: {FIXTURE_DIR}")
        for p in sorted(FIXTURE_DIR.iterdir()):
            if p.suffix in (".mp4", ".jpg", ".md"):
                kb = p.stat().st_size / 1024
                print(f"  {p.name:<28s} {kb:>9.1f} KB")
        return 0

    made: list[str] = []
    skipped: list[str] = []

    # 1) 合成素材（rule 后端）
    for name, fn, backend, purpose in SPECS:
        path = FIXTURE_DIR / name
        if path.exists() and not args.force:
            skipped.append(f"{name} (已存在，跳过)")
            continue
        fn(path)
        made.append(f"{name}  [后端={backend}]  {purpose}")

    # 2) 真实照片素材（yolo 后端）
    real_path = FIXTURE_DIR / "real_photo_zoom.mp4"
    if real_path.exists() and not args.force:
        skipped.append("real_photo_zoom.mp4 (已存在，跳过)")
    else:
        out = gen_real_photo_zoom(real_path)
        if out is None:
            print("[警告] 未找到可用的真实照片素材，跳过 real_photo_zoom.mp4")
            print("       请确认 ultralytics 已安装（内含 assets/zidane.jpg）")
        else:
            src, n = out
            made.append(
                f"real_photo_zoom.mp4  [后端=yolo]  真实照片推拉，验证 YOLO 端到端连通性  (源: {src.name}, {n}帧)"
            )

    if made:
        print("已生成:")
        for m in made:
            print(f"  + {m}")
    if skipped:
        print("已跳过:")
        for s in skipped:
            print(f"  - {s}")
    print(f"\n夹具目录: {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
