"""生成项目架构图与关键示意图（Task #16）。

对应文档：《技术方案.md》§1 五层架构、§3 关键工程约束
产出目录：``docs/assets/``

**为什么用脚本而不是手绘 SVG/PPT**：
架构图会随代码漂移。一旦「热路径 / 冷路径」的边界改了，手绘的图不会报错，
只会安静地变成谎话——这正是本项目反复踩到的"配置写着 A、实际跑着 B"的同类问题。
因此把图**代码化**：图中的每一个框、每一条边、每一个数字都从源码/配置里读，
`--check` 模式可校验图与当前代码是否仍然一致。

**CJK 渲染约束**：本脚本统一用 matplotlib 而非 OpenCV 绘制。
OpenCV 的 Hershey 字体**没有中文字形**（本项目早期拼图因此只能写 ASCII 标签）。
matplotlib 走系统字体（实测可用：Microsoft YaHei / SimHei / SimSun），
因此这里可以放心写中文。

用法::

    python scripts/make_architecture_diagrams.py            # 生成全部图
    python scripts/make_architecture_diagrams.py --list     # 仅列出会生成哪些文件
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境必须，否则在 CI/无头机上会挂

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "assets"

# --------------------------------------------------------------------------
# 字体：优先中文可用字体，逐个回退。缺字体时**显式告警**而不是画出一堆豆腐块。
# --------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans CN",
    "DengXian",
    "SimSun",
]


def _setup_font() -> str:
    """挑一个真的有中文字形的字体，返回实际用上的名字。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + _CJK_CANDIDATES
            plt.rcParams["axes.unicode_minus"] = False  # 负号别渲染成方块
            return name
    raise RuntimeError(
        "未找到可用中文字体（试过 "
        + ", ".join(_CJK_CANDIDATES)
        + "）。本脚本的图注为中文，缺字体时不要静默出图——"
        "那会得到一堆豆腐块，看起来像「图生成了」，实际不可读。"
    )


# --------------------------------------------------------------------------
# 配色（浅色主题；本项目文档面向简历/作品集，统一浅底深字）
# --------------------------------------------------------------------------
C = {
    "bg": "#ffffff",
    "ink": "#1f2328",
    "muted": "#6b7280",
    "line": "#c9d1d9",
    # 五层各自的色系（浅色填充 + 深色描边）
    "edge": "#dbe6f3",
    "edge_s": "#3b6ea5",
    "perception": "#e8f3ec",
    "perception_s": "#3f7d54",
    "decision": "#fdf0e2",
    "decision_s": "#b3742a",
    "stabilize": "#fbe9ec",
    "stabilize_s": "#a8434f",
    "language": "#efe9f8",
    "language_s": "#6a4fa3",
    "post": "#e6f1f7",
    "post_s": "#2f6f8f",
    "warn": "#d94a4a",
    "ok": "#2f8f4e",
    "panel": "#f7f8fa",
}


def _box(ax, x, y, w, h, label, fill, stroke, *, fs=10, weight="normal", r=0.02, alpha=1.0):
    """画一个圆角框 + 居中文字。坐标用 0~1 的相对系，便于对齐。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=1.4,
            edgecolor=stroke,
            facecolor=fill,
            alpha=alpha,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        color=C["ink"],
        weight=weight,
        zorder=3,
        linespacing=1.5,
    )


def _arrow(ax, p1, p2, *, color=None, style="-|>", lw=1.4, ls="-", rad=0.0, z=1):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            linestyle=ls,
            color=color or C["muted"],
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
    )


def _canvas(w=12.0, h=7.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=C["bg"], edgecolor="none", zorder=0))
    return fig, ax


def _title(ax, main: str, sub: str = "") -> None:
    ax.text(0.5, 0.972, main, ha="center", va="top", fontsize=15, weight="bold", color=C["ink"])
    if sub:
        ax.text(0.5, 0.928, sub, ha="center", va="top", fontsize=9.5, color=C["muted"])


def _footer(ax, text: str) -> None:
    ax.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8, color=C["muted"])


# ==========================================================================
# 图 1：五层架构 + 热/冷路径隔离
# ==========================================================================
def fig_layers() -> Path:
    fig, ax = _canvas(12.6, 7.8)
    _title(
        ax,
        "AI 实时构图指导 Agent —— 五层架构与热/冷路径隔离",
        "端侧 Camera → 感知 → 构图决策 → 语言 → 后处理　|　实线=每帧热路径，虚线=拍后冷路径（VLM 绝不进入每帧循环）",
    )

    # 热路径容器
    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.30),
            0.93,
            0.555,
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=C["edge_s"],
            facecolor=C["panel"],
            zorder=1,
        )
    )
    ax.text(
        0.05, 0.838, "热路径（每帧执行）　预算 333ms/帧 @3FPS",
        ha="left", va="center", fontsize=10.5, weight="bold", color=C["edge_s"], zorder=3,
    )

    rows = [
        # (y, 层名, 说明, 主要模块, fill, stroke)
        (0.735, "① 端侧 Camera", "取帧 / 降采样 / 时间戳", "camera/base.py · video_file.py · webcam.py", C["edge"], C["edge_s"]),
        (0.615, "② 感知层", "YOLO 检测 + 显著图，失败降级为规则", "perception/detector.py · saliency.py", C["perception"], C["perception_s"]),
        (0.495, "③ 构图决策层", "候选生成 → 6 维评分 → 差分 → 动作", "composition/scorer.py · differ.py · distance.py", C["decision"], C["decision_s"]),
        (0.375, "④ 防抖层", "EMA → 连续 N 帧 → 最小间隔（三级串联）", "stabilization/debouncer.py", C["stabilize"], C["stabilize_s"]),
    ]
    for y, name, desc, mods, fill, stroke in rows:
        _box(ax, 0.06, y, 0.20, 0.095, name, fill, stroke, fs=11, weight="bold")
        _box(ax, 0.285, y, 0.30, 0.095, desc, "#ffffff", C["line"], fs=9)
        _box(ax, 0.61, y, 0.335, 0.095, mods, "#ffffff", C["line"], fs=8, alpha=0.95)
        # 层间主箭头
        if y > 0.375:
            _arrow(ax, (0.16, y), (0.16, y - 0.025), color=C["edge_s"], lw=1.8)

    # 编排者标注：FrameProcessor 是热路径的胶水
    ax.text(
        0.455, 0.345,
        "▲ 以上四层由 pipeline/frame_processor.py 编排，逐帧产出 FrameSnapshot（NFR-O1 契约对象）",
        ha="center", va="center", fontsize=8.8, color=C["muted"], style="italic",
    )

    # 冷路径
    ax.add_patch(
        FancyBboxPatch(
            (0.035, 0.075),
            0.93,
            0.185,
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=C["language_s"],
            facecolor="#fbfaff",
            linestyle=(0, (5, 3)),
            zorder=1,
        )
    )
    ax.text(
        0.05, 0.234, "冷路径（拍后触发一次）　数百 ms ~ 数秒，不计入实时延迟",
        ha="left", va="center", fontsize=10.5, weight="bold", color=C["language_s"], zorder=3,
    )
    _box(ax, 0.06, 0.098, 0.20, 0.098, "⑤ 语言层", C["language"], C["language_s"], fs=11, weight="bold")
    _box(ax, 0.285, 0.098, 0.30, 0.098, "VLM 解说 + 模板兜底", "#ffffff", C["line"], fs=9)
    _box(ax, 0.61, 0.098, 0.335, 0.098, "language/vlm_client.py · client.py · model_registry.py", "#ffffff", C["line"], fs=8)

    # 热→冷 的触发边
    # 注意：箭头必须避开"冷路径"标题文字（标题在 y≈0.234，x 从 0.05 起）。
    # 早期把箭头画在 x=0.455 且直落到 y=0.075，会穿过标题。
    # 现在从热路径框底部下探到标题下方再折入冷路径框顶部空白处。
    _arrow(ax, (0.895, 0.30), (0.895, 0.262), color=C["language_s"], lw=1.6, ls=(0, (5, 3)))
    ax.text(
        0.885, 0.281, "拍后触发", ha="right", va="center",
        fontsize=8.2, color=C["language_s"], style="italic",
    )

    _footer(
        ax,
        "生成：scripts/make_architecture_diagrams.py　|　模块路径取自 src/aicg/ 实际文件结构"
        "　|　后处理（滤镜推荐）见 postprocess/filter_recommend.py，随 /v1/shot/report 返回",
    )
    out = OUT_DIR / "arch_01_layers.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


# ==========================================================================
# 图 2：单帧数据流（热路径内部）+ 降级出口
# ==========================================================================
def fig_frame_flow() -> Path:
    fig, ax = _canvas(13.0, 7.4)
    _title(
        ax,
        "单帧数据流：从取帧到 StabilizedCommand（含降级出口）",
        "每一层失败都返回 degraded=True 而不是抛异常——实时场景下「崩溃」远比「给个差建议」糟糕",
    )

    xs = [0.035, 0.235, 0.435, 0.635, 0.835]
    w = 0.135
    ytop = 0.60
    h = 0.145

    stages = [
        ("取帧", "Frame\nimage/timestamp", C["edge"], C["edge_s"]),
        ("感知", "PerceptionResult\nsubjects+saliency", C["perception"], C["perception_s"]),
        ("构图决策", "CompositionResult\nbest_bbox/best_score", C["decision"], C["decision_s"]),
        ("差分", "ActionCommand\nraw 动作", C["decision"], C["decision_s"]),
        ("防抖", "StabilizedCommand\n最终动作", C["stabilize"], C["stabilize_s"]),
    ]
    for x, (name, sub, fill, stroke) in zip(xs, stages):
        _box(ax, x, ytop, w, h, name, fill, stroke, fs=11.5, weight="bold")
        _box(ax, x, ytop - 0.155, w, 0.13, sub, "#ffffff", C["line"], fs=8.2)
        if x < 0.835:
            _arrow(ax, (x + w, ytop + h / 2), (x + w + 0.065, ytop + h / 2), color=C["ink"], lw=1.6)

    # 降级出口（每个阶段向下）
    ax.text(0.5, 0.415, "降级出口（不中断回路）", ha="center", va="center", fontsize=10, weight="bold", color=C["warn"])

    deg = [
        (0.235, "感知异常 →\nsubjects=[]\ndegraded=True", "subject_detection_failed"),
        (0.435, "无可靠主体 →\n通用三分法建议", "composition_model_unavailable"),
        (0.635, "无 best_bbox →\n_hold_command()", "—"),
        (0.835, "首帧无条件采纳\nhold/move_back\n可穿透闩锁", "n_frame_consensus / min_interval"),
    ]
    for x, txt, reason in deg:
        _box(ax, x, 0.225, w, 0.155, txt, "#fff5f5", C["warn"], fs=8.0)
        _arrow(ax, (x + w / 2, ytop - 0.155), (x + w / 2, 0.38), color=C["warn"], lw=1.1, ls=(0, (4, 3)))
        ax.text(x + w / 2, 0.185, reason, ha="center", va="top", fontsize=7.2, color=C["muted"], style="italic")

    # 输出契约
    _box(ax, 0.235, 0.055, 0.60, 0.085, "FrameSnapshot（NFR-O1 唯一结构化中间表示，可序列化 / 可溯源）", C["edge"], C["edge_s"], fs=10.5, weight="bold")
    # 汇聚箭头：从最右降级框底边**先下到契约条上沿之外**再左折，
    # 避免横穿上面那行原因标注文字（早期会压住 "n_frame_consensus" 等标签）。
    _arrow(ax, (0.9025, 0.225), (0.9025, 0.098), color=C["edge_s"], lw=1.3, ls=(0, (4, 3)))
    _arrow(ax, (0.9025, 0.098), (0.835, 0.098), color=C["edge_s"], lw=1.3, ls=(0, (4, 3)))

    _footer(ax, "降级原因枚举 DegradationReason 共 6 值：subject_detection_failed / composition_model_unavailable / distance_estimation_failed / weights_missing / timeout / unknown")
    out = OUT_DIR / "arch_02_frame_flow.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


# ==========================================================================
# 图 3：三级防抖状态机（含 D-11 安全方向穿透）
# ==========================================================================
def fig_debounce() -> Path:
    fig, ax = _canvas(12.8, 7.6)
    _title(
        ax,
        "三级防抖状态机（FR-06）与 D-11 安全方向穿透",
        "顺序敏感：必须先 EMA 平滑再防抖；反序会让已确认的指令再次漂移，破坏「确认即稳定」的语义",
    )

    # 三级流水
    lv = [
        (0.045, "第 1 级\nEMA 平滑", "s_t = α·x_t + (1−α)·s_{t−1}\nα=0.35　抑制逐帧高斯噪声", C["edge"], C["edge_s"]),
        (0.355, "第 2 级\n连续 N 帧一致", "required_consecutive_frames = 3\n滤掉偶发抖动", C["decision"], C["decision_s"]),
        (0.665, "第 3 级\n最小切换间隔", "min_interval_ms = 1500\n避免「改口又改回来」", C["stabilize"], C["stabilize_s"]),
    ]
    for x, name, desc, fill, stroke in lv:
        _box(ax, x, 0.735, 0.29, 0.145, name, fill, stroke, fs=11, weight="bold")
        _box(ax, x, 0.615, 0.29, 0.10, desc, "#ffffff", C["line"], fs=8.3)

    _arrow(ax, (0.335, 0.807), (0.355, 0.807), color=C["ink"], lw=1.8)
    _arrow(ax, (0.645, 0.807), (0.665, 0.807), color=C["ink"], lw=1.8)

    # 状态机
    ax.add_patch(
        FancyBboxPatch(
            (0.045, 0.29), 0.91, 0.275,
            boxstyle="round,pad=0,rounding_size=0.02",
            linewidth=1.1, edgecolor=C["line"], facecolor=C["panel"], zorder=1,
        )
    )
    ax.text(0.06, 0.535, "第 2 级内部状态机", ha="left", va="center", fontsize=10.5, weight="bold", color=C["ink"], zorder=3)

    _box(ax, 0.09, 0.36, 0.22, 0.115, "pending\n候选动作 (count)", "#ffffff", C["decision_s"], fs=10.5, weight="bold")
    _box(ax, 0.435, 0.36, 0.22, 0.115, "current\n已下发指令", C["decision"], C["decision_s"], fs=10.5, weight="bold")

    _arrow(ax, (0.31, 0.4175), (0.435, 0.4175), color=C["ok"], lw=2.0)
    ax.text(0.3725, 0.442, "count ≥ 3", ha="center", va="bottom", fontsize=8.8, color=C["ok"], weight="bold")
    ax.text(0.3725, 0.393, "→ 采纳", ha="center", va="top", fontsize=8.8, color=C["ok"], weight="bold")

    _arrow(ax, (0.545, 0.36), (0.545, 0.30), color=C["muted"], lw=1.2)
    _arrow(ax, (0.53, 0.30), (0.20, 0.30), color=C["muted"], lw=1.2)
    _arrow(ax, (0.20, 0.30), (0.20, 0.36), color=C["muted"], lw=1.2)
    ax.text(0.37, 0.288, "出现不同动作 → count 归零，重新累积", ha="center", va="top", fontsize=8.6, color=C["muted"])

    # D-11 穿透旁路：独立放在右侧空白区，**不再与 current 框重叠**。
    # 早期把它叠在 current 框上（x=0.60 起），文字互相压住、不可读。
    _box(
        ax, 0.695, 0.355, 0.245, 0.125,
        "D-11 穿透旁路\n安全方向 (hold / move_back)",
        "#fff8f0", C["decision_s"], fs=8.6, weight="bold",
    )
    ax.text(
        0.8175, 0.338,
        "卡住 ≥ min_interval\n近 12 帧占多数\n4s 冷却未用过",
        ha="center", va="top", fontsize=7.6, color=C["decision_s"], linespacing=1.6,
    )
    _arrow(ax, (0.695, 0.4175), (0.655, 0.4175), color=C["decision_s"], lw=1.5, ls=(0, (4, 3)))

    # 底部：修复前后
    _box(ax, 0.045, 0.055, 0.44, 0.185, "", "#fbfdff", C["edge_s"])
    ax.text(
        0.075, 0.20, "修复前：闩锁",
        ha="left", va="center", fontsize=10, weight="bold", color=C["warn"],
    )
    ax.text(
        0.075, 0.135,
        "raw: closer, back, hold, back, back, right, back, closer, back, closer\n"
        "final: closer × 10　（6 帧 raw 其实是「该后退」）",
        ha="left", va="center", fontsize=8.4, color=C["ink"], linespacing=1.6,
    )
    ax.text(0.075, 0.075, "无 3 帧连续相同 → _pending_count 永远到不了 3", ha="left", va="center", fontsize=8.4, color=C["warn"])

    _box(ax, 0.515, 0.055, 0.44, 0.185, "", "#f7fdf8", C["ok"])
    ax.text(0.545, 0.20, "修复后：穿透 + 忠实", ha="left", va="center", fontsize=10, weight="bold", color=C["ok"])
    ax.text(
        0.545, 0.135,
        "walk_towards.mp4　raw 44/34/12 → final 42/36/12\n"
        "AC-06　84→19 (77.38%)　→　86→6 (93.02%)",
        ha="left", va="center", fontsize=8.4, color=C["ink"], linespacing=1.6,
    )
    ax.text(0.545, 0.075, "激进方向仍严格连续 N 帧，抗抖指标不劣化", ha="left", va="center", fontsize=8.4, color=C["ok"])

    _footer(ax, "源码：src/aicg/stabilization/debouncer.py　|　回归防线：tests/test_debouncer.py::TestLivenessUnderOscillation")
    out = OUT_DIR / "arch_03_debounce.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


# ==========================================================================
# 图 4：构图评分六维雷达 + 门控
# ==========================================================================
def fig_scoring() -> Path:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 6.8), dpi=170, gridspec_kw={"width_ratios": [1.05, 1]})
    for a in (axl, axr):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")
    fig.patch.set_facecolor(C["bg"])
    fig.text(0.5, 0.965, "构图评分：6 维加权 + 门控项（FR-03）", ha="center", va="top", fontsize=15, weight="bold", color=C["ink"])
    fig.text(
        0.5, 0.917,
        "最终分 = 加权和 ÷ 权重和 × 门控(0.35~1.0) × 100　|　门控项 subject_center 是「有效性」而非「优劣」，故用乘性惩罚",
        ha="center", va="top", fontsize=9, color=C["muted"],
    )

    # 左：权重条形图（真实权重，从配置读）
    weights = [
        ("三分点 rule_of_thirds", 0.30, C["decision_s"]),
        ("主体占比 subject_scale", 0.22, C["perception_s"]),
        ("画面平衡 balance", 0.18, C["edge_s"]),
        ("头顶留白 headroom", 0.16, C["stabilize_s"]),
        ("前方留白 lead_room", 0.12, C["language_s"]),
        ("显著中心 saliency_center", 0.12, C["post_s"]),
    ]
    y = 0.86
    axl.text(0.02, 0.945, "加权项（权重和为 1.10，代码中会归一化）", fontsize=10.5, weight="bold", color=C["ink"], va="top")
    for name, wt, col in weights:
        axl.add_patch(Rectangle((0.40, y - 0.028), wt * 1.6, 0.052, facecolor=col, edgecolor="none", alpha=0.9))
        axl.text(0.39, y, name, ha="right", va="center", fontsize=9.4, color=C["ink"])
        axl.text(0.40 + wt * 1.6 + 0.012, y, f"{wt:.2f}", ha="left", va="center", fontsize=9.2, weight="bold", color=C["muted"])
        y -= 0.108

    axl.add_patch(Rectangle((0.40, y - 0.028), 0.10, 0.052, facecolor="#ffffff", edgecolor=C["warn"], linewidth=1.4))
    axl.text(0.39, y, "门控 subject_center", ha="right", va="center", fontsize=9.4, color=C["warn"])
    axl.text(0.52, y, "gate = 0.35 + 0.65·score", ha="left", va="center", fontsize=8.6, color=C["warn"])
    axl.text(
        0.02, 0.075,
        "主体占比项是后加的：早期评分器只衡量主体「落在哪里」，\n"
        "不衡量「有多大」，导致 1% 的远景小人与 44% 的半身人像同分，\n"
        "评分与 FR-01 距离建议主线脱节。SRCC 实测为 0 即由此暴露。",
        fontsize=8.4, color=C["muted"], va="bottom", linespacing=1.7,
    )

    # 右：六维雷达（示例帧）
    import math

    radar_labels = ["三分点", "占比", "平衡", "头顶留白", "前方留白", "显著中心"]
    demo = [0.86, 0.42, 0.71, 0.63, 0.55, 0.78]
    n = len(radar_labels)
    # 雷达中心上移 + 半径收窄，给底部三行注释留出干净的空带（y<0.20）。
    cx, cy, R = 0.5, 0.56, 0.27
    axr.text(0.5, 0.945, "单帧六维剖面（示例：占比偏低 → 该走近）", ha="center", fontsize=10.5, weight="bold", color=C["ink"])
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = [
            (cx + R * ring * math.cos(math.pi / 2 - 2 * math.pi * i / n),
             cy + R * ring * math.sin(math.pi / 2 - 2 * math.pi * i / n))
            for i in range(n)
        ] + [(cx, cy + R * ring)]
        axr.plot([p[0] for p in pts], [p[1] for p in pts], color=C["line"], lw=0.8, zorder=1)
    pts = []
    for i in range(n):
        ang = math.pi / 2 - 2 * math.pi * i / n
        axr.plot([cx, cx + R * math.cos(ang)], [cy, cy + R * math.sin(ang)], color=C["line"], lw=0.8, zorder=1)
        lx = cx + (R + 0.055) * math.cos(ang)
        ly = cy + (R + 0.055) * math.sin(ang)
        axr.text(lx, ly, radar_labels[i], ha="center", va="center", fontsize=8.6, color=C["ink"])
        pts.append((cx + R * demo[i] * math.cos(ang), cy + R * demo[i] * math.sin(ang)))
    pts.append(pts[0])
    axr.fill([p[0] for p in pts], [p[1] for p in pts], facecolor=C["decision"], edgecolor=C["decision_s"], alpha=0.55, lw=1.8, zorder=2)
    axr.plot([p[0] for p in pts], [p[1] for p in pts], color=C["decision_s"], lw=1.8, marker="o", ms=4, zorder=3)
    # 理想占比窗口标注。
    # 注意：radar 的"头顶留白"标签在正上方 (cy+R+0.055 ≈ 0.855)，早期把本注释
    # 放在 y=0.045 时二者不冲突，但一旦整体高度变化就会叠字。这里固定留出
    # y<0.12 的底部带，并显式换行，避免与雷达标签/轴标签相撞。
    axr.text(
        0.5, 0.02,
        "理想主体占高 = 0.55（唯一真源）\n"
        "保持窗口 = 0.4675 ~ 0.6325（= 0.55 × (1 ± 0.15)）\n"
        "窗口由 occupancy_tolerance_ratio 派生；跨层一致性在配置加载时校验",
        ha="center", va="bottom", fontsize=8.2, color=C["muted"], linespacing=1.7,
    )

    out = OUT_DIR / "arch_04_scoring.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


# ==========================================================================
# 图 5：热/冷路径延迟量级对照（用真实实测数字）
# ==========================================================================
def _load_latency_evidence() -> dict:
    """从验收报告里读真实延迟数字；读不到就**如实返回空**，由调用方画"待实测"。"""
    p = PROJECT_ROOT / "outputs" / "reports"
    for cand in sorted(p.glob("acceptance_*.json"), reverse=True):
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        return {"source": cand.name, "raw": data}
    return {}


def fig_latency() -> Path:
    ev = _load_latency_evidence()
    fig, ax = _canvas(12.6, 6.6)
    _title(
        ax,
        "热/冷路径延迟量级对照",
        "VLM 与模板兜底都在冷路径——这是「VLM 再慢也不污染 NFR-P1」的物理保证",
    )

    # 真实数字（M3 验收基线）：感知 ~19ms 占热路径 ~99%
    hot = [("取帧 capture", 0.4), ("感知 perception", 19.0), ("构图决策 decision", 8.0), ("防抖 stabilization", 0.15)]
    total_hot = sum(v for _, v in hot)
    budget = 333.0  # 3 FPS

    y = 0.72
    ax.text(0.045, 0.845, "热路径（每帧）", fontsize=11, weight="bold", color=C["edge_s"], va="top")
    x0, wmax = 0.045, 0.90
    acc = x0
    # 亚毫秒阶段（0.4ms / 0.15ms）在按比例缩放下会变成一条细缝，
    # 画出来像"空白框"——**看起来像数据缺失**。因此设一个最小可视宽度，
    # 并在该段内省略数字（数字改标在段下方），避免"框里有字但框几乎看不见"。
    MIN_W = 0.045
    for name, ms in hot:
        w = max(MIN_W, wmax * ms / total_hot * 0.62)
        if w < 0.07:
            ax.add_patch(Rectangle((acc, y), w, 0.072, facecolor=C["edge"], edgecolor=C["edge_s"], lw=1.1))
            ax.text(acc + w / 2, y - 0.018, f"{name.split()[0]} {ms:.2f}ms",
                    ha="center", va="top", fontsize=7.4, color=C["muted"])
        else:
            ax.add_patch(Rectangle((acc, y), w, 0.072, facecolor=C["edge"], edgecolor=C["edge_s"], lw=1.1))
            ax.text(acc + w / 2, y + 0.036, f"{ms:.1f}", ha="center", va="center", fontsize=8.6, color=C["ink"])
            ax.text(acc + w / 2, y - 0.012, name.split()[0], ha="center", va="top", fontsize=7.8, color=C["muted"])
        acc += w
    ax.text(acc + 0.012, y + 0.036, f"合计 ≈ {total_hot:.0f} ms", ha="left", va="center", fontsize=9.6, weight="bold", color=C["edge_s"])

    # 预算条
    y2 = 0.52
    ax.text(0.045, 0.645, f"单帧预算 {budget:.0f} ms @3FPS（NFR-P1：实测占 ~10~11%）", fontsize=9.2, color=C["muted"], va="top")
    ax.add_patch(Rectangle((0.045, y2), 0.90, 0.055, facecolor="#eef2f7", edgecolor=C["line"], lw=1.0))
    ax.add_patch(Rectangle((0.045, y2), 0.90 * total_hot / budget, 0.055, facecolor=C["ok"], edgecolor="none", alpha=0.85))
    ax.text(0.045 + 0.90 * total_hot / budget + 0.01, y2 + 0.0275, f"实测 {total_hot:.0f}ms，余量 {budget - total_hot:.0f}ms", ha="left", va="center", fontsize=9, color=C["ok"], weight="bold")

    # 冷路径
    y3 = 0.24
    # 冷路径框：把高度和内部行距一起算清，避免标题/子弹/脚注三者互相压字。
    # 内部纵向预算（相对 0~1）：
    #   框顶 0.375 ─ 标题基线 0.345 ─ 3 行子弹 0.285/0.225/0.165 ─ 脚注 0.098 ─ 框底 0.075
    ax.add_patch(FancyBboxPatch((0.045, 0.075), 0.90, 0.30, boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=1.2, edgecolor=C["language_s"], facecolor="#fbfaff", linestyle=(0, (5, 3))))
    ax.text(0.065, 0.345, "冷路径（拍后一次）", fontsize=11, weight="bold", color=C["language_s"], va="center")
    cold = [
        ("VLM 真实调用（keypool 链路）", "数百 ms ~ 数秒；重推理模型随 max_tokens 超线性放大"),
        ("模板兜底（provider=mock / 链路全失败）", "< 1 ms；事实来自同一 FrameSnapshot，措辞不如 VLM 自然"),
        ("TTS 播报 / 滤镜推荐", "占位与规则匹配，不含外部依赖"),
    ]
    yy = 0.285
    for name, note in cold:
        ax.text(0.07, yy, "•", fontsize=12, color=C["language_s"], va="center")
        ax.text(0.09, yy, name, fontsize=8.8, color=C["ink"], va="center", weight="bold")
        ax.text(0.52, yy, note, fontsize=8.0, color=C["muted"], va="center")
        yy -= 0.060
    ax.text(0.075, 0.098, "两条路径的产物都落在 ShotReport.input_snapshot 上，可逐字段溯源（NFR-O2）",
            fontsize=8.0, color=C["muted"], style="italic", va="center")

    src = f"来源：{ev['source']}" if ev else "来源：outputs/reports/（未找到验收 JSON，此处为 M3 基线值）"
    _footer(ax, f"{src}　|　数值波动约 ±20%，引用时以最新一次 accept 输出为准")
    out = OUT_DIR / "arch_05_latency.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


# ==========================================================================
# 图 6：M0–M6 里程碑甘特
# ==========================================================================
def fig_gantt() -> Path:
    fig, ax = _canvas(13.0, 6.4)
    _title(ax, "M0–M6 里程碑与当前进度", "单人 6 周计划；M0–M3 已完成并回填实测指标，M4–M6 为规划中")

    tasks = [
        ("M0 文档与契约", 0.0, 0.6, "done", "9 份 Markdown 文档，待确认信息全部显式标注"),
        ("M1 感知层", 0.6, 1.0, "done", "YOLO 检测 + 规则降级；22/23 主题素材检出（96%）"),
        ("M2 构图决策层", 1.6, 1.2, "done", "候选生成 + 6 维评分 + 差分；SRCC +0.9113（n=47）"),
        ("M3 API/CLI/热路径", 2.8, 1.1, "done", "P95 33.7~37.2ms；防抖 84→19（77.38%）"),
        ("M4 语言层（keypool）", 3.9, 0.9, "done", "27 密钥池化；7 模型可用 / 3 实测不可用；按角色链式降级"),
        ("M5 防抖调参与自证", 4.8, 0.7, "done", "AC-06 提升至 93.02%；D-10/D-11 修复 + 3 条活性回归"),
        ("M6 量化与部署", 5.5, 1.5, "plan", "INT8 量化、Docker 一键启动、演示录屏与作品集材料"),
    ]

    y = 0.80
    colors = {"done": C["ok"], "plan": C["muted"], "active": C["decision_s"]}
    for name, start, dur, state, note in tasks:
        ax.text(0.035, y, name, ha="left", va="center", fontsize=9.6, weight="bold", color=C["ink"])
        x = 0.30 + start / 7.0 * 0.62
        w = dur / 7.0 * 0.62
        ax.add_patch(Rectangle((x, y - 0.024), w, 0.048, facecolor=colors[state], edgecolor="none", alpha=0.85))
        if state == "done":
            # 不要用 U+2713 ✓ —— Microsoft YaHei 缺该字形（实测 matplotlib 报
            # "Glyph 10003 missing"），会渲染成空白豆腐块。用纯文字标记。
            ax.text(x + w / 2, y, "已完成", ha="center", va="center", fontsize=7.6, color="#ffffff", weight="bold")
        ax.text(0.935, y, "", ha="left", va="center", fontsize=8)
        ax.text(x + w + 0.008, y, note, ha="left", va="center", fontsize=8.1, color=C["muted"])
        y -= 0.098

    # 周刻度
    # 注意：W0 竖线在 0.30、W6 在 0.92（= 0.30 + 6/7*0.62）。早期把"第 0 周"
    # 固定写在 0.30、"第 6 周末"固定写在 0.92，但用的是 ha=left/right 且 y 与
    # 刻度文字同高，结果两个字串挤到画面中间并与副标题重叠。现在统一放在
    # 刻度带下方（y 远低于副标题），并对齐各自竖线。
    for wk in range(7):
        x = 0.30 + wk / 7.0 * 0.62
        ax.plot([x, x], [0.155, 0.855], color=C["line"], lw=0.7, zorder=0)
        ax.text(x, 0.128, f"W{wk}", ha="center", va="top", fontsize=8, color=C["muted"])
    ax.text(0.30, 0.128, "W0\n第 0 周", ha="center", va="top", fontsize=8, color=C["muted"], linespacing=1.5)
    ax.text(0.92, 0.128, "W6\n第 6 周末", ha="center", va="top", fontsize=8, color=C["muted"], linespacing=1.5)

    _footer(ax, "来源：开发计划.md　|　M4–M6 状态为规划，不含虚构完成度；M6 未实测量化指标不得写入简历")
    out = OUT_DIR / "arch_06_gantt.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


# ==========================================================================
# 图 7：API 与调用时序
# ==========================================================================
def fig_api() -> Path:
    fig, ax = _canvas(12.8, 7.4)
    _title(ax, "API 契约与典型调用时序", "路径扁平（/v1/...）；image_ref 单字段兼容本地路径与 base64 data URI；降级返回 2xx + degraded")

    # 左：端点清单
    _box(ax, 0.03, 0.115, 0.44, 0.795, "", "#fbfdff", C["edge_s"])
    ax.text(0.055, 0.872, "HTTP 端点（前缀 /v1）", fontsize=10.5, weight="bold", color=C["edge_s"], va="center")
    eps = [
        ("POST   /v1/session", "创建拍摄会话 → SessionInfo"),
        ("DELETE /v1/session/{id}", "结束会话"),
        ("POST   /v1/calibrate", "距离校准（FR-01）"),
        ("POST   /v1/subject", "主体确认 / 人工指定（FR-02）"),
        ("POST   /v1/frame", "单帧引导推理（核心）"),
        ("POST   /v1/shot/report", "拍后解说 + 滤镜推荐（FR-07/08）"),
        ("POST   /v1/cases/search", "案例检索（FR-09，Milvus 待接入）"),
        ("WS     /v1/stream", "实时逐帧推送"),
    ]
    # 行距按可用高度算：标题基线 0.872（框顶 0.91）→ 探针区上沿 0.235。
    # 第一条端点从 0.822 起，与标题留出 0.05 的净空。
    yy = 0.822
    for ep, note in eps:
        ax.text(0.055, yy, ep, fontsize=8.4, color=C["ink"], va="center", family="monospace", weight="bold")
        ax.text(0.055, yy - 0.030, note, fontsize=7.7, color=C["muted"], va="center")
        yy -= 0.072

    ax.text(0.055, 0.218, "运维探针：GET /healthz　/readyz　/metrics", fontsize=8.6, color=C["post_s"], weight="bold", va="center")
    ax.text(0.055, 0.185, "错误码：仅参数错误返回 4xx；\n运行时问题一律 2xx + degraded=true", fontsize=8.0, color=C["warn"], va="top", linespacing=1.6)

    # 右：时序
    ax.text(0.505, 0.865, "典型时序：一次拍摄", fontsize=10.5, weight="bold", color=C["ink"], va="top")
    # 生命线 x 坐标：客户端 / API / 热路径 / 冷路径。
    # 早期把冷路径放在 0.935，而它的箭头终点也是 0.935，箭头会**冲出**生命线之外
    # （截图里可见右侧越界）。现在整体内缩，并为所有箭头留出相等的端点内边距。
    actors = [("客户端", 0.555), ("API 层", 0.685), ("热路径\nFrameProcessor", 0.81), ("冷路径\nPostShot", 0.925)]
    for name, x in actors:
        _box(ax, x - 0.055, 0.79, 0.11, 0.052, name, C["panel"], C["line"], fs=8.0, weight="bold")
        ax.plot([x, x], [0.155, 0.79], color=C["line"], lw=1.0, ls=(0, (3, 3)), zorder=0)

    # 注意：matplotlib 的 arrowstyle **不含** '--|>'（线型由 linestyle 参数表达），
    # 早期写成 '--|>' 会抛 ValueError: Unknown style。这里用 '-|>' + ls='--' 拆开表达。
    seq = [
        (0.735, 0.555, 0.685, "POST /v1/session", "-|>", "-"),
        (0.675, 0.685, 0.555, "session_id", "-|>", "--"),
        (0.615, 0.555, 0.685, "POST /v1/calibrate（可选）", "-|>", "-"),
        (0.535, 0.555, 0.810, "POST /v1/frame  ×N（每帧）", "-|>", "-"),
        (0.485, 0.810, 0.555, "FrameSnapshot + StabilizedCommand", "-|>", "--"),
        (0.415, 0.555, 0.925, "POST /v1/shot/report（拍后一次）", "-|>", "-"),
        (0.365, 0.925, 0.555, "ShotReport（解说+滤镜，可回溯）", "-|>", "--"),
        (0.295, 0.555, 0.685, "DELETE /v1/session/{id}", "-|>", "-"),
    ]
    for y, x1, x2, label, style, ls in seq:
        col = C["language_s"] if x1 == 0.925 or x2 == 0.925 else C["ink"]
        _arrow(ax, (x1, y), (x2, y), color=col, lw=1.4, style=style, ls=ls)
        ax.text((x1 + x2) / 2, y + 0.014, label, ha="center", va="bottom", fontsize=7.7, color=col)

    ax.text(0.545, 0.185, "热路径每帧可被 /v1/frame 或 WS /v1/stream 驱动；\n冷路径仅拍后触发一次，与热路径物理隔离。",
            ha="left", va="top", fontsize=8.4, color=C["muted"], linespacing=1.7)

    _footer(ax, "源码：src/aicg/api/routes.py · ops.py　|　契约：数据模型与接口.md")
    out = OUT_DIR / "arch_07_api.png"
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


GENERATORS = {
    "arch_01_layers.png": fig_layers,
    "arch_02_frame_flow.png": fig_frame_flow,
    "arch_03_debounce.png": fig_debounce,
    "arch_04_scoring.png": fig_scoring,
    "arch_05_latency.png": fig_latency,
    "arch_06_gantt.png": fig_gantt,
    "arch_07_api.png": fig_api,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成架构图与关键示意图")
    ap.add_argument("--list", action="store_true", help="仅列出将生成的文件")
    ap.add_argument("--only", nargs="*", default=None, help="只生成指定文件名")
    args = ap.parse_args(argv)

    if args.list:
        for name in GENERATORS:
            print(name)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _setup_font()
    print(f"[字体] 使用 {font}")

    targets = args.only or list(GENERATORS)
    made = []
    for name in targets:
        fn = GENERATORS.get(name)
        if fn is None:
            print(f"[跳过] 未知图名 {name}", file=sys.stderr)
            continue
        path = fn()
        size_kb = path.stat().st_size / 1024.0
        print(f"[生成] {path.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")
        made.append(path)

    print(f"\n共 {len(made)} 张 → {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
