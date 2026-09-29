"""可视化渲染：把 ``FrameSnapshot`` 画回图像上。

对应需求：FR-05（实时引导循环的可观测性）、NFR-O1
对应文档：《技术方案.md》§1 —— 录屏轨道交付物

**这是"录屏双轨"的核心渲染器**：没有它，引导系统只是一个输出 JSON 的
黑盒，无法在录屏里被"看见"。渲染内容刻意做了取舍——

绘制（对用户有意义）：
  - 主体框（绿）+ 最优画框（黄虚线）
  - 三分线（淡）
  - 指令文本 + 米数（大字，居上）
  - 构图评分、模式、景别（角标）
  - 延迟与降级状态（角标，证明实时性）

**刻意不绘制**：所有候选框（上百个，会糊成一团）。候选框证据放在
静态对比图里输出，不进入录屏主画面——录屏要的是"用户视角"，不是
"算法视角"。

字体：使用 OpenCV 内置 Hershey 字体（无外部依赖）；中文无法用 Hershey
渲染，因此画面文字**统一用英文/数字**，中文文案由叠加的前端层负责。
这是诚实的技术约束说明，不做无用的伪中文字形拼凑。
"""

from __future__ import annotations

import numpy as np
import cv2

from ..schemas import FrameSnapshot

# ---- 配色（BGR）----
C_SUBJECT = (90, 220, 90)
"""主体框：绿。"""

C_BEST = (60, 200, 250)
"""最优画框：黄。"""

C_THIRDS = (140, 140, 140)
"""三分线：灰。"""

C_TEXT = (250, 250, 250)
C_WARN = (60, 80, 240)
C_OK = (90, 220, 90)
C_SCORE_BG = (40, 40, 40)

_ACTION_EN = {
    "move_closer": "STEP CLOSER",
    "move_back": "STEP BACK",
    "move_left": "MOVE LEFT",
    "move_right": "MOVE RIGHT",
    "tilt_up": "TILT UP",
    "tilt_down": "TILT DOWN",
    "hold": "HOLD",
}

_SHOT_EN = {
    "特写": "CLOSE-UP",
    "半身": "MEDIUM",
    "七分身": "COWBOY",
    "全身": "FULL",
    "远景": "WIDE",
}

_PATTERN_EN = {
    "rule_of_thirds": "thirds",
    "center": "center",
    "symmetric": "symmetric",
    "diagonal": "diagonal",
    "framing": "framing",
    "unknown": "unknown",
}


def _ascii_only(text: str) -> str:
    """把非 ASCII 字符替换为 '?'。

    **为什么需要**：OpenCV 内置 Hershey 字体不含 CJK 字形，直接绘制中文
    会得到一串 ``?``。本函数把这个"已知约束"显式化——上层若传中文文案
    （如 ``magnitude_text="约 2.1 米"``），渲染前会经过这里降级为 ASCII，
    避免画面出现参差不齐的乱码。

    **生产环境做法**：接入 Pillow + 中文字体（如思源黑体）绘制文字层，
    或由前端在视频流上叠加 HTML 文字。本项目的 Web 轨道采用的是后者。
    """
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)


def _magnitude_ascii(cmd) -> str:
    """把指令差分量转为 ASCII 描述（供 Hershey 字体渲染）。

    语义映射（刻意粗粒度，因为录屏画面上的文字过细反而不可读）：
      - "保持"                  -> ``HOLD``
      - 位移量 >= 0.18          -> ``FAR``
      - 位移量 >= 0.09          -> ``A BIT``
      - 其余                    -> ``SLIGHTLY``
    """
    if cmd.is_hold:
        return ""
    if cmd.action.value in ("move_closer", "move_back"):
        # 缩放类指令：用 d_scale 无法直接拿到，改用 magnitude_raw 分级
        m = float(cmd.magnitude_raw or 0.0)
        # 缩放指令的 magnitude_raw 是位移量，但"米数"信息在 magnitude_text 里。
        # 这里提取其中的数字部分（"约 2.1 米" -> "~2.1m"）。
        import re

        nums = re.findall(r"\d+\.?\d*", cmd.magnitude_text)
        if nums:
            return f"~{nums[0]}m"
        return "ADJUST"
    m = float(cmd.magnitude_raw or 0.0)
    if m >= 0.18:
        return "FAR"
    if m >= 0.09:
        return "A BIT"
    return "SLIGHTLY"


class SnapshotRenderer:
    """把快照渲染到帧图像上。"""

    def __init__(
        self,
        *,
        draw_thirds: bool = True,
        draw_candidates: bool = False,
        scale: float = 1.0,
    ) -> None:
        """
        Args:
            draw_thirds: 是否画三分线参考。
            draw_candidates: 是否画出 Top-K 候选框（默认关闭，见模块文档）。
            scale: 字号缩放，便于输出不同分辨率时保持可读。
        """
        self.draw_thirds = draw_thirds
        self.draw_candidates = draw_candidates
        self.scale = scale

    # ------------------------------------------------------------------
    def render(self, image: np.ndarray, snapshot: FrameSnapshot) -> np.ndarray:
        """渲染并返回新图（不修改入参，避免污染后续处理）。"""
        canvas = image.copy()
        h, w = canvas.shape[:2]
        s = self.scale

        if self.draw_thirds:
            self._draw_thirds(canvas, w, h)

        if self.draw_candidates:
            self._draw_candidates(canvas, snapshot, w, h)

        # 最优画框（黄虚线）
        best = snapshot.composition.best_bbox
        if best is not None:
            self._draw_dashed_rect(canvas, best, w, h, C_BEST, thickness=max(1, int(2 * s)))

        # 主体框（绿实线）
        subject = snapshot.perception.primary_subject
        if subject is not None:
            self._draw_rect(canvas, subject.bbox, w, h, C_SUBJECT, thickness=max(2, int(2 * s)))
            self._put_text(
                canvas,
                subject.label,
                (int(subject.bbox[0] * w), max(12, int(subject.bbox[1] * h) - 6)),
                C_SUBJECT,
                0.45 * s,
            )

        self._draw_command_banner(canvas, snapshot, w, h)
        self._draw_corner_stats(canvas, snapshot, w, h)
        return canvas

    # ------------------------------------------------------------------
    def _draw_thirds(self, canvas, w: int, h: int) -> None:
        for i in (1, 2):
            x = int(w * i / 3)
            y = int(h * i / 3)
            cv2.line(canvas, (x, 0), (x, h), C_THIRDS, 1, cv2.LINE_AA)
            cv2.line(canvas, (0, y), (w, y), C_THIRDS, 1, cv2.LINE_AA)

    def _draw_candidates(self, canvas, snapshot: FrameSnapshot, w: int, h: int) -> None:
        for cand in snapshot.composition.top_candidates:
            self._draw_rect(canvas, cand.bbox, w, h, (180, 180, 120), thickness=1)

    def _draw_rect(self, canvas, bbox, w: int, h: int, color, thickness: int = 2) -> None:
        x1, y1, x2, y2 = bbox
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    def _draw_dashed_rect(self, canvas, bbox, w: int, h: int, color, thickness: int = 2) -> None:
        """画虚线矩形，用于区分"建议画框"与"实际主体框"。"""
        x1, y1, x2, y2 = (int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h))
        dash = 10
        for x in range(x1, x2, dash * 2):
            cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
        for y in range(y1, y2, dash * 2):
            cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
            cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)

    def _draw_command_banner(self, canvas, snapshot: FrameSnapshot, w: int, h: int) -> None:
        """顶部指令横幅：录屏里最需要被"一眼看到"的信息。"""
        cmd = snapshot.command.command
        s = self.scale
        action = _ACTION_EN.get(cmd.action.value, cmd.action.value.upper())
        # magnitude_text 是中文（"约 2.1 米"/"保持"），Hershey 字体无法渲染。
        # 这里用 ASCII 量化值替代：从 magnitude_raw 反推相对幅度并分级。
        text = f"{action}  {_magnitude_ascii(cmd)}"

        is_hold = cmd.is_hold
        color = C_OK if is_hold else C_WARN

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, 2)
        pad = int(10 * s)
        box_h = th + pad * 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, box_h), C_SCORE_BG, -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(
            canvas, text, (pad, th + pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, color, 2, cv2.LINE_AA,
        )

        # 切换/抑制状态标记：证明防抖在起作用
        cmd_st = snapshot.command
        if cmd_st.suppressed_by is not None:
            tag = f"HOLDING ({cmd_st.suppressed_by.value})"
            cv2.putText(canvas, tag, (pad, box_h + int(18 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (200, 200, 200), 1, cv2.LINE_AA)
        elif cmd_st.is_changed:
            cv2.putText(canvas, "SWITCH", (pad, box_h + int(18 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, C_WARN, 1, cv2.LINE_AA)

    def _draw_corner_stats(self, canvas, snapshot: FrameSnapshot, w: int, h: int) -> None:
        s = self.scale
        comp = snapshot.composition
        shot = _SHOT_EN.get(comp.shot_size_label, _ascii_only(comp.shot_size_label or "-"))
        pattern = _PATTERN_EN.get(comp.pattern.value, comp.pattern.value)
        lines = [
            f"score {comp.composition_score:.0f}/100  best {comp.best_score or 0:.0f}",
            f"pattern {pattern}  shot {shot}",
            f"latency {snapshot.latency.total_ms:.1f}ms",
        ]
        if snapshot.degraded:
            lines.append(f"DEGRADED: {snapshot.degradation_reason.value if snapshot.degradation_reason else '-'}")

        y = h - int(12 * s) - int(16 * s) * (len(lines) - 1)
        for i, line in enumerate(lines):
            color = C_WARN if (snapshot.degraded and i == len(lines) - 1) else C_TEXT
            cv2.putText(
                canvas, line, (int(8 * s), y + i * int(16 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, line, (int(8 * s), y + i * int(16 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, color, 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    def _put_text(self, canvas, text: str, org, color, font_scale: float) -> None:
        """带黑描边的文字，保证在任何背景上可读。"""
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def render_snapshot(image: np.ndarray, snapshot: FrameSnapshot, **kwargs) -> np.ndarray:
    """便捷函数：单帧渲染。"""
    return SnapshotRenderer(**kwargs).render(image, snapshot)
