"""差分量 → 动作指令（FR-04 的核心）。

对应需求：FR-04（动作指令生成）、FR-12（兜底协商）
对应文档：《数据模型与接口.md》§2.4

调研原文的技术含义：
    "把构图模型输出的「最优画框」与「当前画框」做差分，差分方向翻译成
     自然语言动作（后退/前进/左移/右移/俯拍/仰拍）。『约 1.7 米』
     是差分量的量化。"

本模块负责这个"翻译"过程。关键设计考虑：

1. **方向判定必须稳定**：轻微的框偏移不应产生指令翻转，否则会与防抖层
   打架、加剧抖动。因此设置了 ``dead_zone``（死区），小于该阈值的差分
   视为"保持"。
2. **差分量要可量化也可读**：同时输出 ``magnitude_raw``（归一化位移，
   用于指标）与 ``magnitude_text``（人话，用于展示）。
3. **必须可拒绝**：``can_skip`` 恒为 True，对应"无法后退，就这样构图"。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ActionCommand, ActionType, Urgency
from ..schemas.perception import BBox
from ..utils.image import bbox_center, bbox_height, bbox_width

# 差分方向判定阈值（归一化坐标）
_DEFAULT_DEAD_ZONE = 0.035
"""小于此位移量视为"保持"。取值依据：归一化 0.035 ≈ 1080p 下的 38px，
低于人眼可感知的构图差异，避免微动触发改口。"""

_URGENCY_HIGH = 0.18
_URGENCY_MEDIUM = 0.09

_TARGET_OCCUPANCY_IN_FRAME_FALLBACK = 0.55
"""理想主体占画面高度的比例——**仅作降级兜底**。

[D-10 修复 2026-09-29] 本模块此前把 0.68 硬编码为模块常量，与评分层的
``ScoringConfig.ideal_subject_height``（0.55）和距离层的
``ideal_distance_m``（2.6m，反推约 0.71）**三处口径互相矛盾**。

现在的真源是 :attr:`aicg.settings.ScoringConfig.ideal_subject_height`，
由 :class:`CommandDiffer` 构造时注入。本常量只在与配置解耦的场景
（如单元测试里直接 ``CommandDiffer()``）下兜底，**取值必须与
``configs/default.yaml`` 的 ``ideal_subject_height`` 保持一致**。

详见 ``docs/research/构图指令一致性与防抖闩锁缺陷_D-10_D-11.md``。
"""

_OCCUPANCY_TOLERANCE_FALLBACK = 0.0825
"""占比容差兜底值，= ``0.55 * 0.15``（见 ``occupancy_tolerance_ratio``）。

同样**仅作降级兜底**；正常路径由配置注入。
"""

_THIRDS_POSITIONS = (1.0 / 3.0, 2.0 / 3.0, 0.5)
"""主体"合理水平位置"的集合：左右三分点 + 画面中心。

理由：这三处是构图学上唯三被普遍接受的横向落点。主体落在其中任一处
附近，就**不应再要求横向移动**——否则会出现"用户已在三分点，AI 还让他
继续挪"的荒谬建议。
"""

_POSITION_TOLERANCE = 0.07
"""距离最近合理落点的容差。超过此值才认为需要横向移动。

取值依据：三分点与中心的最小间距是 ``|0.5 - 0.333| = 0.167``，
容差 0.07 意味着"三处合理位置之间留出约 0.027 的判别间隙"，
既不会把三分点附近判成偏移，也不会把明显偏到边缘的主体判成合规。
"""


@dataclass
class DeltaResult:
    """差分计算结果。"""

    dx: float
    """水平位移（归一化，正值表示目标框在原框右侧）。"""

    dy: float
    """垂直位移（归一化，正值表示目标框在原框下方）。"""

    d_scale: float
    """尺度比：当前主体占比 / 目标占比。>1 表示主体偏大（应后退）。"""

    magnitude: float
    """合成位移量（归一化）。"""

    direction: str
    """粗方向：``backward`` / ``forward`` / ``left`` / ``right`` / ``up`` / ``down`` / ``hold``。"""

    reason: str = ""
    """方向判定的依据，用于归因与调试（NFR-O2）。"""


def _dominant_magnitude(delta: DeltaResult) -> float:
    """取"主导本次决策"的偏离量，作为 ``magnitude_raw``。

    **为什么不能直接用 ``delta.magnitude``**：
    ``magnitude`` 只是横向+纵向的合成位移。当决策由**尺度**驱动时
    （例如主体太小需要靠近，但恰好水平居中），合成位移可能为 0，
    于是 ``magnitude_raw`` 也变成 0——然而这不是 hold，指标上会失真
    （防抖层的 EMA 也拿不到有效信号）。

    因此这里取二者中的较大值，且对尺度偏离做量纲换算：
    占比偏离 ``|1 - d_scale|`` 是比例量，乘以一个经验系数后与位移量
    可比。系数取 3.0 的由来：占比偏离 0.08（容差边界）对应
    ``0.08 * 3.0 = 0.24``，与"显著位移"（``_URGENCY_HIGH = 0.18``）
    同量级，使两者的 urgency 分级保持一致。
    """
    scale_term = abs(1.0 - delta.d_scale) * 3.0
    return max(delta.magnitude, scale_term)


class CommandDiffer:
    """把当前框与最优框的差分翻译为动作指令。"""

    def __init__(
        self,
        dead_zone: float = _DEFAULT_DEAD_ZONE,
        assumed_person_height_m: float = 1.65,
        focal_constant: float = 26.0,
        sensor_height_mm: float = 24.0,
        target_occupancy: float = _TARGET_OCCUPANCY_IN_FRAME_FALLBACK,
        occupancy_tolerance: float = _OCCUPANCY_TOLERANCE_FALLBACK,
    ) -> None:
        """
        Args:
            dead_zone: 死区阈值，小于该位移视为保持。
            assumed_person_height_m / focal_constant / sensor_height_mm:
                与距离估算保持同源参数，用于把尺度差换算成米数。
            target_occupancy: **理想主体占画面高度的比例**。必须与评分层的
                ``ScoringConfig.ideal_subject_height`` 取同一个值——这是
                D-10 修复的核心：三层共用一份"理想构图"定义，而不是各存
                一份互相矛盾的字面量。
            occupancy_tolerance: 占比容差（绝对值）。应与 ``target_occupancy``
                同源派生，见 ``ScoringConfig.derived_occupancy_tolerance``。

        两个新参数都有兜底默认值，因此 ``CommandDiffer()`` 这种裸构造仍然
        可用（单元测试需要）；但**生产路径必须从配置注入**
        （见 :class:`aicg.pipeline.frame_processor.FrameProcessor`），
        否则会悄悄退回旧的不自洽状态。
        """
        self.dead_zone = dead_zone
        self.assumed_height_m = assumed_person_height_m
        self.focal_constant = focal_constant
        self.sensor_height_mm = sensor_height_mm
        self.target_occupancy = target_occupancy
        self.occupancy_tolerance = occupancy_tolerance

    # ------------------------------------------------------------------
    @property
    def hold_window(self) -> tuple[float, float]:
        """"距离已合适"的占高窗口 ``(下界, 上界)``，供文档与前端一致性校验。

        前端 ``web/index.html`` 里的 ``D10_HOLD_LO/HI`` 必须与本属性一致；
        ``scripts/verify_web_demo.py`` 在浏览器侧断言这一点。
        """
        lo = self.target_occupancy * (1.0 - self.occupancy_tolerance / max(self.target_occupancy, 1e-9))
        return (lo, self.target_occupancy + self.occupancy_tolerance)

    # ------------------------------------------------------------------
    def compute_delta(self, current_bbox: BBox, best_bbox: BBox) -> DeltaResult:
        """计算当前框与最优框的差分。

        Args:
            current_bbox: 当前主体框（归一化）。
            best_bbox: 构图模型输出的最优画框（归一化）。

        Returns:
            差分结果。尺度主导时返回前进/后退，否则按位移方向返回平移/俯仰。

        决策优先级（scale → translate，且各有独立容差）:

            1. **尺度优先**：只要占比偏离超过 ``self.occupancy_tolerance``，
               就输出前进/后退。理由：改变站位是高成本动作，必须先解决
               "该站多远"，否则用户会为微调左右而来回走动。
            2. **位移次之**：占比已合适时，才看水平/垂直位移，且必须
               超过 :attr:`dead_zone`。
            3. **否则保持**：两者都无显著差异时输出 ``hold``。

        尺度语义（易错点，务必对齐）：
            比较的是**主体占整幅画面的比例**与目标占比
            （``self.target_occupancy``，默认 0.55，与评分层同源），
            而不是两个框的高度。因为 ``best_bbox`` 是"取景范围"、
            ``current_bbox`` 是"主体本身"，两者尺度不可直接相比——
            例如"主体高 0.77、最佳画框高 1.0"并不意味着该靠近。

        位移语义（易错点，务必对齐——这是第二个真实 bug 的复盘）：
            **不能直接比较 ``current_bbox`` 与 ``best_bbox`` 的中心**。
            因为 ``best_bbox`` 是三分法候选框，它故意把主体安排在偏右/偏左
            三分之一处，所以它的中心天然偏离主体中心。若直接相减，会把
            "主体已正确落在右三分点"误判成"需要向左移动"。

            正确做法：比较**主体相对基准的位置**。
            :attr:`_TRANSLATE_REFERENCE` 提供两种基准——

            - ``subject``：主体中心相对**画面中心**的偏移（默认）。
              语义是"主体该往画面中心靠还是要离开中心"。
            - ``bbox``：主体中心相对 ``best_bbox`` 中心的偏移。
              语义是"主体该在画框内往哪挪"，仅在 ``best_bbox`` 是真实
              取景范围（非构图锚点）时才适用。

            默认选 ``subject``：因为用户能移动的是**自己与画面的相对关系**，
            而不是"主体在某个候选框内的位置"。后者用户无法直接感知。

        实测证据（同一视频第 6 帧）::

            主体占高 0.684、水平中心 0.631（正好在右三分点）
            最优画框中心 0.565  ← 三分法构图使其天然偏左
            直接相减 → dx = -0.066 超过死区 → 误报 "move_left"
            改用画面中心基准 → dx = 0.631 - 0.5 = 0.131，但主体已在
            三分点(0.667)附近，容差内 → 正确输出 "hold"

        .. note::
           上例中"占高 0.684"在 D-10 修复后**已高于**理想占比窗口上界
           （0.55 × 1.15 = 0.6325），因此现在会先走尺度分支输出
           ``backward``（该后退）而非 ``hold``。这是**修复后的预期行为**：
           旧代码把 0.68 当理想值，才会认为 0.684 无需调整。
        """
        cx_cur, cy_cur = bbox_center(current_bbox)
        cx_best, cy_best = bbox_center(best_bbox)

        h_cur = max(1e-6, bbox_height(current_bbox))
        current_occupancy = min(1.0, h_cur)  # 主体占整幅画面的比例
        d_scale = current_occupancy / self.target_occupancy
        scale_deviation = abs(1.0 - d_scale)

        # --- 水平位移：以"最近的合理落点"为基准，而非 best_bbox 中心 ---
        nearest_anchor = min(_THIRDS_POSITIONS, key=lambda a: abs(cx_cur - a))
        dx = nearest_anchor - cx_cur
        dy = cy_best - cy_cur  # 垂直方向没有三分锚点概念，保留中心差

        magnitude = (dx**2 + dy**2) ** 0.5

        direction = "hold"
        reason = "尺度和位置均在容差内"

        # --- 优先级 1：尺度（使用量纲匹配的容差）---
        if scale_deviation > self.occupancy_tolerance:
            direction = "forward" if d_scale < 1.0 else "backward"
            reason = (
                f"主体占画面 {current_occupancy:.2f}，偏离目标 "
                f"{self.target_occupancy:.2f} 达 {scale_deviation:.2f}"
                f"（容差 {self.occupancy_tolerance:.3f}）"
            )
        # --- 优先级 2：横向（仅当主体不贴近任何合理落点时）---
        elif abs(dx) > _POSITION_TOLERANCE:
            direction = "right" if dx > 0 else "left"
            reason = (
                f"主体水平位置 {cx_cur:.2f}，距最近的合理落点 "
                f"{nearest_anchor:.2f} 达 {abs(dx):.2f}（容差 {_POSITION_TOLERANCE}）"
            )
        # --- 优先级 3：纵向 ---
        elif abs(dy) > self.dead_zone:
            direction = "down" if dy > 0 else "up"
            reason = f"垂直偏移 {dy:.3f} 超过死区 {self.dead_zone}"

        return DeltaResult(
            dx=dx,
            dy=dy,
            d_scale=d_scale,
            magnitude=magnitude,
            direction=direction,
            reason=reason,
        )

    # ------------------------------------------------------------------
    def to_command(
        self,
        current_bbox: BBox,
        best_bbox: BBox,
        confidence: float = 1.0,
    ) -> ActionCommand:
        """生成动作指令。

        Args:
            current_bbox: 当前主体框。
            best_bbox: 最优画框。
            confidence: 构图决策的置信度，透传到指令。

        Returns:
            动作指令。**保证返回**，无有效差分时返回 ``hold``。
        """
        delta = self.compute_delta(current_bbox, best_bbox)

        action_map = {
            "forward": ActionType.MOVE_CLOSER,
            "backward": ActionType.MOVE_BACK,
            "left": ActionType.MOVE_LEFT,
            "right": ActionType.MOVE_RIGHT,
            "up": ActionType.TILT_UP,
            "down": ActionType.TILT_DOWN,
            "hold": ActionType.HOLD,
        }
        action = action_map[delta.direction]
        is_hold = action is ActionType.HOLD

        return ActionCommand(
            action=action,
            magnitude_text="保持" if is_hold else self._magnitude_text(delta, current_bbox),
            magnitude_raw=0.0 if is_hold else round(_dominant_magnitude(delta), 4),
            urgency=self._urgency(delta, is_hold),
            is_hold=is_hold,
            confidence=float(max(0.0, min(1.0, confidence))),
            can_skip=True,  # FR-12：任何建议都必须可被拒绝
        )

    # ------------------------------------------------------------------
    def _magnitude_text(self, delta: DeltaResult, current_bbox: BBox) -> str:
        """把差分量转成人话。

        缩放类动作换算为米（符合调研观察到的"约 1.7 米"表述）；
        平移类换算为"步"；俯仰类给出角度描述。

        注意：``hold`` 的文案由 :meth:`to_command` 直接给出"保持"，
        不进入本函数——避免出现"保持"却附带"调整一下"的矛盾文案。
        """
        if delta.direction in ("forward", "backward"):
            # 用同源的距离公式换算：当前主体占比 -> 当前距离；
            # 目标占比 -> 目标距离；两者之差即需要移动的米数。
            cur_ratio = max(1e-6, bbox_height(current_bbox))
            target_ratio = self.target_occupancy
            cur_m = self._box_height_to_distance(cur_ratio)
            target_m = self._box_height_to_distance(target_ratio)
            diff = abs(target_m - cur_m)
            return f"约 {diff:.1f} 米" if diff >= 0.1 else "稍微调整距离"

        if delta.direction in ("left", "right"):
            # 归一化位移映射到"步"：一步约覆盖画面 15%
            steps = delta.magnitude / 0.15
            if steps >= 1.5:
                return f"约 {int(round(steps))} 步"
            return "稍微挪一点"

        if delta.direction in ("up", "down"):
            return "稍微调整角度"

        return "调整一下"

    def _box_height_to_distance(self, box_height_ratio: float) -> float:
        """框高占比 → 距离（m）。与 :class:`DistanceEstimator` 同源公式。

        用「等效视角」推导，避免等效焦距与物理传感器尺寸混用导致的量纲错误。
        """
        import math

        ratio = max(1e-4, min(box_height_ratio, 0.999))
        fov_v_deg = 2.0 * math.degrees(
            math.atan(self.sensor_height_mm / (2.0 * self.focal_constant))
        )
        theta = min(max(math.radians(ratio * fov_v_deg), 1e-4), math.radians(170.0))
        return (self.assumed_height_m / 2.0) / math.tan(theta / 2.0)

    def _urgency(self, delta: DeltaResult, is_hold: bool) -> Urgency:
        """紧急度分级。与 ``magnitude_raw`` 使用同一套量纲（见 :func:`_dominant_magnitude`）。"""
        if is_hold:
            return Urgency.LOW
        m = _dominant_magnitude(delta)
        if m >= _URGENCY_HIGH:
            return Urgency.HIGH
        if m >= _URGENCY_MEDIUM:
            return Urgency.MEDIUM
        return Urgency.LOW
