"""差分量与动作指令的单元测试（FR-04）。

对应文档：《测试与验收.md》§3.1 单元测试重点

**这些测试锁定的都是真实修复过的 bug**，是回归防线而非形式化覆盖：

1. 尺度判定必须用"占比 vs 目标占比"，不能用"两个框的高度差"；
2. 尺度容差与位移容差必须量纲匹配，不能用 ``dead_zone * 2`` 近似；
3. 横向判定必须以"合理落点"为基准，不能直接比较 ``best_bbox`` 中心。
"""

from __future__ import annotations

import pytest

from aicg.composition.differ import (
    _OCCUPANCY_TOLERANCE_FALLBACK,
    _TARGET_OCCUPANCY_IN_FRAME_FALLBACK,
    _THIRDS_POSITIONS,
    CommandDiffer,
)
from aicg.schemas import ActionType

# 兼容别名：本文件内下文统一用这两个名字引用"目标占比"与"容差"。
# [D-10 修复 2026-09-29] 这两个值已不再是模块级唯一的定义——生产路径由
# ``ScoringConfig.ideal_subject_height`` 注入。这里保留别名是为了让断言
# 仍然只表达"相对关系"（如 ±容差的 k 倍），而不是把 0.55 这个具体数字
# 抄进测试里。
_TARGET_OCCUPANCY_IN_FRAME = _TARGET_OCCUPANCY_IN_FRAME_FALLBACK
_OCCUPANCY_TOLERANCE = _OCCUPANCY_TOLERANCE_FALLBACK


@pytest.fixture
def differ() -> CommandDiffer:
    return CommandDiffer()


def _box(cx: float, cy: float, h: float, aspect: float = 0.42):
    """按中心、高度、宽高比构造 bbox。"""
    w = h * aspect
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------------------
# 尺度判定
# ---------------------------------------------------------------------------
class TestScaleDecision:
    def test_subject_too_small_commands_closer(self, differ):
        """主体占比明显小于目标 → 前进。"""
        cur = _box(0.5, 0.5, 0.30)
        best = (0.0, 0.0, 1.0, 1.0)
        cmd = differ.to_command(cur, best)
        assert cmd.action is ActionType.MOVE_CLOSER

    def test_subject_too_large_commands_back(self, differ):
        """主体占比明显大于目标 → 后退。"""
        cur = _box(0.5, 0.5, 0.95)
        best = (0.0, 0.0, 1.0, 1.0)
        cmd = differ.to_command(cur, best)
        assert cmd.action is ActionType.MOVE_BACK

    def test_ideal_occupancy_is_hold(self, differ):
        """占比恰好等于目标时不应产生纵向指令（尺度和位移均在容差内）。"""
        cur = _box(0.5, 0.5, _TARGET_OCCUPANCY_IN_FRAME)
        best = (0.0, 0.0, 1.0, 1.0)
        cmd = differ.to_command(cur, best)
        assert cmd.action is ActionType.HOLD
        assert cmd.is_hold is True
        assert cmd.magnitude_text == "保持"

    def test_within_tolerance_is_hold(self, differ):
        """占比偏离在容差内 → 保持（避免无意义的微调指令）。"""
        h = _TARGET_OCCUPANCY_IN_FRAME * (1.0 + _OCCUPANCY_TOLERANCE * 0.5)
        cur = _box(0.5, 0.5, h)
        cmd = differ.to_command(cur, (0.0, 0.0, 1.0, 1.0))
        assert cmd.action is ActionType.HOLD

    def test_boundary_just_over_tolerance_triggers(self, differ):
        """刚超出容差 → 应产生缩放指令（边界行为锁定）。"""
        h = _TARGET_OCCUPANCY_IN_FRAME * (1.0 + _OCCUPANCY_TOLERANCE * 1.2)
        cur = _box(0.5, 0.5, h)
        cmd = differ.to_command(cur, (0.0, 0.0, 1.0, 1.0))
        assert cmd.action is ActionType.MOVE_BACK


# ---------------------------------------------------------------------------
# 位移判定（回归：三分法候选框导致误报 move_left）
# ---------------------------------------------------------------------------
class TestTranslationDecision:
    def test_subject_at_thirds_anchor_is_hold(self, differ):
        """主体已在三分点、占比合适 → 保持。

        回归点：早期实现直接比较 ``best_bbox`` 中心，而三分法候选框的
        中心天然偏离主体，导致误报 ``move_left``。
        """
        for anchor in _THIRDS_POSITIONS:
            cur = _box(anchor, 0.5, _TARGET_OCCUPANCY_IN_FRAME)
            # 故意构造一个中心明显偏左的三分法式候选框
            best = (0.1, 0.05, 0.75, 0.95)
            cmd = differ.to_command(cur, best)
            assert cmd.action is ActionType.HOLD, f"anchor={anchor} 误报 {cmd.action}"

    def test_subject_far_left_commands_right(self, differ):
        """主体明显偏左 → 右移。"""
        cur = _box(0.14, 0.5, _TARGET_OCCUPANCY_IN_FRAME)
        cmd = differ.to_command(cur, (0.0, 0.0, 1.0, 1.0))
        assert cmd.action is ActionType.MOVE_RIGHT

    def test_subject_far_right_commands_left(self, differ):
        """主体明显偏右 → 左移。"""
        cur = _box(0.88, 0.5, _TARGET_OCCUPANCY_IN_FRAME)
        cmd = differ.to_command(cur, (0.0, 0.0, 1.0, 1.0))
        assert cmd.action is ActionType.MOVE_LEFT

    def test_scale_takes_priority_over_translation(self, differ):
        """尺度与位移同时显著时，优先给缩放类指令。

        理由：改变站位成本高，先解决"站多远"再微调左右，避免用户来回走。
        """
        cur = _box(0.05, 0.5, 0.30)  # 又偏左，又太小
        cmd = differ.to_command(cur, (0.0, 0.0, 1.0, 1.0))
        assert cmd.action is ActionType.MOVE_CLOSER


# ---------------------------------------------------------------------------
# 指令契约不变量
# ---------------------------------------------------------------------------
class TestCommandInvariants:
    def test_can_skip_always_true(self, differ):
        """FR-12：任何建议都必须可被拒绝。"""
        for h in (0.2, 0.68, 0.95):
            cmd = differ.to_command(_box(0.5, 0.5, h), (0.0, 0.0, 1.0, 1.0))
            assert cmd.can_skip is True

    def test_hold_has_zero_magnitude(self, differ):
        """hold 指令的 magnitude_raw 必须为 0，且文案为"保持"。"""
        cmd = differ.to_command(_box(0.5, 0.5, _TARGET_OCCUPANCY_IN_FRAME), (0.0, 0.0, 1.0, 1.0))
        assert cmd.magnitude_raw == 0.0
        assert cmd.magnitude_text == "保持"

    def test_non_hold_has_positive_magnitude(self, differ):
        cmd = differ.to_command(_box(0.5, 0.5, 0.25), (0.0, 0.0, 1.0, 1.0))
        assert cmd.magnitude_raw is not None and cmd.magnitude_raw > 0.0

    def test_distance_text_contains_meters(self, differ):
        """缩放类指令的文案必须包含米数（调研要求的量化表达）。"""
        cmd = differ.to_command(_box(0.5, 0.5, 0.25), (0.0, 0.0, 1.0, 1.0))
        assert "米" in cmd.magnitude_text

    def test_confidence_clamped(self, differ):
        """置信度越界时必须被夹紧到 [0, 1]。"""
        cmd = differ.to_command(_box(0.5, 0.5, 0.3), (0.0, 0.0, 1.0, 1.0), confidence=5.0)
        assert cmd.confidence == 1.0
        cmd = differ.to_command(_box(0.5, 0.5, 0.3), (0.0, 0.0, 1.0, 1.0), confidence=-3.0)
        assert cmd.confidence == 0.0


# ---------------------------------------------------------------------------
# 距离换算自洽性
# ---------------------------------------------------------------------------
class TestDistanceConsistency:
    def test_larger_box_means_closer(self, differ):
        """框越大 → 距离越近（单调性）。"""
        d_small = differ._box_height_to_distance(0.3)
        d_large = differ._box_height_to_distance(0.9)
        assert d_large < d_small

    def test_known_value_matches_research_doc(self, differ):
        """校验调研给出的量级：占画面 70% 高度 ≈ 2m 左右。

        这是量纲正确性的关键回归——早期版本因混用等效焦距与物理传感器
        尺寸，得到 0.00m（错 1000 倍）。
        """
        d = differ._box_height_to_distance(0.70)
        assert 1.5 < d < 3.0, f"占高 0.70 时应约 2m，实际 {d:.2f}m"
