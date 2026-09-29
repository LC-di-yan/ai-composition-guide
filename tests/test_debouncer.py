"""指令防抖测试（FR-06）—— 护城河的回归防线。

对应文档：《测试与验收.md》§3.2 集成测试、§4 性能验收

本文件的测试目标不是"代码能跑"，而是**锁定防抖的三条硬性质**：

1. **首帧必采纳**：不能出现启动空窗；
2. **稳定信号不抖动**：同一意图持续时不得改口；
3. **噪声被抑制**：高频抖动不得转化为指令翻转；
4. **真实意图切换不被屏蔽**：该改口时要改口（不能"防抖防死"）。

第 4 条最容易被忽略——防抖做过头会让系统变得迟钝，用户体验反而更差。
"""

from __future__ import annotations

import pytest

from aicg.observability import CommandSwitchTracker
from aicg.schemas import ActionCommand, ActionType, SuppressReason, Urgency
from aicg.settings import DebounceConfig, EmaConfig, StabilizationConfig
from aicg.stabilization import CommandDebouncer


def _cmd(action: ActionType, magnitude: float = 0.2) -> ActionCommand:
    return ActionCommand(
        action=action,
        magnitude_text="test",
        magnitude_raw=magnitude,
        urgency=Urgency.MEDIUM,
        is_hold=action is ActionType.HOLD,
        confidence=0.8,
        can_skip=True,
    )


def _cfg(frames: int = 3, interval_ms: int = 1500, ema: bool = True) -> StabilizationConfig:
    return StabilizationConfig(
        ema=EmaConfig(enabled=ema, alpha=0.35),
        debounce=DebounceConfig(
            enabled=True, required_consecutive_frames=frames, min_interval_ms=interval_ms
        ),
    )


# ---------------------------------------------------------------------------
# 1. 首帧与稳定性
# ---------------------------------------------------------------------------
class TestBasicStability:
    def test_first_frame_adopted(self):
        """首帧无条件采纳（否则用户会看到"无指令"空窗）。"""
        d = CommandDebouncer(_cfg())
        out = d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=0)
        assert out.command.action is ActionType.MOVE_CLOSER
        assert out.is_changed is True

    def test_same_command_never_switches(self):
        """持续同一指令：不应产生任何切换。"""
        d = CommandDebouncer(_cfg())
        changes = 0
        for i in range(50):
            out = d.stabilize(_cmd(ActionType.MOVE_BACK), timestamp_ms=i * 333)
            changes += int(out.is_changed)
        assert changes == 1, "仅首帧应计为一次采纳"

    def test_jittery_input_suppressed(self):
        """高频抖动输入 → 切换次数应大幅低于输入变化次数。"""
        d = CommandDebouncer(_cfg(frames=3, interval_ms=1500))
        tracker = CommandSwitchTracker()
        actions = [
            ActionType.MOVE_CLOSER,
            ActionType.MOVE_BACK,
            ActionType.HOLD,
            ActionType.MOVE_LEFT,
        ]
        for i in range(200):
            raw = _cmd(actions[i % len(actions)])
            out = d.stabilize(raw, timestamp_ms=i * 333)
            tracker.add(out.command.action.value, i * 333)

        # 原始输入切换 200 次；防抖后必须显著下降
        assert tracker.switch_count <= 3, f"切换 {tracker.switch_count} 次，抑制不足"

    def test_noise_reduction_ratio(self):
        """量化验收：抖动输入的切换频率下降应 >= 80%（对齐实测 100%）。"""
        d = CommandDebouncer(_cfg())
        tracker = CommandSwitchTracker()
        n = 180
        for i in range(n):
            # 模拟手持抖动：动作每帧都在变
            raw = _cmd(ActionType.MOVE_CLOSER if i % 2 == 0 else ActionType.MOVE_BACK)
            out = d.stabilize(raw, timestamp_ms=int(i * 1000 / 15))
            tracker.add(out.command.action.value, int(i * 1000 / 15))
        assert tracker.switch_count <= 2


# ---------------------------------------------------------------------------
# 2. 连续 N 帧一致性
# ---------------------------------------------------------------------------
class TestNFrameConsensus:
    def test_switch_requires_n_frames(self):
        """新指令必须连续出现 N 帧才被采纳。"""
        d = CommandDebouncer(_cfg(frames=3))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)

        # 只出现 2 帧 → 不应切换
        for i in range(1, 3):
            out = d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=i * 1000)
            assert out.command.action is ActionType.HOLD
            assert out.suppressed_by is SuppressReason.N_FRAME_CONSENSUS

        # 第 3 帧 → 采纳
        out = d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=3000)
        assert out.command.action is ActionType.MOVE_CLOSER
        assert out.is_changed is True

    def test_interrupted_consensus_resets(self):
        """候选被其它指令打断 → 计数重置。"""
        d = CommandDebouncer(_cfg(frames=3))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)

        d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=1000)
        d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=2000)
        d.stabilize(_cmd(ActionType.MOVE_LEFT), timestamp_ms=3000)  # 打断
        out = d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=4000)
        assert out.command.action is ActionType.HOLD
        assert out.suppressed_by is SuppressReason.N_FRAME_CONSENSUS


# ---------------------------------------------------------------------------
# 3. 最小切换间隔
# ---------------------------------------------------------------------------
class TestMinInterval:
    def test_interval_blocks_rapid_switch(self):
        """通过一致性检查但间隔不足 → 仍应被抑制。"""
        d = CommandDebouncer(_cfg(frames=2, interval_ms=2000))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        # 连续 2 帧新指令，但距上次仅 100ms
        d.stabilize(_cmd(ActionType.MOVE_RIGHT), timestamp_ms=50)
        out = d.stabilize(_cmd(ActionType.MOVE_RIGHT), timestamp_ms=100)
        assert out.command.action is ActionType.HOLD
        assert out.suppressed_by is SuppressReason.MIN_INTERVAL

    def test_interval_allows_switch_after_elapsed(self):
        """间隔满足后应放行。"""
        d = CommandDebouncer(_cfg(frames=2, interval_ms=1000))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        d.stabilize(_cmd(ActionType.MOVE_RIGHT), timestamp_ms=500)
        out = d.stabilize(_cmd(ActionType.MOVE_RIGHT), timestamp_ms=2000)
        assert out.command.action is ActionType.MOVE_RIGHT

    def test_hold_bypasses_interval(self):
        """"保持"是安全指令，应豁免间隔约束。

        理由：让用户继续错误移动 1.5 秒才收到"停下"，体验与结果都更差。
        """
        d = CommandDebouncer(_cfg(frames=1, interval_ms=5000))
        d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=0)
        out = d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=100)
        assert out.command.action is ActionType.HOLD

    def test_move_back_bypasses_interval(self):
        """"后退"同样豁免：距离过近需立即纠正。"""
        d = CommandDebouncer(_cfg(frames=1, interval_ms=5000))
        d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=0)
        out = d.stabilize(_cmd(ActionType.MOVE_BACK), timestamp_ms=100)
        assert out.command.action is ActionType.MOVE_BACK


# ---------------------------------------------------------------------------
# 4. 不得"防抖防死"
# ---------------------------------------------------------------------------
class TestResponsiveness:
    def test_real_intent_change_eventually_passes(self):
        """持续的真实意图变化最终必须被采纳（防止防抖导致迟钝）。"""
        d = CommandDebouncer(_cfg(frames=3, interval_ms=1000))
        d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=0)
        adopted = False
        for i in range(1, 30):
            out = d.stabilize(_cmd(ActionType.MOVE_BACK), timestamp_ms=i * 500)
            if out.command.action is ActionType.MOVE_BACK:
                adopted = True
                break
        assert adopted, "连续 30 帧的真实意图未能生效，防抖过度"

    def test_debounce_disabled_passes_everything(self):
        """关闭防抖时指令应逐帧透传（供对照实验使用）。"""
        cfg = StabilizationConfig(
            ema=EmaConfig(enabled=False, alpha=1.0),
            debounce=DebounceConfig(enabled=False, required_consecutive_frames=3, min_interval_ms=1500),
        )
        d = CommandDebouncer(cfg)
        seq = [ActionType.MOVE_CLOSER, ActionType.MOVE_BACK, ActionType.HOLD]
        for i, act in enumerate(seq):
            out = d.stabilize(_cmd(act), timestamp_ms=i * 100)
            assert out.command.action is act


# ---------------------------------------------------------------------------
# 4b. 活跃性：摆动信号不得闩锁（D-11 回归防线）
# ---------------------------------------------------------------------------
class TestLivenessUnderOscillation:
    """[D-11 修复 2026-09-29] 锁定"安全方向必须可达"与"抗抖不得放松"。

    **真实缺陷复盘**：``_should_bypass_interval`` 声明 ``hold`` 与
    ``move_back`` 是安全方向、豁免最小切换间隔，但候选动作**根本过不了
    第 2 级**（连续 N 帧），这条豁免永远不可达——一处"写了但不生效"的
    内部自相矛盾。

    修复**刻意收窄**：只让安全方向在持续超过 ``min_interval_ms`` 后穿透
    第 2 级；激进方向（靠近/横移/俯仰）仍严格要求连续 N 帧。
    原因是 FR-06 明确要求「连续 N 帧一致才切换」，改成"窗口多数"会让
    纯交替噪声每帧都被判为共识、翻转更频繁（实测两序列的 3 帧窗口多数
    分布完全同形，无法区分）。
    """

    def test_safety_direction_released_after_interval(self):
        """安全方向（hold）在持续超过最小间隔后必须能生效（豁免可达）。"""
        d = CommandDebouncer(_cfg(frames=3, interval_ms=1000))
        d.stabilize(_cmd(ActionType.MOVE_CLOSER), timestamp_ms=0)
        # 交替喂 hold / closer：hold 永远凑不满连续 3 帧。
        # 但按安全方向豁免，应在超过 min_interval_ms 后被放行。
        adopted_hold = False
        for i in range(1, 12):
            act = ActionType.HOLD if i % 2 else ActionType.MOVE_CLOSER
            out = d.stabilize(_cmd(act), timestamp_ms=i * 400)
            if out.command.action is ActionType.HOLD:
                adopted_hold = True
                break
        assert adopted_hold, "安全方向指令在摆动信号下无法生效，豁免失效"

    def test_aggressive_direction_still_requires_consecutive_n(self):
        """激进方向（move_closer）在摆动信号下**不得**穿透（抗抖不放松）。

        这是与上一条的平衡点：放宽安全方向不能把抗抖一起放开。
        交替 closer/closer/back 时，closer 永远无法连续 3 帧 →
        必须一直被抑制在初始 hold 上。
        """
        d = CommandDebouncer(_cfg(frames=3, interval_ms=1000))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        leaked = False
        for i in range(1, 30):
            act = ActionType.MOVE_CLOSER if i % 3 else ActionType.MOVE_LEFT
            out = d.stabilize(_cmd(act), timestamp_ms=i * 1000)
            if out.command.action is ActionType.MOVE_CLOSER:
                leaked = True
                break
        assert not leaked, "激进方向在摆动信号下穿透了防抖层，抗抖被破坏"

    def test_single_frame_outlier_still_filtered(self):
        """窗口判定放宽后，**单帧孤点仍不得穿透**。"""
        d = CommandDebouncer(_cfg(frames=3, interval_ms=1500))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=500)
        # 一个孤立的 move_left（噪声）
        out = d.stabilize(_cmd(ActionType.MOVE_LEFT), timestamp_ms=1000)
        assert out.command.action is ActionType.HOLD, "单帧噪声穿透了防抖层"
        assert out.suppressed_by is SuppressReason.N_FRAME_CONSENSUS


# ---------------------------------------------------------------------------
# 5. EMA 平滑
# ---------------------------------------------------------------------------
class TestEma:
    def test_ema_first_value_is_raw(self):
        """首帧直接用原值初始化，避免从 0 缓升造成假指令。"""
        from aicg.stabilization import EmaSmoother

        e = EmaSmoother(alpha=0.35)
        assert e.update(0.8) == pytest.approx(0.8)

    def test_ema_smooths_step(self):
        """阶跃输入下 EMA 应逐步逼近而非跳变。"""
        from aicg.stabilization import EmaSmoother

        e = EmaSmoother(alpha=0.3)
        e.update(0.0)
        v = e.update(1.0)
        assert 0.0 < v < 1.0
        for _ in range(50):
            v = e.update(1.0)
        assert v > 0.95

    def test_ema_alpha_validation(self):
        from aicg.stabilization import EmaSmoother

        with pytest.raises(ValueError):
            EmaSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            EmaSmoother(alpha=1.5)

    def test_reset_clears_state(self):
        from aicg.stabilization import EmaSmoother

        e = EmaSmoother(alpha=0.3)
        e.update(0.9)
        e.reset()
        assert e.value is None
        assert e.update(0.2) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 6. 统计与契约字段
# ---------------------------------------------------------------------------
class TestStatsAndContract:
    def test_stats_keys(self):
        d = CommandDebouncer(_cfg())
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        st = d.stats()
        for k in ("frames_seen", "ema_alpha", "required_frames", "min_interval_ms"):
            assert k in st

    def test_suppressed_by_none_when_changed(self):
        d = CommandDebouncer(_cfg(frames=1))
        out = d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        assert out.is_changed is True
        assert out.suppressed_by is None

    def test_raw_command_preserved(self):
        """``raw_command`` 必须保留原始输入，供归因与可视化。"""
        d = CommandDebouncer(_cfg(frames=5))
        d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        raw = _cmd(ActionType.MOVE_LEFT, magnitude=0.42)
        out = d.stabilize(raw, timestamp_ms=100)
        assert out.raw_command.action is ActionType.MOVE_LEFT
        assert out.raw_command.magnitude_raw == pytest.approx(0.42)
        assert out.command.action is ActionType.HOLD

    def test_reset_restores_initial_state(self):
        d = CommandDebouncer(_cfg())
        d.stabilize(_cmd(ActionType.MOVE_LEFT), timestamp_ms=0)
        d.reset()
        out = d.stabilize(_cmd(ActionType.HOLD), timestamp_ms=0)
        assert out.is_changed is True  # 重置后首帧仍应采纳
