"""指令防抖（FR-06）—— 本项目护城河的核心实现。

对应需求：FR-06（指令防抖）、NFR-P2（引导稳定性）
对应文档：《技术方案.md》§3 关键工程约束

调研的核心判断：
    "这个功能的『神奇感』60% 来自**实时性工程**（低延迟、稳定、不抖）……
     大多数人做得出来 demo，做不出『变动中仍准确且不抖』的体验。"

因此本模块是项目工程价值最集中的地方。三级防抖串联：

1. **EMA 时域平滑**（:class:`EmaSmoother`）
   对差分量做指数移动平均，抑制逐帧高斯噪声。响应速度由 ``alpha`` 控制。

2. **连续 N 帧一致性**（:class:`CommandDebouncer`）
   新指令必须连续出现 N 帧才被采纳，滤掉偶发抖动。

3. **最小切换间隔**（:class:`CommandDebouncer`）
   两次切换之间设置时间下限，避免"改口又改回来"。

设计要点：三者**顺序敏感**——必须先平滑再防抖。若先防抖后平滑，
平滑会让已经确认的指令再次漂移，破坏"确认即稳定"的语义。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ActionCommand, ActionType, StabilizedCommand, SuppressReason
from ..settings import StabilizationConfig

_SAFETY_RELEASE_COOLDOWN_MS = 4000
"""安全方向豁免的冷却时间（ms）。

两次"安全方向闩锁释放"之间至少间隔这么久。取值依据：
``min_interval_ms`` 默认 1500ms，而真实拍摄中一次真正的意图变化通常伴随
若干帧的过渡；冷却取 ~2.7× 最小间隔，既能解开闩锁，又不足以在噪声序列上
形成新的翻转源。实测（``handheld_jitter.mp4``，AC-06 基准素材）：
引入豁免后切换 19 → 10 次/240 帧，未劣化。
"""

_LONG_WINDOW_SIZE = 12
"""长窗长度（帧），用于判别候选动作是"少数派噪声"还是"持续新意图"。

**为什么必须有这一条**：单靠"卡住时长 + 冷却"仍不足以区分两者——
``closer, back, hold, left`` 这类全方向循环在任意 4 秒窗口内也满足
"卡住且是安全方向"，会被反复触发（实测 200 帧内 14 次切换，远超
``test_jittery_input_suppressed`` 允许的 3 次）。

判别依据：**候选动作在较长horizon内是否占据多数**。
- 全方向循环：任一动作在 12 帧里只占 ~1/4，**不是**多数 → 不释放；
- 真实意图变化：新动作在 12 帧里往往占多数（如真实素材里的 move_back
  在 10 帧序列中占 6 帧）→ 释放。

窗口取 12 的理由：约等于常见过渡时长（0.5~1s @ 12~24fps）的量级，
足够容纳零星异类，又不至于把整段历史都算进来导致迟钝。
"""


class EmaSmoother:
    """指数移动平均平滑器。

    EMA 递推式::

        s_t = alpha * x_t + (1 - alpha) * s_{t-1}

    ``alpha`` 越大越跟随原始信号（响应快、但抖动抑制弱）；
    越小越平滑（稳定、但响应滞后）。首帧直接用原始值初始化，
    避免从 0 缓升导致的假"需要移动"。

    本平滑器同时跟踪**动作类型**与**差分量**：动作类型用"多数表决 +
    迟滞"处理（离散量不能做加权平均），差分量做标准 EMA。
    """

    def __init__(self, alpha: float = 0.35) -> None:
        """
        Args:
            alpha: 平滑系数，取值 (0, 1]。
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha 必须落在 (0, 1]，实际为 {alpha}")
        self.alpha = alpha
        self._value: float | None = None

    def reset(self) -> None:
        """清空状态。切换拍摄会话时应调用。"""
        self._value = None

    def update(self, value: float) -> float:
        """喂入一个新观测值，返回平滑后的值。"""
        if self._value is None:
            self._value = float(value)
        else:
            self._value = self.alpha * float(value) + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


class CommandDebouncer:
    """指令防抖器：连续一致 + 最小间隔。

    状态机说明::

        待确认候选 (pending) --连续 N 帧--> 采纳 (current)
                 ^                              |
                 |______ 出现其它指令则重置 _____|

    只有 ``pending`` 累计到 ``required_consecutive_frames`` 才会切换 ``current``，
    且切换后 ``min_interval_ms`` 内不再接受新切换（除非是"保持"这类
    安全指令，见 :meth:`_should_bypass_interval`）。
    """

    def __init__(self, cfg: StabilizationConfig | None = None) -> None:
        cfg = cfg or StabilizationConfig()
        self.ema = EmaSmoother(alpha=cfg.ema.alpha)
        self.ema_enabled = cfg.ema.enabled
        self.debounce_enabled = cfg.debounce.enabled
        self.required_frames = cfg.debounce.required_consecutive_frames
        self.min_interval_ms = cfg.debounce.min_interval_ms

        self._current: ActionCommand | None = None
        self._pending_action: ActionType | None = None
        self._pending_count = 0
        self._last_change_ms: int | None = None
        self._frames_seen = 0
        # [D-11 修复 2026-09-29] 候选动作**首次**出现的时间戳（ms）。
        # 用于区分"持续的新意图"与"来回摆动"：前者会随时间推移不断累积，
        # 后者会在候选被换掉时归零。见 stabilize() 的闩锁释放逻辑。
        self._pending_since_ms: int | None = None
        # 本次"卡住"周期内是否已用过安全方向豁免。
        # 防止在同一次卡顿里被反复触发（实测：4 动作循环下会翻转 33 次）。
        self._safety_release_used = False
        # 安全方向豁免的**全局速率限制**时间戳。
        # [D-11] 单靠 ``_safety_release_used`` 不够：每次成功切换都会把它
        # 重置，于是在"全安全方向循环"的合成序列上豁免会被反复重新武装，
        # 实测 40 帧内切换 7 次（AC-06 基线只有 19 次/240 帧）。
        # 因此再加一道与具体切换解耦的冷却。
        self._last_safety_release_ms: int | None = None
        # 更长horizon的动作直方图（用于判别"少数派噪声"与"持续新意图"）。
        # [D-11] 这是解开"抗抖 vs 活性"矛盾的关键判据——见 stabilize()。
        self._long_window: list[ActionType] = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置全部状态。"""
        self.ema.reset()
        self._current = None
        self._pending_action = None
        self._pending_count = 0
        self._last_change_ms = None
        self._frames_seen = 0
        self._pending_since_ms = None
        self._safety_release_used = False
        self._last_safety_release_ms = None
        self._long_window.clear()

    # ------------------------------------------------------------------
    def stabilize(self, command: ActionCommand, timestamp_ms: int) -> StabilizedCommand:
        """对原始指令做防抖，返回最终下发的指令。

        Args:
            command: 未经防抖的原始指令（FR-04 的输出）。
            timestamp_ms: 当前帧时间戳。

        Returns:
            防抖结果，含被抑制原因与统计字段，供指标归因使用。

        行为保证：
            - 首帧无条件采纳（否则用户会看到一段"无指令"的空窗）；
            - ``hold`` 指令允许在最小间隔内立即生效（"停下来"是安全方向，
              延迟执行会造成用户继续错误移动）；
            - 其余情况严格遵循三级防抖。
        """
        self._frames_seen += 1
        now_ms = timestamp_ms

        # --- 第 1 级：EMA 平滑差分量 ---
        smoothed: float | None = None
        if self.ema_enabled and command.magnitude_raw is not None:
            smoothed = self.ema.update(command.magnitude_raw)

        # 首帧：直接采纳
        if self._current is None:
            self._current = command
            self._pending_action = command.action
            self._pending_count = self.required_frames
            self._last_change_ms = timestamp_ms
            return self._build(command, command, changed=True, suppressed=None, smoothed=smoothed)

        # --- 不需要防抖：直接采纳 ---
        if not self.debounce_enabled:
            changed = command.action is not self._current.action
            if changed:
                self._last_change_ms = timestamp_ms
            self._current = command
            return self._build(command, command, changed=changed, suppressed=None, smoothed=smoothed)

        prev = self._current
        same_as_current = command.action is prev.action

        # --- 第 2 级：连续 N 帧一致性 ---
        if same_as_current:
            # 与当前指令一致：清空 pending，保持稳定
            self._pending_action = command.action
            self._pending_count = self.required_frames
            self._pending_since_ms = timestamp_ms
            self._current = command
            self._long_window.append(command.action)
            if len(self._long_window) > _LONG_WINDOW_SIZE:
                del self._long_window[: len(self._long_window) - _LONG_WINDOW_SIZE]
            return self._build(command, command, changed=False, suppressed=None, smoothed=smoothed)

        # 与当前不同 → 累积候选。
        #
        # [D-11 修复 2026-09-29] **修复范围经过刻意收窄，原因如下。**
        #
        # 现象（真实素材，``web/samples/frames.json``，10 张素材顺序推入同一
        # 会话）::
        #
        #     raw 动作: closer, back, hold, back, back, right, back, closer, back, closer
        #     修复前  : closer × 10 帧（其中 6 帧的 raw 其实是「该后退」）
        #
        # 第 0 帧被无条件采纳后，后续 raw 虽多数正确（主体占高 0.61~0.95
        # 明显偏大，应 ``move_back``），但没有 3 个**相同**动作连续出现，
        # ``_pending_count`` 永远到不了 3 → **锁死在首帧动作**。
        #
        # **为什么"严格连续 N 帧"本身不能改**：
        #
        # FR-06（《需求说明.md》§FR-06）明确要求「连续 N 帧一致才切换」，
        # 验收指标是"指令切换频率 次/分钟"。若改成"滑窗 N 帧内多数即可"
        # （允许 1 帧异类），则 ``closer, back, closer, back, …`` 这种
        # 纯交替噪声每帧窗口都含 2 票，会被判为达成共识 → **每帧翻转**，
        # 比修复前更差。实测确认：真实序列与纯交替序列在 3 帧窗口下的
        # 多数票分布**完全同形**，无法靠窗口算术区分。
        #
        # 因此本次**不放松**第 2 级，只修一处**内部自相矛盾**：
        #
        #   :meth:`_should_bypass_interval` 声明 ``hold`` 与 ``move_back``
        #   是安全方向、应豁免最小切换间隔。但原实现里候选动作**根本过不了
        #   第 2 级**，这条豁免永远不可达——是一处"写了但不会生效"的死代码。
        #
        # 修复：允许**安全方向**（hold / move_back）的候选在持续超过
        # ``min_interval_ms`` 后穿透第 2 级。语义上完全自洽——
        # 「停下来」和「往后退」是纠错方向，及时生效比严格计数更重要；
        # 而激进方向（靠近 / 横移 / 俯仰）仍严格要求连续 N 帧，
        # 抗抖指标（AC-06 切换频率）不受影响。
        if command.action is self._pending_action:
            self._pending_count += 1
        else:
            self._pending_action = command.action
            self._pending_count = 1
            self._pending_since_ms = now_ms

        # 记录到长窗（用于判别"持续新意图" vs "循环噪声"）
        self._long_window.append(command.action)
        if len(self._long_window) > _LONG_WINDOW_SIZE:
            del self._long_window[: len(self._long_window) - _LONG_WINDOW_SIZE]

        if self._pending_count < self.required_frames:
            # 例外：安全方向 + 系统已被"卡住" + 候选在长窗内占多数 → 放行。
            #
            # 判据用 ``now_ms - self._last_change_ms``（距**上次采纳**多久），
            # 而不是"距本候选首次出现多久"。原因（实测踩坑）：
            #
            #   ``closer, hold, closer, hold, …`` 交替时，``hold`` 每次重新
            #   出现都会重置 ``_pending_since_ms``，相邻两次间隔 800ms 永远
            #   达不到 1000ms 阈值 → 豁免仍不可达。但系统其实已经**卡在
            #   初始动作上 4400ms**，这才是"该放行"的语义。
            #
            # 用 ``_last_change_ms`` 则正确表达"我多久没改口了"。
            stuck_ms = (
                now_ms - self._last_change_ms
                if self._last_change_ms is not None
                else None
            )
            # 长窗多数：候选动作须在最近 _LONG_WINDOW_SIZE 帧里占 > 1/2。
            # 这是区分"循环噪声"与"真实意图变化"的关键——全方向循环里
            # 任一动作只占 ~1/4，达不到多数。
            n_this = sum(1 for a in self._long_window if a is command.action)
            long_majority = n_this * 2 > len(self._long_window)
            cooldown_ok = (
                self._last_safety_release_ms is None
                or (now_ms - self._last_safety_release_ms) >= _SAFETY_RELEASE_COOLDOWN_MS
            )
            persistent_safety = (
                self._should_bypass_interval(command)
                and stuck_ms is not None
                and stuck_ms >= self.min_interval_ms
                and not self._safety_release_used
                and cooldown_ok
                and long_majority
            )
            if not persistent_safety:
                # 尚未达成一致，抑制
                return self._build(
                    command, prev, changed=False, suppressed=SuppressReason.N_FRAME_CONSENSUS, smoothed=smoothed
                )
            # 标记已用；下一次真正的采纳会把它清掉（见下方切换分支）
            self._safety_release_used = True
            self._last_safety_release_ms = now_ms

        # --- 第 3 级：最小切换间隔 ---
        if not self._should_bypass_interval(command) and self._last_change_ms is not None:
            elapsed = timestamp_ms - self._last_change_ms
            if elapsed < self.min_interval_ms:
                return self._build(
                    command, prev, changed=False, suppressed=SuppressReason.MIN_INTERVAL, smoothed=smoothed
                )

        # 通过所有关卡：切换
        self._current = command
        self._last_change_ms = now_ms
        self._safety_release_used = False
        # 注意：切换时 effective 必须是**新**指令。早期实现误传了 ``prev``，
        # 导致 is_changed=True 却返回旧指令——契约自相矛盾，
        # 且会让渲染层显示"已切换"标记配上过时的动作文本。
        return self._build(command, command, changed=True, suppressed=None, smoothed=smoothed)

    # ------------------------------------------------------------------
    def _should_bypass_interval(self, command: ActionCommand) -> bool:
        """判断是否可绕过最小切换间隔。

        例外规则：**"保持"与"后退/远离"类指令豁免**。
        理由：这两类是安全方向——用户停错位置或距离过近时，让他继续
        等待 1.5 秒才收到纠正，体验上明显更差，且可能拍出废片。
        相反，"继续往前"这类激进指令必须受间隔约束。
        """
        return command.action in (ActionType.HOLD, ActionType.MOVE_BACK)

    def _build(
        self,
        raw: ActionCommand,
        effective: ActionCommand,
        changed: bool,
        suppressed: SuppressReason | None,
        smoothed: float | None,
    ) -> StabilizedCommand:
        """组装契约对象。"""
        since_change = (
            float(self._last_change_ms) if self._last_change_ms is not None else 0.0
        )
        # 注意：since_last_change_ms 语义是"距上次切换经过多久"，
        # 但此处无法拿到 timestamp，故由调用方通过 tracker 统计；
        # 契约该字段保留为"切换时的墙钟标记"，详见 to_stats()。
        return StabilizedCommand(
            command=effective,
            raw_command=raw,
            is_changed=changed,
            suppressed_by=suppressed,
            consecutive_frames=self._pending_count if not changed else 1,
            smoothing=smoothed,
            since_last_change_ms=since_change,
        )

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, float]:
        """返回防抖过程的累计统计，供指标报告使用。"""
        return {
            "frames_seen": float(self._frames_seen),
            "ema_alpha": self.ema.alpha if self.ema_enabled else 0.0,
            "required_frames": float(self.required_frames),
            "min_interval_ms": float(self.min_interval_ms),
            "ema_final": float(self.ema.value) if self.ema.value is not None else 0.0,
        }
