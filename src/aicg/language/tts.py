"""语音播报（FR-10，P2 优先级）。

对应需求：FR-10（TTS 语音播报）
对应文档：《需求说明.md》FR-10

**M3 阶段状态：接口占位（P2 未落地）。**

调研结论：实时引导的语音播报**不能等 TTS 合成完成**——若把 TTS 放入
逐帧回路，合成耗时（数百毫秒）会直接击穿 NFR-P1 的延迟预算。正确做法
是"**文本先行、语音异步追赶**"：指令文本立即上屏，TTS 在后台线程合成，
渲染端丢弃过期的任务（stale drop）。

因此本模块只定义接口与"丢弃策略"，实际合成后端在 M5 之后接入。
`NullTtsEngine` 为默认实现：`speak()` 直接返回 None，不阻塞主回路。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..observability import get_logger

log = get_logger("language.tts")


@dataclass
class SpeechTask:
    """一次播报任务。"""

    text: str
    frame_id: int
    seq: int


class BaseTtsEngine(ABC):
    """TTS 引擎抽象接口。

    所有实现都必须是**非阻塞**的（``speak`` 立即返回），否则会污染实时回路。
    """

    name: str = "base"

    @abstractmethod
    def speak(self, text: str, frame_id: int = 0) -> str | None:
        """提交一段文本供播报。

        Returns:
            音频引用（路径 / URL / id）；不产生音频时返回 None。
        """

    def close(self) -> None:  # pragma: no cover - 可选钩子
        """释放资源。"""


class NullTtsEngine(BaseTtsEngine):
    """空实现：不产生音频，只保证接口存在（M3 默认）。

    这是**诚实降级**：需求 FR-10 属 P2，在 M3 里程碑中以占位形式存在，
    不会伪造一个听不见的音频文件。
    """

    name = "null"

    def __init__(self) -> None:
        self._spoken: list[SpeechTask] = []

    def speak(self, text: str, frame_id: int = 0) -> str | None:
        self._spoken.append(SpeechTask(text=text, frame_id=frame_id, seq=len(self._spoken)))
        log.debug("TTS(占位) 收到播报请求: frame=%d text=%s", frame_id, text)
        return None

    @property
    def history(self) -> list[SpeechTask]:
        """记录已接收的播报请求，便于测试断言（不产生音频）。"""
        return list(self._spoken)


class AsyncTtsEngine(BaseTtsEngine):
    """异步 TTS 引擎骨架（M5+ 接入真实合成后端时启用）。

    设计：单后台线程 + "最新任务胜出"（stale drop）。旧任务在提交新任务时
    被标记为过期，合成完成后直接丢弃，避免音频越播越滞后。
    """

    name = "async"

    def __init__(self, synthesizer=None) -> None:
        self._synth = synthesizer
        self._lock = threading.Lock()
        self._latest_seq = 0
        self._seq = 0
        self._closed = False

    def speak(self, text: str, frame_id: int = 0) -> str | None:
        if self._closed or self._synth is None:
            return None
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._latest_seq = seq
        # 真实实现应在后台线程执行以下逻辑：
        #   audio = self._synth(text)
        #   if seq != self._latest_seq: return None   # 已过期，丢弃
        # M3 阶段仅保留骨架，不启动线程（避免未受控的后台资源）。
        log.debug("AsyncTtsEngine(骨架) 提交任务 seq=%d frame=%d", seq, frame_id)
        return None

    def close(self) -> None:
        self._closed = True


def build_tts_engine(enabled: bool = False) -> BaseTtsEngine:
    """按配置构建 TTS 引擎。

    Args:
        enabled: 是否启用真实合成。M3 阶段恒为 False → 返回占位实现。
    """
    if not enabled:
        return NullTtsEngine()
    return AsyncTtsEngine()
