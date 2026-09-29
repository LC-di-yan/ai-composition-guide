"""语言层：把结构化构图结果翻译成人话（FR-07 / FR-08 / FR-10）。

对应文档：《技术方案.md》§2.4 语言层

**为什么语言层不放进实时回路：**

调研反复强调"魔力 = 60% 实时工程 + 30% 评分选点 + 10% 语言包装"。
语言只占 10%，因此它必须**跑在回路之外**——否则 VLM 的网络往返
（数百毫秒~数秒）会直接摧毁 NFR-P1 的端到端延迟预算。

落地方式：
- **逐帧**：指令文本由 ``composition.differ`` 直接产出（纯规则，微秒级）；
- **关键节点**（拍摄完成）：才调用 ``VlmClient`` 生成有"口吻"的解说。

因此本层的 ``VlmClient`` 是**按需调用**，与逐帧回路解耦。
"""

from __future__ import annotations

from .tts import AsyncTtsEngine, BaseTtsEngine, NullTtsEngine, build_tts_engine
from .vlm_client import NarrationResult, PersonaAssets, VlmClient, build_facts

__all__ = [
    "AsyncTtsEngine",
    "BaseTtsEngine",
    "NarrationResult",
    "NullTtsEngine",
    "PersonaAssets",
    "VlmClient",
    "build_facts",
    "build_tts_engine",
    "build_vlm_client",
]


def build_vlm_client(cfg=None) -> VlmClient:
    """按配置构建 VLM 客户端（默认从全局配置读取语言层段）。

    Args:
        cfg: ``LanguageConfig``。为 None 时读取全局配置。
    """
    if cfg is None:
        from ..settings import get_settings

        cfg = get_settings().language
    return VlmClient(cfg)
