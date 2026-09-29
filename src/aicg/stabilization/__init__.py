"""指令防抖层（FR-06）。

对应需求：FR-06（指令防抖）、NFR-P2（引导稳定性）
对应文档：《技术方案.md》§3 关键工程约束

这是调研认定"技术护城河"所在的模块——大多数人能做出 demo，
但做不出"变动中仍准确且不抖"的体验。

三级串联：EMA 平滑 → 连续 N 帧一致 → 最小切换间隔。
"""

from .debouncer import CommandDebouncer, EmaSmoother

__all__ = ["CommandDebouncer", "EmaSmoother", "build_debouncer"]


def build_debouncer(settings=None) -> CommandDebouncer:
    """按配置构造防抖器。"""
    if settings is None:
        from ..settings import get_settings

        settings = get_settings()
    return CommandDebouncer(cfg=settings.stabilization)
