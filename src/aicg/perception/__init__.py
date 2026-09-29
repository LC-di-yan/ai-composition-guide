"""感知层：主体检测与显著性估计。

对应需求：FR-02（主体确认）
对应文档：《技术方案.md》§2.2

双实现策略（「真实模型 + 可降级桩并存」决策）：
- :class:`YoloPerception` —— 真实模型，权重存在时优先使用；
- :class:`RulePerception` —— 无依赖兜底，保证权重缺失时链路不断。

:func:`build_perception` 按配置与权重存在性自动选择。
"""

from .base import BasePerception, SubjectSelector
from .detector import YoloPerception
from .factory import infer_backend_for_source, perception_from_settings
from .saliency import RulePerception

__all__ = [
    "BasePerception",
    "SubjectSelector",
    "YoloPerception",
    "RulePerception",
    "build_perception",
    "perception_from_settings",
    "infer_backend_for_source",
]


def build_perception(backend: str = "auto", **kwargs) -> BasePerception:
    """构造感知后端（底层工厂，按裸参数构造）。

    Args:
        backend: ``yolo`` / ``rule`` / ``auto``。
            ``auto``：YOLO 权重可加载则用 YOLO，否则自动降级为 rule。
        **kwargs: 透传给具体后端的构造参数（两种后端共享的参数会被各自忽略
            不支持的项）。**yolo 后端必须提供 ``weights``**。

    Returns:
        可用的感知后端实例，**保证返回一个可用对象**（NFR-R1）。

    .. warning::

        **上层业务代码不要直接调用本函数**，请使用
        :func:`aicg.perception.factory.perception_from_settings`。

        原因：本函数只接受裸参数，**不会读取配置**中的权重路径、
        置信度阈值、IOU 阈值与 device。若无参调用
        ``build_perception("auto")``，会因缺少 ``weights`` 而静默退回
        规则后端，产生"配置写着 yolo、实际跑着 rule"的静默降级——
        指标与性能结论都会因此失真，且难以察觉。

        本项目已在 ``benchmark_latency.py`` 与 ``guiding_loop`` 中
        踩过该坑，因此将本函数保留为**底层实现细节**。
    """
    if backend == "rule":
        return RulePerception(**_pick(kwargs, RulePerception))

    if backend == "yolo":
        return YoloPerception(**_pick(kwargs, YoloPerception))

    # auto：优先 YOLO，失败即降级
    try:
        yolo = YoloPerception(**_pick(kwargs, YoloPerception))
        if yolo.available:
            return yolo
        return RulePerception(**_pick(kwargs, RulePerception))
    except Exception:  # noqa: BLE001
        return RulePerception(**_pick(kwargs, RulePerception))


def _pick(kwargs: dict, cls: type) -> dict:
    """只保留目标类构造函数接受的参数。"""
    import inspect

    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    picked = {k: v for k, v in kwargs.items() if k in params}
    # 兼容配置里的命名差异：yolo 用 weights，rule 用 saliency_work_size
    if cls is YoloPerception and "weights" in picked:
        picked["weights"] = str(picked["weights"])
    return picked
