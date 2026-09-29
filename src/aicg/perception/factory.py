"""感知层工厂：把配置翻译成具体的感知后端实例。

对应需求：NFR-M2（配置外置）、NFR-R1（模型失败不卡死）
把"配置 -> 实现"的映射集中在此，避免各调用点重复拼装参数。
"""

from __future__ import annotations

from pathlib import Path

from .base import BasePerception
from .detector import YoloPerception
from .saliency import RulePerception

# 素材名 → 推荐后端。依据见 tests/fixtures/README.md（"双源分流"）。
#
# **为什么需要这张表**：合成素材由规则后端的肤色/形状先验检出，YOLO 对
# "代码画出来的人形"零响应；真实素材才能被 YOLO 检出。若不按素材切换
# 后端，会得到"全程降级、0% 降幅"的无意义指标（实测踩过）。把它放在
# 库里而非脚本里，是为了让脚本、CLI、API 共用同一套判断，且可被测试覆盖。
_FIXTURE_BACKEND_HINTS: dict[str, str] = {
    "walk_towards": "rule",
    "handheld_jitter": "rule",
    "synthetic_portrait": "rule",
    "real_photo_zoom": "yolo",
}


def infer_backend_for_source(source: str) -> str | None:
    """由素材名推断推荐后端；无法判断时返回 ``None``（交由配置决定）。

    Args:
        source: 帧源标识（视频路径 / ``camera:0`` / 图片目录）。

    Returns:
        ``"rule"`` / ``"yolo"``，或 ``None`` 表示无推荐。
    """
    if source.startswith("camera:"):
        return None
    return _FIXTURE_BACKEND_HINTS.get(Path(source).stem)


def perception_from_settings(settings=None) -> BasePerception:
    """从全局配置构造感知后端。

    行为契约（NFR-R1）：
    - 配置 ``backend=auto`` 且 YOLO 权重可加载 → 返回 :class:`YoloPerception`；
    - 权重缺失 / ultralytics 未安装 / 加载异常 → **返回** :class:`RulePerception`，
      不抛异常，保证引导引擎总能启动。
    """
    if settings is None:
        from ..settings import get_settings

        settings = get_settings()

    pcfg = settings.perception
    det = pcfg.detector
    backend = pcfg.effective_backend()

    if backend == "yolo":
        yolo = YoloPerception(
            weights=str(det.resolved_weights()),
            device=det.device,
            conf_threshold=det.conf_threshold,
            iou_threshold=det.iou_threshold,
            subject_labels=list(det.subject_labels),
            prefer_person=det.prefer_person,
        )
        if yolo.available:
            return yolo

    return RulePerception(
        saliency_work_size=pcfg.saliency.work_size,
        prefer_person=det.prefer_person,
    )


__all__ = ["perception_from_settings", "infer_backend_for_source"]
