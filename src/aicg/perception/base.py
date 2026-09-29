"""感知层抽象基类。

对应需求：FR-02（主体确认）、NFR-M1（模块可替换）
对应文档：《技术方案.md》§2.2 感知层选型

设计原则：上层（composition / pipeline）只依赖本模块的抽象，
不依赖任何具体模型实现，从而满足"替换感知实现，其他层无需改动"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..schemas import PerceptionResult


class BasePerception(ABC):
    """感知后端协议。

    实现类需保证：
    1. 永不抛出未捕获异常导致引导循环中断（NFR-R1）——失败时应返回带
       ``degraded=True`` 的空结果；
    2. 输出 ``subjects`` 中至多一个 ``is_primary=True``；
    3. ``bbox`` 一律为归一化坐标。
    """

    #: 后端标识，会写入 ``PerceptionResult.backend``
    name: str = "base"

    #: 该后端依赖的模型版本，写入 ``PerceptionResult.model_versions``
    def model_versions(self) -> dict[str, str]:
        return {}

    @abstractmethod
    def infer(self, image: np.ndarray, frame_id: int, timestamp_ms: int) -> PerceptionResult:
        """对单帧图像做推理。

        Args:
            image: BGR 图像数组。
            frame_id: 帧序号。
            timestamp_ms: 帧时间戳。

        Returns:
            感知结果。实现内部必须自行兜底，不得向上抛异常。
        """

    def warmup(self, image_shape: tuple[int, int] | None = None) -> None:
        """预热（加载权重、跑一次空推理）。默认无操作。

        Args:
            image_shape: 可选的实际推理尺寸 ``(高, 宽)``。

        **为什么需要传尺寸**：实测发现 Ultralytics 的推理耗时与输入尺寸
        强相关——用 480x640 预热后，首次跑 480x270 仍要 181ms，之后才
        降到 ~25ms。原因是推理引擎会按输入尺寸做一次规划/编译，尺寸一变
        就得重来。因此对"一次调用只推一帧"的场景（CLI、单图打分），
        必须用真实尺寸预热，否则报告的延迟会虚高约 7 倍。
        """
        return None

    def close(self) -> None:
        """释放资源。默认无操作。"""
        return None


class SubjectSelector:
    """主体选择策略：多主体时决定谁是"当前拍摄主体"。

    抽成独立策略类的原因是：这是**产品决策**而非模型能力——
    调研指出 Doka 在"多主体/无主体"场景下会失败，故做了人工兜底；
    本类只负责自动选择部分，人工兜底由上层 pipeline 处理（FR-02）。
    """

    def __init__(self, prefer_person: bool = True) -> None:
        self.prefer_person = prefer_person

    def select(self, candidates: list[tuple[str, tuple[float, float, float, float], float]]) -> int:
        """从候选中选出主体索引。

        Args:
            candidates: 列表元素为 ``(label, bbox, confidence)``。

        Returns:
            选中的索引；候选为空时返回 -1。

        选择规则（按优先级）：
            1. 人优先（若 prefer_person）
            2. 面积最大（主体通常最占画面）
            3. 置信度最高
            4. 靠近画面中心
        """
        if not candidates:
            return -1

        from ..utils.image import bbox_area, bbox_center

        def sort_key(item: tuple[int, tuple[str, tuple[float, float, float, float], float]]):
            idx, (label, bbox, conf) = item
            is_person = 1 if (self.prefer_person and label == "person") else 0
            area = bbox_area(bbox)
            cx, cy = bbox_center(bbox)
            # 距画面中心的距离，越小越好 -> 取负值参与降序排序
            center_dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            return (is_person, round(area, 4), round(conf, 4), -center_dist)

        ranked = sorted(enumerate(candidates), key=sort_key, reverse=True)
        return ranked[0][0]
