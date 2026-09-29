"""帧源抽象层。

对应需求：FR-05（实时引导循环）
对应文档：《目录结构.md》分层规则 —— 本层是最底层，不依赖任何其他业务层。

设计动机：交付形态为「Web + 录屏双轨」，因此把"从哪里取帧"抽象成
``FrameSource`` 协议，使同一套引导引擎既能跑视频回放（指标可复现），
也能挂浏览器摄像头（现场演示），无需改动上层任何代码。

对外只暴露 :class:`Frame`，其中 ``image`` 为 OpenCV BGR 数组。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..utils.image import resize_keep_aspect


@dataclass
class Frame:
    """一帧图像及其元信息。"""

    frame_id: int
    image: np.ndarray
    """BGR 图像数组。"""

    timestamp_ms: int
    """帧时间戳（Unix ms）。视频回放时为按 FPS 推算的逻辑时间戳。"""

    capture_ms: float = 0.0
    """本帧的取帧耗时（ms），用于延迟分解。"""

    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        """``[宽, 高]``，单位 px。"""
        h, w = self.image.shape[:2]
        return (w, h)


class FrameSource(ABC):
    """帧源协议。

    所有帧源必须实现 :meth:`frames`，以迭代器方式按需产出帧，避免一次性
    把整段视频读入内存。
    """

    def __init__(self, downsample_width: int | None = None) -> None:
        """
        Args:
            downsample_width: 降采样目标宽度（px）。None 表示不降采样。
                对应 FR-05「降采样至 320~640px」。
        """
        self.downsample_width = downsample_width
        self._closed = False

    @abstractmethod
    def _iter_raw(self) -> Iterator[Frame]:
        """子类实现：产出原始分辨率的帧。"""

    def frames(self) -> Iterator[Frame]:
        """产出（可能已降采样的）帧序列，并记录取帧耗时。"""
        for raw in self._iter_raw():
            t0 = time.perf_counter()
            img = raw.image
            if self.downsample_width:
                img = resize_keep_aspect(img, self.downsample_width)
            raw.image = img
            raw.capture_ms = (time.perf_counter() - t0) * 1000.0
            if self._closed:
                return
            yield raw

    def close(self) -> None:
        """释放资源。可重复调用。"""
        self._closed = True

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()

    # 供上层做能力探测
    @property
    def total_frames(self) -> int | None:
        """总帧数；未知时为 None（如实时摄像头）。"""
        return None
