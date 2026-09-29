"""视频文件帧源。

对应需求：FR-05（实时引导循环）
对应文档：《开发计划.md》M3 交付物「视频 Demo（录屏，帧序列模拟实时）」

这是「录屏回放」轨道的实现：按 ``target_fps`` 抽取视频帧，逻辑时间戳
按帧间隔推算（而非真实墙钟），因此指标可复现（NFR-O3）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from ..utils.image import imread_unicode  # noqa: F401  (保证中文路径能力可用)
from .base import Frame, FrameSource


class VideoFileSource(FrameSource):
    """从视频文件按目标帧率抽帧。

    用法::

        with VideoFileSource("demo.mp4", target_fps=3.0) as src:
            for frame in src:
                ...
    """

    def __init__(
        self,
        path: str | Path,
        target_fps: float = 3.0,
        downsample_width: int | None = 480,
        max_frames: int | None = None,
    ) -> None:
        """
        Args:
            path: 视频文件路径（兼容中文路径）。
            target_fps: 目标抽帧率（FR-05：1~5 FPS）。视频原生 FPS 低于该值时
                不插帧，只按原生帧率产出。
            downsample_width: 降采样目标宽度。
            max_frames: 最多产出多少帧；None 表示读到文件尾。
        """
        super().__init__(downsample_width)
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"视频不存在: {self.path}")
        self.target_fps = target_fps
        self.max_frames = max_frames

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"视频无法打开（编码不支持或文件损坏）: {self.path}")

        self.native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if self.native_fps <= 1e-3 or self.native_fps > 240:
            # 部分容器 FPS 元数据不可靠，退化为 25
            self.native_fps = 25.0
        self.native_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # 抽帧步长：原生帧率高于目标帧率时跳帧
        self._step = max(1, int(round(self.native_fps / self.target_fps))) if self.target_fps > 0 else 1
        # 逻辑帧间隔（ms），用于生成可复现的时间戳
        self._interval_ms = int(round(1000.0 / min(self.target_fps, self.native_fps)))

    @property
    def total_frames(self) -> int | None:
        if self.native_total <= 0:
            return None
        n = self.native_total // self._step
        return min(n, self.max_frames) if self.max_frames else n

    @property
    def duration_s(self) -> float:
        """视频原始时长（秒）。"""
        return self.native_total / self.native_fps if self.native_fps > 0 else 0.0

    def _iter_raw(self) -> Iterator[Frame]:
        out_id = 0
        native_idx = 0
        while True:
            ok = self._cap.grab()
            if not ok:
                break

            if native_idx % self._step == 0:
                ok, img = self._cap.retrieve()
                if not ok or img is None:
                    native_idx += 1
                    continue
                yield Frame(
                    frame_id=out_id,
                    image=img,
                    # 逻辑时间戳：保证同一输入视频每次运行的指标完全一致
                    timestamp_ms=out_id * self._interval_ms,
                    meta={
                        "native_index": native_idx,
                        "native_fps": round(self.native_fps, 3),
                        "source": str(self.path),
                    },
                )
                out_id += 1
                if self.max_frames is not None and out_id >= self.max_frames:
                    break

            native_idx += 1

        self.close()

    def close(self) -> None:
        if not self._closed:
            self._cap.release()
        super().close()
