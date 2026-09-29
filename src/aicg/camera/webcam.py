"""本地摄像头 / 图片目录帧源。

对应需求：FR-05
- :class:`WebcamSource` 用于挂真机或笔记本摄像头，做现场演示（Web 双轨之一）；
- :class:`ImageFolderSource` 用于把一组图片当作「伪视频」，方便在没有视频
  素材时快速构造可复现的测试序列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from ..utils.image import imread_unicode
from .base import Frame, FrameSource

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class WebcamSource(FrameSource):
    """本地摄像头帧源（实时）。"""

    def __init__(
        self,
        device_index: int = 0,
        target_fps: float = 3.0,
        downsample_width: int | None = 480,
        max_frames: int | None = None,
        warmup_frames: int = 3,
    ) -> None:
        """
        Args:
            device_index: 摄像头索引。
            target_fps: 送 AI 的抽帧率；通过取帧间隔实现，而非改变摄像头采集率。
            downsample_width: 降采样目标宽度。
            max_frames: 最多产出帧数；None 表示直到调用方停止。
            warmup_frames: 打开后丢弃的前若干帧（自动曝光/白平衡未收敛）。
        """
        super().__init__(downsample_width)
        self.device_index = device_index
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.warmup_frames = warmup_frames

        self._cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError(f"摄像头 {device_index} 无法打开（被占用或无权限）")
        for _ in range(max(0, warmup_frames)):
            self._cap.read()

        self._interval_ms = int(round(1000.0 / target_fps)) if target_fps > 0 else 0

    def _iter_raw(self) -> Iterator[Frame]:
        out_id = 0
        while True:
            ok, img = self._cap.read()
            if not ok or img is None:
                break
            yield Frame(
                frame_id=out_id,
                image=img,
                timestamp_ms=out_id * self._interval_ms,
                meta={"device_index": self.device_index, "source": "webcam"},
            )
            out_id += 1
            if self.max_frames is not None and out_id >= self.max_frames:
                break

        self.close()

    def close(self) -> None:
        if not self._closed:
            self._cap.release()
        super().close()


class ImageFolderSource(FrameSource):
    """图片目录帧源：把有序图片序列当作视频回放。

    用于无视频素材时构造可复现的测试序列（指标复现性优于随机图）。
    """

    def __init__(
        self,
        folder: str | Path,
        target_fps: float = 3.0,
        downsample_width: int | None = 480,
        max_frames: int | None = None,
        recursive: bool = False,
    ) -> None:
        super().__init__(downsample_width)
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"目录不存在: {self.folder}")
        self.target_fps = target_fps
        self.max_frames = max_frames

        pattern = "**/*" if recursive else "*"
        self._files = sorted(
            p for p in self.folder.glob(pattern) if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        if not self._files:
            raise FileNotFoundError(f"目录下没有可用图片: {self.folder}")

        self._interval_ms = int(round(1000.0 / target_fps)) if target_fps > 0 else 0

    @property
    def total_frames(self) -> int | None:
        n = len(self._files)
        return min(n, self.max_frames) if self.max_frames else n

    def _iter_raw(self) -> Iterator[Frame]:
        for i, p in enumerate(self._files):
            if self.max_frames is not None and i >= self.max_frames:
                break
            img = imread_unicode(p)
            yield Frame(
                frame_id=i,
                image=img,
                timestamp_ms=i * self._interval_ms,
                meta={"path": str(p), "source": "image_folder"},
            )
        self.close()


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")
