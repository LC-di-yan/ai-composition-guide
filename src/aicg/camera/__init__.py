"""帧源层：把「从哪里取帧」与「如何推理」解耦。

支持两种交付轨道（对应「Web + 录屏双轨」决策）：

- 录屏回放轨：:class:`VideoFileSource` / :class:`ImageFolderSource` —— 指标可复现
- 实时演示轨：:class:`WebcamSource` —— 可现场演示
"""

from .base import Frame, FrameSource
from .video_file import VideoFileSource
from .webcam import ImageFolderSource, WebcamSource

__all__ = [
    "Frame",
    "FrameSource",
    "VideoFileSource",
    "WebcamSource",
    "ImageFolderSource",
    "open_source",
]


def open_source(
    ref: str,
    target_fps: float = 3.0,
    downsample_width: int | None = 480,
    max_frames: int | None = None,
) -> FrameSource:
    """按引用类型自动选择帧源。

    Args:
        ref: 视频文件路径、图片目录路径，或 ``camera:<index>``（如 ``camera:0``）。
        target_fps: 目标抽帧率。
        downsample_width: 降采样目标宽度。
        max_frames: 最多产出帧数。

    Returns:
        对应的 :class:`FrameSource` 实例。

    Raises:
        ValueError: 引用无法识别。
        FileNotFoundError: 路径不存在。
    """
    if ref.startswith("camera:"):
        idx = int(ref.split(":", 1)[1] or 0)
        return WebcamSource(idx, target_fps, downsample_width, max_frames)

    from pathlib import Path

    p = Path(ref)
    if p.is_dir():
        return ImageFolderSource(p, target_fps, downsample_width, max_frames)
    if p.is_file():
        return VideoFileSource(p, target_fps, downsample_width, max_frames)
    raise FileNotFoundError(f"帧源不存在: {ref}")
