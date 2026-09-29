"""图像与几何通用工具。

本模块是唯一允许直接操作像素坐标的地方：对外统一暴露**归一化坐标**，
内部按需在像素空间与归一化空间之间转换。

对应需求：NFR-M1（模块可替换）、NFR-O1（结构化中间表示）
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from ..schemas.perception import BBox

# --------------------------------------------------------------------------
# 图像读取
# --------------------------------------------------------------------------


class ImageLoadError(RuntimeError):
    """图像无法解码时抛出。"""


def imread_unicode(path: str | Path) -> np.ndarray:
    """读取图像，兼容中文路径（``cv2.imread`` 在 Windows 上不支持非 ASCII 路径）。"""
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"图像不存在: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"图像解码失败: {p}")
    return img


def imwrite_unicode(path: str | Path, image: np.ndarray) -> None:
    """写出图像，兼容中文路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix if p.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ImageLoadError(f"图像编码失败: {p}")
    buf.tofile(str(p))


def decode_base64_image(data_uri: str) -> np.ndarray:
    """解码 ``data:image/...;base64,xxx`` 或裸 base64 字符串。"""
    payload = data_uri.split(",", 1)[1] if "," in data_uri[:64] else data_uri
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageLoadError(f"base64 解码失败: {e}") from e
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError("base64 内容不是有效图像")
    return img


def load_image(image_ref: str | Path) -> np.ndarray:
    """统一图像加载入口：支持本地路径与 base64 data URI。

    Args:
        image_ref: 本地文件路径，或 ``data:image/png;base64,...`` 形式的字符串。

    Returns:
        OpenCV BGR 图像数组。

    Raises:
        ImageLoadError: 无法读取或解码。
    """
    s = str(image_ref)
    if s.startswith("data:") or (len(s) > 512 and "/" not in s[:64]):
        return decode_base64_image(s)
    return imread_unicode(s)


def resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    """按目标宽度等比缩放；若原图更窄则原样返回（不放大）。

    Args:
        image: 输入图像。
        target_width: 目标宽度（px）。

    Returns:
        缩放后的图像（可能与输入是同一对象）。
    """
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / float(w)
    return cv2.resize(image, (target_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def to_gray(image: np.ndarray) -> np.ndarray:
    """转灰度；已是单通道则原样返回。"""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# 坐标转换
# --------------------------------------------------------------------------


def bbox_px_to_norm(bbox_px: tuple[float, float, float, float], size: tuple[int, int]) -> BBox:
    """像素框 → 归一化框。

    Args:
        bbox_px: ``[x1, y1, x2, y2]`` 像素坐标。
        size: ``[宽, 高]`` 像素尺寸。
    """
    w, h = size
    x1, y1, x2, y2 = bbox_px
    return (
        float(np.clip(x1 / w, 0.0, 1.0)),
        float(np.clip(y1 / h, 0.0, 1.0)),
        float(np.clip(x2 / w, 0.0, 1.0)),
        float(np.clip(y2 / h, 0.0, 1.0)),
    )


def bbox_norm_to_px(bbox: BBox, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """归一化框 → 像素框（整型，供绘图使用）。"""
    w, h = size
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * w)),
        int(round(y1 * h)),
        int(round(x2 * w)),
        int(round(y2 * h)),
    )


def bbox_center(bbox: BBox) -> tuple[float, float]:
    """归一化框中心点。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def bbox_height(bbox: BBox) -> float:
    return bbox[3] - bbox[1]


def bbox_area(bbox: BBox) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_iou(a: BBox, b: BBox) -> float:
    """两个归一化框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def clamp_bbox(bbox: BBox) -> BBox:
    """把框裁剪回 [0,1] 范围，并保证 x2>x1、y2>y1。"""
    x1, y1, x2, y2 = (float(np.clip(c, 0.0, 1.0)) for c in bbox)
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-6)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-6)
    return (x1, y1, x2, y2)


def crop_norm(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """按归一化框裁剪图像。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_norm_to_px(bbox, (w, h))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    return image[y1:y2, x1:x2]


# --------------------------------------------------------------------------
# 显著性（轻量、无模型依赖）
# --------------------------------------------------------------------------


def spectral_residual_saliency(gray: np.ndarray, work_size: int = 160) -> np.ndarray:
    """频谱残差显著性图（Spectral Residual, Hou & Zhang 2007）。

    选择该算法的原因：纯 numpy/FFT 实现，无需模型权重，可作为感知层的
    无依赖兜底，用于「视觉重点」估计（FR-02 / FR-07）。

    Args:
        gray: 单通道灰度图。
        work_size: 内部计算尺寸，越小越快。

    Returns:
        与 ``gray`` 同尺寸的显著性图，取值 0~1。
    """
    h, w = gray.shape[:2]
    scale = work_size / float(max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, (max(8, int(w * scale)), max(8, int(h * scale))))
    else:
        small = gray

    f = np.fft.fft2(small.astype(np.float32))
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)

    # 平均谱 -> 谱残差
    avg_log_amp = cv2.blur(log_amp, (3, 3))
    residual = log_amp - avg_log_amp

    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5)

    # 归一化到 0~1
    lo, hi = float(sal.min()), float(sal.max())
    sal = (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)

    if sal.shape != (h, w):
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    return sal


def to_bgr_heatmap(saliency: np.ndarray) -> np.ndarray:
    """把 0~1 显著性图转为 BGR 伪彩热力图，供可视化。"""
    u8 = np.clip(saliency * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def encode_png_base64(image: np.ndarray) -> str:
    """把图像编码为 PNG data URI，供接口返回。"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ImageLoadError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def bytes_to_array(raw: bytes) -> np.ndarray:
    """字节流 → 图像数组。"""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError("字节流不是有效图像")
    return img


def array_to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """图像数组 → 字节流。"""
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ImageLoadError(f"编码失败: {ext}")
    return buf.tobytes()


def to_bytesio(image: np.ndarray, ext: str = ".png") -> BytesIO:
    return BytesIO(array_to_bytes(image, ext))
