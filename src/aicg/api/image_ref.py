"""图像引用解码（API 层入参适配）。

对应文档：《数据模型与接口.md》§3.3 —— "帧图像传输方式：待确认"

**设计决策（原文档留白，此处定型）**：

`image_ref` 支持三种形态，按前缀自动识别：

============ =============================== ==========================
形态          写法                             适用场景
============ =============================== ==========================
base64       ``data:image/jpeg;base64,....``  前端 canvas 截帧（Web 轨）
base64(裸)   ``<base64 字符串>``               客户端自行编码，未带 mime
文件路径     ``tests/fixtures/x.jpg``          本地 Demo / 测试（录屏轨）
============ =============================== ==========================

**为什么同时支持路径**：录屏轨与测试需要直接喂本地文件，强制转 base64
只会让脚本更难读。但要**限制路径来源**——见下方安全说明。

安全边界（NFR-S2）：
    路径模式仅在 ``app.env`` 为 ``dev``/``test`` 时启用。生产环境若开放
    任意路径读，等于给出一个任意文件读取漏洞。默认配置 ``env=dev``，
    因此本地 Demo 可用；生产部署改为 ``env=prod`` 即自动关闭该通道。
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import cv2
import numpy as np

from ..observability import get_logger

log = get_logger("api.image")

_DATA_URI_PREFIX = "data:image/"

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

_MIN_BASE64_LEN = 64
"""合法图像 base64 的最小长度。一张哪怕 8x8 的 PNG 编码后也远超此值。"""

_B64_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
)


class ImageDecodeError(ValueError):
    """图像解码失败（参数非法，应由路由转成 4xx）。"""


def _decode_base64(payload: str) -> np.ndarray:
    """解码 base64 图像载荷（可带 data URI 前缀）。"""
    if payload.startswith(_DATA_URI_PREFIX):
        # 形如 data:image/jpeg;base64,xxxx
        sep = payload.find(",")
        if sep < 0:
            raise ImageDecodeError("data URI 缺少逗号分隔符")
        payload = payload[sep + 1 :]

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageDecodeError(f"base64 解码失败: {e}") from e

    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("base64 内容不是有效图像（cv2.imdecode 返回 None）")
    return img


def _decode_path(path_str: str, *, allow_path: bool) -> np.ndarray:
    """从本地路径读图。"""
    if not allow_path:
        raise ImageDecodeError(
            "生产环境（env=prod）不接受本地路径形式的 image_ref，"
            "请使用 base64 data URI 传输"
        )
    p = Path(path_str)
    if not p.exists():
        raise ImageDecodeError(f"图像路径不存在: {p}")
    # cv2.imread 在 Windows 上不支持非 ASCII 路径，改用 imdecode 兜底
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(f"无法解码图像文件: {p}")
    return img


def looks_like_base64(s: str) -> bool:
    """判断字符串是 base64 还是文件路径。

    **这里踩过一个真实的坑，务必注意**：第一版实现为了区分 Windows 路径，
    直接判定"含 ``/`` 即为路径"。但 **base64 字母表本来就包含 ``/``**
    （标准 Base64 的 64 个字符是 ``A-Za-z0-9+/``），于是所有真实图像的
    base64 都被误判成路径 → base64 输入全线失败。

    正确的判别思路：不看"含什么字符"，而看**是否具备路径的结构特征**：

    - 含 ``\\``（Windows 分隔符，base64 字母表**不含**它）；
    - 含 Windows 盘符模式 ``X:``；
    - 以 ``./`` 或 ``../`` 开头（相对路径）；
    - 以常见图像扩展名结尾（``.jpg`` 等）；
    - 长度过短（合法 base64 图像不可能只有几十字符）。

    **注意不要用"以 ``/`` 开头"来判路径**：标准 Base64 的字母表含 ``/``，
    JPEG 的 base64 恰好就以 ``/9j/`` 开头（这是个绝佳的陷阱）。
    真正的解码失败由 :func:`decode_image_ref` 抛出 ``ImageDecodeError`` 兜底。
    """
    if s.startswith(_DATA_URI_PREFIX):
        return True

    # 反斜杠是 Windows 路径独占特征（base64 字母表不含）
    if "\\" in s:
        return False

    # Windows 盘符：形如 C:\ 或 C:/
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return False

    # 相对路径前缀
    if s.startswith(("./", "../")):
        return False

    # 图像扩展名结尾（路径最强信号）
    if s.lower().endswith(_IMAGE_EXTS):
        return False

    # 太短的不可能是图像 base64
    if len(s) < _MIN_BASE64_LEN:
        return False

    # 剩余情况：检查前缀是否为合法 base64 字符集
    sample = s[:512]
    return all(c in _B64_CHARS for c in sample)


def decode_image_ref(image_ref: str, *, allow_path: bool = True) -> np.ndarray:
    """把 ``image_ref`` 解码为 BGR 图像数组。

    Args:
        image_ref: base64 / data URI / 本地路径。
        allow_path: 是否允许本地路径（生产环境应传 False，NFR-S2）。

    Returns:
        ``np.ndarray``，shape ``(H, W, 3)``，BGR。

    Raises:
        ImageDecodeError: 解码失败。路由层应转成 400。
    """
    if not image_ref or not image_ref.strip():
        raise ImageDecodeError("image_ref 为空")

    ref = image_ref.strip()
    if looks_like_base64(ref):
        return _decode_base64(ref)
    return _decode_path(ref, allow_path=allow_path)


def encode_image_png_base64(image: np.ndarray) -> str:
    """把图像编码为 data URI（供响应里回传标注图）。"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ImageDecodeError("图像编码失败")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


__all__ = [
    "ImageDecodeError",
    "decode_image_ref",
    "encode_image_png_base64",
    "looks_like_base64",
]
