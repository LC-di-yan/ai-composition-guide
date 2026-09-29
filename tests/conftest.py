"""测试配置：路径注入与共享 fixture。

对应需求：NFR-M1（可测试性）
对应文档：《测试与验收.md》§1 测试分层

设计原则：
- **不依赖已安装的包**：通过 ``sys.path`` 注入 ``src/``，使 ``pytest``
  在未 ``pip install -e .`` 的环境也能直接跑；
- **不依赖网络/GPU**：所有测试只跑 CPU 路径，感知层用桩件或规则后端；
- **不依赖真实媒体文件**：合成图像由代码生成，保证 CI 可复现（NFR-O3）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def synthetic_portrait() -> np.ndarray:
    """合成人像帧：肤色身体 + 渐变背景，供感知与构图测试使用。

    尺寸 480x640（宽x高）；主体框归一化约 ``(0.40, 0.18, 0.62, 0.95)``。
    """
    h, w = 640, 480
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        v = int(200 - 90 * y / h)
        img[y, :] = (v + 30, v + 10, v)  # 冷色渐变天空

    gy = int(h * 0.75)
    rng = np.random.default_rng(3)
    img[gy:, :] = rng.integers(90, 130, size=(h - gy, w, 3), dtype=np.uint8)

    bh, bw = 0.77 * h, 0.22 * w
    cx, cy = 0.51 * w, 0.565 * h
    import cv2

    cv2.ellipse(img, (int(cx), int(cy)), (int(bw / 2), int(bh / 2)), 0, 0, 360, (150, 170, 215), -1)
    return img


@pytest.fixture
def settings_factory():
    """返回一个"构造配置"的工厂，便于每个测试定制参数而不污染全局单例。"""
    from aicg.settings import load_settings, reset_settings_cache

    created: list = []

    def _make(**overrides):
        reset_settings_cache()
        cfg = load_settings(overrides=overrides or None)
        created.append(cfg)
        return cfg

    yield _make
    reset_settings_cache()
