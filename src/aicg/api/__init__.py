"""API 层：FastAPI 服务，把五层能力暴露为 HTTP/WebSocket 接口。

对应文档：《数据模型与接口.md》§3 接口清单
对应需求：FR-01~FR-09、NFR-R4、NFR-O3

两种传输形态服务两条交付轨道：

- **REST**（``routes.build_router``）—— 录屏轨与调试用：一问一答，易测；
- **WebSocket**（``routes.build_ws_router``）—— Web 轨：连上后持续推帧，
  省掉每帧 HTTP 往返，适合真跑实时引导。

启动方式::

    uvicorn aicg.api.app:create_app --factory --reload
    # 或
    python -m aicg.cli serve
"""

from __future__ import annotations

from .app import API_VERSION, AppState, create_app
from .image_ref import ImageDecodeError, decode_image_ref, encode_image_png_base64
from .mappers import snapshot_to_response
from .session_store import InMemorySessionStore, Session, SessionStore

__all__ = [
    "API_VERSION",
    "AppState",
    "ImageDecodeError",
    "InMemorySessionStore",
    "Session",
    "SessionStore",
    "create_app",
    "decode_image_ref",
    "encode_image_png_base64",
    "snapshot_to_response",
]
