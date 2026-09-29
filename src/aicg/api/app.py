"""FastAPI 应用工厂与依赖装配。

对应文档：《数据模型与接口.md》§3 接口清单
对应需求：FR-05、NFR-R4（健康检查）、NFR-O3（指标暴露）

**分层约定**：本模块只负责"装配"——把配置、感知后端、会话存储、
各管线对象组装成一个 app。具体路由逻辑在 ``routes.py``，两者分离
使得测试可以只注入假依赖而不启动真实模型。

**执行模型（重要）**：

单帧推理是 CPU/GPU 密集的同步计算，因此路由定义为 ``def``（非 ``async def``），
让 FastAPI 自动丢进线程池，避免**阻塞事件循环**。若写成 ``async def`` 却
内部同步调用 YOLO，整个服务的并发能力会退化到 1。
这是 FastAPI 最常见的性能陷阱之一，实测中极易踩到。
只有涉及 ``await``（如 WebSocket 收发）的路由才应是 ``async``。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..observability import get_logger, setup_logging
from ..perception import perception_from_settings
from ..settings import Settings, get_settings
from .session_store import InMemorySessionStore, SessionStore

log = get_logger("api.app")

API_VERSION = "v1"


class AppState:
    """挂在 ``app.state`` 上的依赖容器。

    用显式容器而非模块级全局变量：测试可构造独立的 AppState，
    互不干扰；同时避免"导入即加载 YOLO 权重"这类副作用。
    """

    def __init__(self, settings: Settings, *, sessions: SessionStore | None = None) -> None:
        self.settings = settings
        self.sessions: SessionStore = sessions or InMemorySessionStore()
        self._perception = None
        self._processor = None
        self._postshot = None
        self._subject_pipeline = None
        self._calibration_pipeline = None
        self._case_search = None

    # --- 惰性构建：避免导入 app 即触发权重加载（拖慢启动、妨碍测试）---
    @property
    def perception(self):
        if self._perception is None:
            self._perception = perception_from_settings(self.settings)
            log.info("感知后端就绪: %s", self._perception.name)
        return self._perception

    @property
    def processor(self):
        if self._processor is None:
            from ..pipeline import FrameProcessor

            self._processor = FrameProcessor(self.perception, self.settings)
        return self._processor

    @property
    def postshot(self):
        if self._postshot is None:
            from ..pipeline import build_postshot

            self._postshot = build_postshot(self.settings)
        return self._postshot

    @property
    def subject_pipeline(self):
        if self._subject_pipeline is None:
            from ..pipeline import SubjectPipeline

            self._subject_pipeline = SubjectPipeline(self.perception, self.settings)
        return self._subject_pipeline

    @property
    def calibration_pipeline(self):
        if self._calibration_pipeline is None:
            from ..pipeline import CalibrationPipeline

            self._calibration_pipeline = CalibrationPipeline(self.perception, self.settings)
        return self._calibration_pipeline

    @property
    def case_search(self):
        """案例检索服务（FR-09）。惰性：不触发感知后端加载，连 Milvus
        失败也延迟到首次请求才暴露（然后走 200 + degraded）。"""
        if self._case_search is None:
            from ..retrieval import CaseSearchService

            self._case_search = CaseSearchService(
                self.settings,
                processor_getter=lambda: self.processor,
            )
        return self._case_search


def _cors_origins(settings: Settings) -> list[str]:
    """开发环境放开本地来源；生产环境收紧（NFR-S2）。

    本地 Demo 前端可能是 ``file://``（origin 为 ``null``）或
    ``http://localhost:5173``（Vite 默认端口）。硬编码单一来源会让
    用户配一次改一次，体验很差；但生产环境不能放开 ``*``。
    """
    if settings.app.env in ("dev", "test"):
        return [
            "http://localhost",
            "http://localhost:5173",
            "http://127.0.0.1",
            "http://127.0.0.1:5173",
            "null",  # file:// 打开时的 Origin
        ]
    return []


def create_app(settings: Settings | None = None, *, state: AppState | None = None) -> FastAPI:
    """构建 FastAPI 应用。

    Args:
        settings: 全局配置；None 时读单例。
        state: 可注入的依赖容器（测试用）。

    Returns:
        配置完成的 ``FastAPI`` 实例。
    """
    cfg = settings or get_settings()
    setup_logging(cfg.app.log_level)

    app_state = state or AppState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("API 启动: %s", cfg.summary())
        # **预热（重要性能修正）**：YOLO 首次推理含权重反序列化 + CUDA
        # kernel 编译，实测冷启动可达 6~7 秒。若不预热，Web 轨的**第一帧**
        # 就是这个数字，用户体验与延迟指标都会被一次性污染。
        # 预热放在启动期（而非首帧），把成本从"用户等待"挪到"服务就绪"。
        # 预热失败不影响服务可用——降级为"首帧慢"，绝不因此拒绝启动。
        try:
            app_state.perception.warmup()
            log.info("感知后端预热完成: %s", app_state.perception.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("感知预热失败（不影响可用性，首帧会较慢）: %s", exc)
        yield
        # 优雅关闭：清理过期会话，记录统计
        swept = app_state.sessions.sweep()
        log.info("API 关闭: 清理会话 %d 个", swept)

    app = FastAPI(
        title="AI 实时构图指导 Agent",
        version="0.1.0",
        description=(
            "实时构图指导：看画面 → 判构图 → 给建议 → 后处理。\n\n"
            "**降级约定**：任何一层失败都返回可用响应（``degraded=true``），"
            "绝不返回 5xx。见《数据模型与接口.md》§3.4。"
        ),
        lifespan=lifespan,
    )

    origins = _cors_origins(cfg)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.deps = app_state

    from .ops import build_ops_router
    from .routes import build_router, build_ws_router

    app.include_router(build_router(app_state))
    app.include_router(build_ws_router(app_state))
    app.include_router(build_ops_router(app_state))

    _mount_web(app, cfg)
    _install_error_handlers(app)
    return app


def _mount_web(app: FastAPI, settings: Settings) -> None:
    """把 ``web/`` 静态演示页挂到 ``/``（同源，免 CORS 配置）。

    **为什么挂在 API 服务里**：Web 轨演示页天然需要调用同一份 API。
    若分开部署，就得处理 CORS、端口、跨域凭证；挂在一起则 ``serve``
    一条命令同时提供页面与接口，演示与自测成本最低。

    目录不存在时**静默跳过**（不报错）——本项目允许只交付 API 而不带
    前端，强制要求目录存在会让 ``serve`` 在最小部署下直接失败。
    """
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    web_dir = Path(settings.project_root) / "web"
    if not (web_dir / "index.html").exists():
        log.debug("未发现 web/index.html，跳过前端挂载")
        return
    # html=True 使 ``/`` 返回 index.html
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    log.info("已挂载 Web 演示页: %s", web_dir)


def _install_error_handlers(app: FastAPI) -> None:
    """把 FastAPI 默认的 ``{"detail": ...}`` 统一成项目的错误契约。

    为什么必须做：`HTTPException` 的默认响应体是 ``{"detail": {...}}``，
    而我们约定的是 ``{"error": {"code", "message", "detail"}}``。若不统一，
    同一个 API 会同时存在两种错误结构，客户端无法用一套逻辑解析。
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(_request, exc: StarletteHTTPException):
        d = exc.detail
        if isinstance(d, dict) and "code" in d:
            body = {"error": d}
        else:
            body = {"error": {"code": "bad_request", "message": str(d), "detail": None}}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(_request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "bad_request",
                    "message": "请求参数校验失败",
                    "detail": exc.errors(),
                }
            },
        )


# 供 ``uvicorn aicg.api.app:app`` 直接使用
def _lazy_app() -> FastAPI:
    return create_app()


__all__ = ["API_VERSION", "AppState", "create_app"]
