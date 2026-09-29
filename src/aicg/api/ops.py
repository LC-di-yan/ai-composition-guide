"""运维端点：健康检查与指标暴露（API-09 / API-10）。

对应需求：NFR-R4（可恢复性）、NFR-O3（指标暴露）

**为什么健康检查要区分 liveness 与 readiness**：

- ``/healthz``（liveness）：进程还活着吗？只要没死就返回 200。
  若这里去探测模型权重，权重一时读不到会让编排器反复重启一个
  **其实健康**的进程，反而放大故障。
- ``/readyz``（readiness）：能接业务流量吗？这里才检查感知后端是否可用。
  未就绪时返回 503，让负载均衡把流量摘走，等就绪再放回。

把两者混为一谈（最常见做法）会导致"模型加载慢 → 被判定为死 → 重启风暴"。
本项目的感知后端是**惰性构建**的，若 liveness 里触发构建，每次探针
都会拉起 YOLO，既慢又浪费，因此这里严格分开。

``/metrics`` 输出 Prometheus 文本格式（无需引入 prometheus_client 依赖，
手写格式成本极低且可控）。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from ..observability import get_logger
from .app import API_VERSION, AppState
from .session_store import MAX_SESSIONS

log = get_logger("api.ops")


def build_ops_router(state: AppState) -> APIRouter:
    """构建 /healthz /readyz /metrics（不带 /v1 前缀）。"""
    r = APIRouter(tags=["ops"])

    @r.get("/healthz", summary="存活探针（liveness）")
    def healthz() -> dict[str, str]:
        # 不做任何重活：探针必须极快且无副作用
        return {"status": "ok"}

    @r.get("/readyz", summary="就绪探针（readiness）")
    def readyz():
        """检查业务依赖是否可用。未就绪返回 503。"""
        try:
            # 这里才触发感知后端构建（可能加载权重）
            backend = state.perception.name
            available = True
            detail = {"perception_backend": backend}
        except Exception as e:  # noqa: BLE001
            available = False
            detail = {"error": f"{type(e).__name__}: {e}"}

        body = {
            "status": "ready" if available else "not_ready",
            "env": state.settings.app.env,
            "sessions": state.sessions.count(),
            **detail,
        }
        if not available:
            return JSONResponse(status_code=503, content=body)
        return body

    @r.get("/metrics", summary="Prometheus 指标")
    def metrics() -> PlainTextResponse:
        lines: list[str] = []

        lines.append("# HELP aicg_sessions_active 当前活跃会话数")
        lines.append("# TYPE aicg_sessions_active gauge")
        lines.append(f"aicg_sessions_active {state.sessions.count()}")

        lines.append("# HELP aicg_sessions_max 会话容量上限")
        lines.append("# TYPE aicg_sessions_max gauge")
        lines.append(f"aicg_sessions_max {MAX_SESSIONS}")

        try:
            backend = state.perception.name
            lines.append("# HELP aicg_perception_ready 感知后端是否可用")
            lines.append("# TYPE aicg_perception_ready gauge")
            lines.append(f"aicg_perception_ready 1")
            lines.append("# HELP aicg_perception_info 感知后端信息")
            lines.append("# TYPE aicg_perception_info gauge")
            lines.append(f'aicg_perception_info{{backend="{backend}"}} 1')
        except Exception:  # noqa: BLE001
            lines.append("aicg_perception_ready 0")

        lines.append("# HELP aicg_schema_version 契约版本")
        lines.append("# TYPE aicg_schema_version gauge")
        lines.append("aicg_schema_version 1")

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return r


__all__ = ["build_ops_router"]
