"""API 路由（API-01 ~ API-10）。

对应文档：《数据模型与接口.md》§3.1 接口清单
对应需求：FR-01~FR-09、NFR-R4、NFR-O3

**统一错误约定（《数据模型与接口.md》§3.3）**：

原文档把"错误码体系"标为待确认，此处定型为下表。核心原则：
**参数类错误返回 4xx，运行类问题返回 200 + ``degraded``**——
后者是关键设计，实时引导"崩掉"比"给个差建议"糟糕得多（NFR-R1）。

============ ====== ==========================================
错误码         状态码  场景
============ ====== ==========================================
``bad_request`` 400    参数缺失/格式非法（如 image_ref 空）
``invalid_image`` 400  图像解码失败
``session_not_found`` 404 会话不存在或已过期
``not_implemented`` 501 功能未实现（通用兜底；案例检索 M6-2 起已接入，
Milvus 不可用时返回 200 + ``degraded`` 而非 501）
``internal_error`` 500 未预期异常（兜底，正常情况下不应出现）
============ ====== ==========================================

**执行模型**：见 ``app.py`` 头部说明——密集计算路由用 ``def`` 而非 ``async def``。
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..observability import get_logger
from ..schemas import (
    CalibrateRequest,
    CaseSearchRequest,
    FrameRequest,
    SessionInfo,
    ShotReportRequest,
    SubjectOverrideRequest,
)
from .app import API_VERSION, AppState
from .image_ref import ImageDecodeError, decode_image_ref, encode_image_png_base64
from .mappers import snapshot_to_response

log = get_logger("api.routes")


# ----------------------------------------------------------------------
# 错误工具
# ----------------------------------------------------------------------
def _error(code: str, message: str, detail: Any = None) -> JSONResponse:
    """构造统一错误响应体。"""
    return JSONResponse(
        status_code=_STATUS_FOR_CODE.get(code, 400),
        content={"error": {"code": code, "message": message, "detail": detail}},
    )


_STATUS_FOR_CODE = {
    "bad_request": 400,
    "invalid_image": 400,
    "session_not_found": 404,
    "not_implemented": 501,
    "internal_error": 500,
}


def _decode_or_400(image_ref: str, state: AppState):
    """解码图像，失败则抛 400。"""
    allow_path = state.settings.app.env in ("dev", "test")
    try:
        return decode_image_ref(image_ref, allow_path=allow_path)
    except ImageDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_image", "message": str(e), "detail": None},
        ) from e


def _require_session(state: AppState, session_id: str):
    sess = state.sessions.get(session_id)
    if sess is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "session_not_found",
                "message": f"会话不存在或已过期: {session_id}",
                "detail": None,
            },
        )
    return sess


# ----------------------------------------------------------------------
def build_router(state: AppState) -> APIRouter:
    """构建路由。闭包捕获 ``state``，避免全局单例。"""
    r = APIRouter(prefix=f"/{API_VERSION}", tags=["guiding"])

    # ================= API-01 创建会话 =================
    @r.post("/session", response_model=SessionInfo, summary="创建拍摄会话")
    def create_session() -> SessionInfo:
        sess = state.sessions.create()
        return SessionInfo(
            session_id=sess.session_id,
            created_at_ms=sess.created_at_ms,
            frame_count=0,
            shot_count=0,
            language_calls=0,
        )

    # ================= API-02 结束会话 =================
    @r.delete("/session/{session_id}", summary="结束会话")
    def end_session(session_id: str) -> dict[str, Any]:
        ok = state.sessions.delete(session_id)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "session_not_found",
                    "message": f"会话不存在: {session_id}",
                    "detail": None,
                },
            )
        return {"ok": True, "session_id": session_id}

    # ================= API-03 距离校准 =================
    @r.post("/calibrate", summary="距离校准（FR-01）")
    def calibrate(req: CalibrateRequest) -> dict[str, Any]:
        img = _decode_or_400(req.image_ref, state)
        outcome = state.calibration_pipeline.calibrate(img, frame_id=0)
        cal = outcome.result

        # 校准结果回填到会话上下文，后续帧据此附带绝对米数
        if req.session_id:
            sess = state.sessions.get(req.session_id)
            if sess is not None:
                sess.ctx.calibration_distance_m = cal.current_distance_m

        return {
            "method": cal.estimation_method.value,
            "current_distance_m": cal.current_distance_m,
            "min_distance_m": cal.min_distance_m,
            "max_distance_m": cal.max_distance_m,
            "position_ratio": cal.position_ratio,
            "is_in_range": cal.is_in_range,
            "advice": cal.advice_text,
            "error_margin_m": cal.error_margin,
            "subject_bbox": list(outcome.subject_bbox) if outcome.subject_bbox else None,
            "subject_label": outcome.subject_label,
            "backend": outcome.backend,
            "degraded": outcome.degraded,
        }

    # ================= API-04 主体确认 =================
    @r.post("/subject", summary="主体确认 / 人工指定（FR-02）")
    def subject(req: SubjectOverrideRequest) -> dict[str, Any]:
        _require_session(state, req.session_id)
        img = _decode_or_400(req.image_ref, state)

        if req.bbox is not None:
            outcome = state.subject_pipeline.override(img, tuple(req.bbox), frame_id=0)
        else:
            outcome = state.subject_pipeline.detect(img, frame_id=0)

        return {
            "subjects": [
                {
                    "subject_id": s.subject_id,
                    "label": s.label,
                    "bbox": list(s.bbox),
                    "confidence": s.confidence,
                    "is_primary": s.is_primary,
                    "source": s.source.value,
                }
                for s in outcome.subjects
            ],
            "primary_subject_id": outcome.primary.subject_id if outcome.primary else None,
            "needs_user_input": outcome.needs_user_input,
            "backend": outcome.backend,
            "degraded": outcome.degraded,
        }

    # ================= API-05 单帧引导推理（核心）=================
    @r.post("/frame", summary="单帧引导推理（核心）")
    def frame(req: FrameRequest) -> dict[str, Any]:
        sess = _require_session(state, req.session_id)
        img = _decode_or_400(req.image_ref, state)

        from ..camera.base import Frame

        f = Frame(
            image=img,
            frame_id=req.frame_id,
            timestamp_ms=req.timestamp_ms,
        )
        # 人工主体覆盖优先（FR-02 兜底）
        prev_override = sess.ctx.subject_override_bbox
        if req.subject_override is not None:
            sess.ctx.subject_override_bbox = tuple(req.subject_override)

        try:
            snapshot = state.processor.process(f, sess.ctx)
        finally:
            sess.ctx.subject_override_bbox = prev_override

        sess.latency.add(snapshot.latency)

        resp = snapshot_to_response(snapshot)

        # 可选：附带该帧标注可视化（与录屏轨同一渲染器，保证两轨一致）
        if req.persist_visual:
            from ..viz import SnapshotRenderer

            renderer = SnapshotRenderer(scale=1.0)
            canvas = renderer.render(img, snapshot)
            resp["visual_png_base64"] = encode_image_png_base64(canvas)

        # 可选：触发语言层（成本敏感，默认关闭，NFR-E3）
        if req.enable_language:
            if sess.language_calls >= state.settings.language.max_calls_per_session:
                resp["language_skipped"] = "session_call_limit_reached"
            else:
                outcome = state.postshot.generate(snapshot, img, with_image=False)
                sess.language_calls += 1
                resp["narration"] = {
                    "text": outcome.report.composition_narration,
                    "is_fallback": outcome.report.is_fallback,
                    "model": outcome.report.vlm_model,
                    "prompt_version": outcome.report.prompt_version,
                    "language_ms": outcome.language_ms,
                }

        return resp

    # ================= API-07 拍后解说 =================
    @r.post("/shot/report", summary="拍后解说 + 滤镜推荐（FR-07/FR-08）")
    def shot_report(req: ShotReportRequest) -> dict[str, Any]:
        sess = _require_session(state, req.session_id)
        img = _decode_or_400(req.image_ref, state)

        # 允许复用已有快照（避免重复推理，且保证与引导时结论一致）
        snapshot = None
        if req.snapshot_frame_id is not None:
            for s in reversed(sess.ctx.snapshots):
                if s.frame_id == req.snapshot_frame_id:
                    snapshot = s
                    break
        if snapshot is None:
            from ..camera.base import Frame

            f = Frame(image=img, frame_id=0, timestamp_ms=int(time.time() * 1000))
            snapshot = state.processor.process(f, sess.ctx)

        if sess.language_calls >= state.settings.language.max_calls_per_session:
            return _error(
                "bad_request",
                f"语言层调用已达会话上限 {state.settings.language.max_calls_per_session}",
            )

        outcome = state.postshot.generate(snapshot, img, with_image=False, speak=False)
        sess.language_calls += 1
        sess.shot_count += 1
        report = outcome.report

        return {
            "shot_id": report.shot_id,
            "narration": report.composition_narration,
            "filter": {
                "name": report.filter_name,
                "reason": report.filter_reason,
            },
            "is_fallback": report.is_fallback,
            "vlm_model": report.vlm_model,
            "prompt_version": report.prompt_version,
            "token_usage": report.token_usage.model_dump() if report.token_usage else None,
            "language_ms": outcome.language_ms,
        }

    # ================= API-08 案例检索（FR-09，Milvus）=================
    @r.post("/cases/search", summary="案例检索（FR-09，Milvus 向量检索）")
    def case_search(req: CaseSearchRequest) -> dict[str, Any]:
        from ..retrieval import VALID_PATTERNS

        # 参数类错误 400（唯一允许 4xx 的路径）；其余一切失败均由
        # service 转成 200 + degraded（NFR-R1）。
        if not req.image_ref and not (req.query_text or "").strip():
            return _error(
                "bad_request",
                "image_ref 与 query_text 至少提供一个（以图搜图 / 以文搜图）",
            )
        if req.pattern is not None and req.pattern not in VALID_PATTERNS:
            return _error(
                "bad_request",
                f"pattern 取值非法: {req.pattern}，合法值: {sorted(VALID_PATTERNS)}",
            )

        img = None
        if req.image_ref:
            img = _decode_or_400(req.image_ref, state)

        outcome = state.case_search.search(
            image=img,
            query_text=req.query_text,
            top_k=req.top_k,
            pattern=req.pattern,
        )
        return outcome.to_response()

    return r


# ----------------------------------------------------------------------
def build_ws_router(state: AppState) -> APIRouter:
    """WebSocket 流式引导（API-06）。

    与 ``POST /v1/frame`` 的关系：**同一套处理逻辑，不同的传输形态**。
    REST 是"我问一帧你答一帧"；WS 是"连上后持续推帧、持续回结果"，
    省掉每帧 HTTP 往返开销，适合真的在跑实时引导的场景。

    消息协议（请求）::

        {"type": "frame", "frame_id": 1, "timestamp_ms": 0, "image_ref": "..."}
        {"type": "ping"}

    消息协议（响应）::

        {"type": "result", "frame_id": 1, "data": {...}}   # 同 REST 帧响应
        {"type": "pong", "t_ms": ...}
        {"type": "error", "code": "...", "message": "..."}

    **关键设计：单帧错误不断连**。既然 REST 约定"运行类问题返回 degraded
    而非 5xx"，WS 也应只在协议级错误时才断，否则用户一帧传错图就掉线。
    """
    ws_router = APIRouter()

    @ws_router.websocket(f"/{API_VERSION}/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        session_id: str | None = None
        conn_start = time.perf_counter()
        frames_ok = 0
        frames_err = 0

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    frames_err += 1
                    await ws.send_json(
                        {"type": "error", "code": "bad_request", "message": "消息不是合法 JSON"}
                    )
                    continue

                mtype = msg.get("type")

                if mtype == "ping":
                    await ws.send_json({"type": "pong", "t_ms": int(time.time() * 1000)})
                    continue

                if mtype == "init":
                    sess = state.sessions.create()
                    session_id = sess.session_id
                    await ws.send_json({"type": "ready", "session_id": session_id})
                    continue

                if mtype != "frame":
                    frames_err += 1
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": "bad_request",
                            "message": f"未知消息类型: {mtype}",
                        }
                    )
                    continue

                if session_id is None:
                    # 自动建会话，降低客户端接入门槛
                    sess = state.sessions.create()
                    session_id = sess.session_id
                    await ws.send_json({"type": "ready", "session_id": session_id})
                else:
                    sess = state.sessions.get(session_id)
                    if sess is None:
                        await ws.send_json(
                            {
                                "type": "error",
                                "code": "session_not_found",
                                "message": "会话已过期，请重新 init",
                            }
                        )
                        session_id = None
                        continue

                # --- 处理一帧 ---
                try:
                    allow_path = state.settings.app.env in ("dev", "test")
                    img = decode_image_ref(
                        str(msg.get("image_ref", "")), allow_path=allow_path
                    )
                except ImageDecodeError as e:
                    frames_err += 1
                    await ws.send_json(
                        {"type": "error", "code": "invalid_image", "message": str(e)}
                    )
                    continue

                from ..camera.base import Frame

                f = Frame(
                    image=img,
                    frame_id=int(msg.get("frame_id", 0)),
                    timestamp_ms=int(msg.get("timestamp_ms", 0)),
                )
                snapshot = state.processor.process(f, sess.ctx)
                sess.latency.add(snapshot.latency)
                frames_ok += 1

                await ws.send_json(
                    {
                        "type": "result",
                        "frame_id": snapshot.frame_id,
                        "data": snapshot_to_response(snapshot),
                    }
                )

        except WebSocketDisconnect:
            log.info(
                "WebSocket 断开: session=%s 成功=%d 失败=%d 时长=%.1fs",
                session_id,
                frames_ok,
                frames_err,
                time.perf_counter() - conn_start,
            )
        except Exception as e:  # noqa: BLE001 - WS 异常不得让进程崩掉
            log.error("WebSocket 异常: %s", e)
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    return ws_router


__all__ = ["build_router", "build_ws_router"]
