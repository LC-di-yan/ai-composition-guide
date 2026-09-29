"""API 层测试（API-01 ~ API-10）。

对应文档：《数据模型与接口.md》§3
对应需求：FR-01~FR-09、NFR-R1（降级不抛）、NFR-R4

**测试策略**：用 ``redis``-free 的内存会话存储 + 规则感知后端，
全程不加载 YOLO、不联网，保证 CI 可复现。API 层的价值在于
"契约是否稳定、降级是否生效、错误是否统一"，而非算法正确性
（后者由各层单测覆盖）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from aicg.api import create_app
from aicg.api.session_store import InMemorySessionStore
from aicg.api.app import AppState
from aicg.settings import load_settings

FIXTURE = "tests/fixtures/synthetic_portrait.jpg"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """规则后端 + 内存会话的测试客户端（module 级复用，省去重复建 app）。"""
    cfg = load_settings(overrides={"perception.backend": "rule"})
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session_id(client: TestClient) -> str:
    r = client.post("/v1/session")
    assert r.status_code == 200
    return r.json()["session_id"]


# ======================================================================
class TestOps:
    """API-09 / API-10 运维端点。"""

    def test_healthz_is_fast_and_minimal(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readyz_reports_backend(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["perception_backend"] == "rule"

    def test_metrics_is_prometheus_text(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        text = r.text
        assert "aicg_sessions_active" in text
        assert "aicg_perception_ready" in text
        # Prometheus 文本格式要求每行 HELP/TYPE/值 结构
        assert "# HELP" in text and "# TYPE" in text

    def test_healthz_does_not_build_perception(self):
        """liveness 探针不得触发权重加载（否则重启风暴）。

        构造一个"感知构建必炸"的 AppState，若 /healthz 仍返回 200，
        说明它确实没碰感知层。
        """
        cfg = load_settings(overrides={"perception.backend": "rule"})
        state = AppState(cfg)
        state._perception = None

        app = create_app(cfg, state=state)
        with TestClient(app) as c:
            # 记录是否访问过 perception
            accessed = {"n": 0}

            class Boom:
                @property
                def name(self):
                    accessed["n"] += 1
                    raise RuntimeError("不应在 liveness 中被调用")

            state._perception = Boom()
            r = c.get("/healthz")
            assert r.status_code == 200
            assert accessed["n"] == 0


# ======================================================================
class TestSession:
    """API-01 / API-02 会话生命周期。"""

    def test_create_session(self, client):
        r = client.post("/v1/session")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"].startswith("s-")
        assert body["frame_count"] == 0
        assert body["language_calls"] == 0

    def test_sessions_are_unique(self, client):
        a = client.post("/v1/session").json()["session_id"]
        b = client.post("/v1/session").json()["session_id"]
        assert a != b

    def test_delete_session(self, client):
        sid = client.post("/v1/session").json()["session_id"]
        assert client.delete(f"/v1/session/{sid}").status_code == 200
        # 删除后不再可用
        assert client.delete(f"/v1/session/{sid}").status_code == 404

    def test_delete_unknown_session(self, client):
        r = client.delete("/v1/session/nope")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "session_not_found"


class TestSessionStoreUnit:
    """会话存储的 TTL / 容量行为（不经过 HTTP）。"""

    def test_ttl_expiry(self):
        fake = {"t": 0}
        store = InMemorySessionStore(ttl_s=1.0, clock_ms=lambda: fake["t"])
        s = store.create()
        assert store.get(s.session_id) is not None

        fake["t"] = 2000  # 超过 1s TTL
        assert store.get(s.session_id) is None

    def test_touch_extends_life(self):
        fake = {"t": 0}
        store = InMemorySessionStore(ttl_s=1.0, clock_ms=lambda: fake["t"])
        s = store.create()
        fake["t"] = 800
        assert store.get(s.session_id) is not None  # 访问续命
        fake["t"] = 1500  # 距上次访问 700ms < 1000ms
        assert store.get(s.session_id) is not None

    def test_capacity_evicts_lru(self):
        fake = {"t": 0}
        store = InMemorySessionStore(ttl_s=9999, max_sessions=3, clock_ms=lambda: fake["t"])
        ids = []
        for i in range(4):
            fake["t"] = i * 10
            ids.append(store.create().session_id)
        assert store.count() == 3
        # 最早创建的应被淘汰
        assert store.get(ids[0]) is None
        assert store.get(ids[3]) is not None

    def test_sweep_returns_count(self):
        fake = {"t": 0}
        store = InMemorySessionStore(ttl_s=1.0, clock_ms=lambda: fake["t"])
        store.create()
        store.create()
        fake["t"] = 5000
        assert store.sweep() == 2
        assert store.count() == 0


# ======================================================================
class TestCalibrate:
    """API-03 距离校准。"""

    def test_calibrate_returns_range_and_advice(self, client):
        r = client.post("/v1/calibrate", json={"image_ref": FIXTURE})
        assert r.status_code == 200
        b = r.json()
        assert b["min_distance_m"] < b["max_distance_m"]
        assert b["advice"]
        assert b["method"]
        assert "degraded" in b

    def test_calibrate_writes_to_session(self, client):
        sid = client.post("/v1/session").json()["session_id"]
        r = client.post("/v1/calibrate", json={"session_id": sid, "image_ref": FIXTURE})
        assert r.status_code == 200
        # 校准值应回填到会话上下文（后续帧可用）
        assert r.json()["current_distance_m"] is not None

    def test_calibrate_bad_image(self, client):
        r = client.post("/v1/calibrate", json={"image_ref": "no/such/file.jpg"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_image"


# ======================================================================
class TestSubject:
    """API-04 主体确认与人工指定。"""

    def test_auto_detect(self, client, session_id):
        r = client.post(
            "/v1/subject", json={"session_id": session_id, "image_ref": FIXTURE}
        )
        assert r.status_code == 200
        b = r.json()
        assert isinstance(b["subjects"], list)
        assert b["backend"] == "rule"

    def test_manual_override(self, client, session_id):
        """人工指定主体应优先于自动检测（FR-02 兜底路径）。"""
        bbox = [0.4, 0.2, 0.6, 0.95]
        r = client.post(
            "/v1/subject",
            json={"session_id": session_id, "image_ref": FIXTURE, "bbox": bbox},
        )
        assert r.status_code == 200
        b = r.json()
        assert b["primary_subject_id"] is not None
        # 人工指定后不应再要求用户输入
        assert b["needs_user_input"] is False
        primary = next(s for s in b["subjects"] if s["subject_id"] == b["primary_subject_id"])
        assert primary["source"] == "manual"

    def test_requires_valid_session(self, client):
        r = client.post("/v1/subject", json={"session_id": "nope", "image_ref": FIXTURE})
        assert r.status_code == 404


# ======================================================================
class TestFrame:
    """API-05 单帧引导推理（核心接口）。"""

    def test_frame_contract_shape(self, client, session_id):
        """响应必须包含契约约定的全部顶层字段。"""
        r = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        )
        assert r.status_code == 200
        b = r.json()
        for key in (
            "schema_version",
            "frame_id",
            "timestamp_ms",
            "frame_size",
            "command",
            "composition",
            "subject",
            "latency",
            "degraded",
            "backend",
        ):
            assert key in b, f"响应缺少契约字段: {key}"

    def test_command_shape(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        ).json()
        cmd = b["command"]
        for key in (
            "action",
            "direction",
            "magnitude_text",
            "magnitude_raw",
            "urgency",
            "is_hold",
            "confidence",
            "can_skip",
            "is_changed",
            "raw_action",
            "suppressed_by",
        ):
            assert key in cmd, f"指令缺少字段: {key}"
        # FR-12：任何建议都必须可拒绝
        assert cmd["can_skip"] is True
        # hold 时文案必须是"保持"，不得自相矛盾（曾修过的真实 bug）
        if cmd["is_hold"]:
            assert cmd["magnitude_text"] == "保持"
            assert cmd["magnitude_raw"] == 0.0

    def test_direction_is_known_enum(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        ).json()
        assert b["command"]["direction"] in (
            "forward",
            "backward",
            "left",
            "right",
            "up",
            "down",
            "hold",
        )

    def test_latency_breakdown_sums(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        ).json()
        lat = b["latency"]
        parts = lat["capture_ms"] + lat["perception_ms"] + lat["decision_ms"] + lat["stabilization_ms"]
        assert lat["total_ms"] == pytest.approx(parts, abs=0.5)

    def test_subject_override_in_frame(self, client, session_id):
        """请求内携带 subject_override 应生效，且不污染会话（用完即弃）。"""
        r = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
                "subject_override": [0.4, 0.2, 0.6, 0.95],
            },
        )
        assert r.status_code == 200
        assert r.json()["subject"] is not None
        # 下一帧不带覆盖时，应回到自动检测（覆盖不得粘住）
        b2 = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 2,
                "timestamp_ms": 100,
                "image_ref": FIXTURE,
            },
        ).json()
        assert b2["subject"]["source"] != "manual"

    def test_unknown_session_404(self, client):
        r = client.post(
            "/v1/frame",
            json={
                "session_id": "nope",
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        )
        assert r.status_code == 404

    def test_empty_image_ref_400(self, client, session_id):
        r = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": "",
            },
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_image"

    def test_missing_field_422_with_unified_shape(self, client):
        """参数缺失应返回 422，且错误体符合统一契约。"""
        r = client.post("/v1/frame", json={"frame_id": 1})
        assert r.status_code == 422
        assert "error" in r.json()
        assert r.json()["error"]["code"] == "bad_request"

    def test_all_errors_share_same_shape(self, client, session_id):
        """所有错误响应必须是同一结构（客户端才能用一套逻辑解析）。"""
        cases = [
            client.post(
                "/v1/frame",
                json={
                    "session_id": "nope",
                    "frame_id": 1,
                    "timestamp_ms": 0,
                    "image_ref": FIXTURE,
                },
            ),
            client.post(
                "/v1/frame",
                json={
                    "session_id": session_id,
                    "frame_id": 1,
                    "timestamp_ms": 0,
                    "image_ref": "",
                },
            ),
            client.post("/v1/frame", json={"frame_id": 1}),
            client.delete("/v1/session/nope"),
        ]
        for r in cases:
            assert r.status_code >= 400
            body = r.json()
            assert set(body.keys()) == {"error"}
            assert {"code", "message", "detail"} <= set(body["error"].keys())

    def test_persist_visual_returns_png_data_uri(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
                "persist_visual": True,
            },
        ).json()
        assert "visual_png_base64" in b
        assert b["visual_png_base64"].startswith("data:image/png;base64,")

    def test_persist_visual_absent_by_default(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        ).json()
        assert "visual_png_base64" not in b


# ======================================================================
class TestLanguage:
    """语言层触发与成本控制（NFR-E3）。"""

    def test_language_off_by_default(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        ).json()
        assert "narration" not in b

    def test_language_on_returns_narration(self, client, session_id):
        b = client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
                "enable_language": True,
            },
        ).json()
        assert "narration" in b
        n = b["narration"]
        assert n["text"]
        assert isinstance(n["is_fallback"], bool)
        assert "language_ms" in n

    def test_language_call_limit_enforced(self):
        """达到会话调用上限后应明确跳过，而不是静默继续烧钱。"""
        cfg = load_settings(
            overrides={"perception.backend": "rule", "language.max_calls_per_session": 2}
        )
        app = create_app(cfg)
        with TestClient(app) as c:
            sid = c.post("/v1/session").json()["session_id"]
            payload = {
                "session_id": sid,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
                "enable_language": True,
            }
            r1 = c.post("/v1/frame", json=payload).json()
            r2 = c.post("/v1/frame", json=payload).json()
            r3 = c.post("/v1/frame", json=payload).json()
            assert "narration" in r1
            assert "narration" in r2
            assert r3.get("language_skipped") == "session_call_limit_reached"


# ======================================================================
class TestShotReport:
    """API-07 拍后解说 + 滤镜推荐。"""

    def test_report_shape(self, client, session_id):
        r = client.post(
            "/v1/shot/report",
            json={"session_id": session_id, "image_ref": FIXTURE},
        )
        assert r.status_code == 200
        b = r.json()
        assert b["narration"]
        assert b["filter"]["name"]
        assert "is_fallback" in b
        assert b["prompt_version"]

    def test_report_reuses_snapshot(self, client, session_id):
        """传入 snapshot_frame_id 时应复用已有快照，保证结论一致。"""
        client.post(
            "/v1/frame",
            json={
                "session_id": session_id,
                "frame_id": 1,
                "timestamp_ms": 0,
                "image_ref": FIXTURE,
            },
        )
        r = client.post(
            "/v1/shot/report",
            json={"session_id": session_id, "image_ref": FIXTURE, "snapshot_frame_id": 1},
        )
        assert r.status_code == 200
        assert r.json()["shot_id"]


# ======================================================================
class TestCaseSearch:
    """API-08 案例检索（Milvus 未接入）。"""

    def test_returns_501_not_empty_list(self, client):
        """必须是 501，不能是空列表——空列表会掩盖"功能未实现"。"""
        r = client.post("/v1/cases/search", json={"query_text": "三分法", "top_k": 3})
        assert r.status_code == 501
        b = r.json()["error"]
        assert b["code"] == "not_implemented"
        assert b["detail"]["planned_milestone"] == "M5"


# ======================================================================
class TestWebSocket:
    """API-06 流式引导。"""

    def test_ping_pong(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong"
            assert "t_ms" in msg

    def test_init_creates_session(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "init"})
            msg = ws.receive_json()
            assert msg["type"] == "ready"
            assert msg["session_id"].startswith("s-")

    def test_frame_returns_result(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "init"})
            ws.receive_json()
            ws.send_json(
                {
                    "type": "frame",
                    "frame_id": 1,
                    "timestamp_ms": 0,
                    "image_ref": FIXTURE,
                }
            )
            msg = ws.receive_json()
            assert msg["type"] == "result"
            assert msg["frame_id"] == 1
            assert "command" in msg["data"]

    def test_auto_creates_session_on_first_frame(self, client):
        """不显式 init 也应能用（降低接入门槛）。"""
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json(
                {
                    "type": "frame",
                    "frame_id": 1,
                    "timestamp_ms": 0,
                    "image_ref": FIXTURE,
                }
            )
            first = ws.receive_json()
            assert first["type"] == "ready"
            second = ws.receive_json()
            assert second["type"] == "result"

    def test_non_json_does_not_drop_connection(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_text("this is not json")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_request"
            # 连接仍然可用 —— 这是关键（单帧错误不该踢掉用户）
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_unknown_type_does_not_drop_connection(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "totally_unknown"})
            err = ws.receive_json()
            assert err["type"] == "error"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_bad_image_does_not_drop_connection(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "init"})
            ws.receive_json()
            ws.send_json(
                {"type": "frame", "frame_id": 1, "timestamp_ms": 0, "image_ref": ""}
            )
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "invalid_image"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_sequence_of_frames(self, client):
        """连续推帧应逐帧返回，且 frame_id 对应。"""
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "init"})
            ws.receive_json()
            for i in range(3):
                ws.send_json(
                    {
                        "type": "frame",
                        "frame_id": i,
                        "timestamp_ms": i * 100,
                        "image_ref": FIXTURE,
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "result"
                assert msg["frame_id"] == i


# ======================================================================
class TestImageRef:
    """图像引用解码（含安全边界）。"""

    def test_path_mode_dev(self):
        from aicg.api.image_ref import decode_image_ref

        img = decode_image_ref(FIXTURE, allow_path=True)
        assert img is not None and img.ndim == 3

    def test_path_mode_prod_rejected(self):
        """生产环境不接受路径，避免任意文件读取漏洞（NFR-S2）。"""
        from aicg.api.image_ref import ImageDecodeError, decode_image_ref

        with pytest.raises(ImageDecodeError):
            decode_image_ref(FIXTURE, allow_path=False)

    def test_base64_roundtrip(self):
        import base64

        import cv2

        from aicg.api.image_ref import decode_image_ref

        raw = cv2.imread(FIXTURE)
        ok, buf = cv2.imencode(".jpg", raw)
        assert ok
        b64 = base64.b64encode(buf.tobytes()).decode()
        img = decode_image_ref(b64, allow_path=False)
        assert img.shape[:2] == raw.shape[:2]

    def test_data_uri_prefix(self):
        import base64

        import cv2

        from aicg.api.image_ref import decode_image_ref

        raw = cv2.imread(FIXTURE)
        ok, buf = cv2.imencode(".png", raw)
        uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
        img = decode_image_ref(uri, allow_path=False)
        assert img.ndim == 3

    def test_empty_rejected(self):
        from aicg.api.image_ref import ImageDecodeError, decode_image_ref

        with pytest.raises(ImageDecodeError):
            decode_image_ref("")

    def test_garbage_base64_rejected(self):
        from aicg.api.image_ref import ImageDecodeError, decode_image_ref

        junk = "A" * 200  # 合法 base64 字符，但不是图像
        with pytest.raises(ImageDecodeError):
            decode_image_ref(junk, allow_path=False)

    def test_windows_path_not_mistaken_for_base64(self):
        """Windows 路径含 ``\\`` 与 ``:``，不应被误判成 base64。"""
        from aicg.api.image_ref import looks_like_base64

        assert looks_like_base64(r"C:\Users\x\photo.jpg") is False

    def test_base64_containing_slash_still_detected(self):
        """base64 字母表含 ``/``，绝不能因为出现 ``/`` 就判为路径。

        这是本项目修过的一个真实 bug：第一版用"含 ``/`` 即路径"做判别，
        导致所有真实图像的 base64 都被当成路径 → base64 输入全线失败。
        """
        import base64

        import cv2

        from aicg.api.image_ref import decode_image_ref, looks_like_base64

        raw = cv2.imread(FIXTURE)
        ok, buf = cv2.imencode(".jpg", raw)
        b64 = base64.b64encode(buf.tobytes()).decode()
        assert "/" in b64, "本测试前提：该 base64 确实含 /"
        assert looks_like_base64(b64) is True
        # 且必须能真正解码成功
        img = decode_image_ref(b64, allow_path=False)
        assert img.shape[:2] == raw.shape[:2]

    def test_relative_path_not_mistaken_for_base64(self):
        from aicg.api.image_ref import looks_like_base64

        assert looks_like_base64("./tests/fixtures/x.png") is False
        assert looks_like_base64("../a/b.jpg") is False
        assert looks_like_base64("tests/fixtures/y.jpeg") is False

    def test_absolute_path_detected_by_extension(self):
        """绝对路径靠扩展名判定，而非"以 / 开头"（后者会误伤 base64）。"""
        from aicg.api.image_ref import looks_like_base64

        assert looks_like_base64("/abs/path/img.jpeg") is False

    def test_short_string_not_base64(self):
        from aicg.api.image_ref import looks_like_base64

        assert looks_like_base64("abc") is False

    def test_encode_png_data_uri(self):
        import cv2

        from aicg.api.image_ref import encode_image_png_base64

        uri = encode_image_png_base64(cv2.imread(FIXTURE))
        assert uri.startswith("data:image/png;base64,")


# ======================================================================
class TestOpenAPISchema:
    """OpenAPI 文档应包含全部约定端点（契约可被前端消费）。"""

    def test_all_endpoints_documented(self, client):
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        expected = [
            "/v1/session",
            "/v1/session/{session_id}",
            "/v1/calibrate",
            "/v1/subject",
            "/v1/frame",
            "/v1/shot/report",
            "/v1/cases/search",
            "/healthz",
            "/readyz",
            "/metrics",
        ]
        for p in expected:
            assert p in paths, f"OpenAPI 缺少端点: {p}"

    def test_openapi_is_valid_json(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        json.loads(r.text)
