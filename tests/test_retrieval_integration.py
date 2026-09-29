"""FR-09 真·Milvus 集成测试（**默认跳过**，有真实库才跑）。

为什么单独一个文件而不是塞进 test_retrieval.py：那里全是 mock/stub，
可在 CI 裸环境跑；这里要求**真正的 Milvus 服务端或 Milvus Lite 文件**，
语义不同、代价不同、失败含义也不同，混在一起会让"CI 绿了"这句话
失真。

启用方式（本机免 Docker 的用法，见《Milvus本地方案调研.md》§6）::

    # 1) 建库（Milvus Lite 本地文件）
    set AICG_RETRIEVAL_URI=<项目根>\\outputs\\index\\composition_cases.db
    python scripts/build_case_index.py

    # 2) 跑本文件的集成测试
    python -m pytest tests/test_retrieval_integration.py -v

**编码的两个真实缺陷**（都是先真跑才暴露出来的）:

1. 跨进程读取时 collection 处于 ``released`` → search 报 ``code=101``；
   ``MilvusStore`` 必须自己 load（``test_new_store_instance_can_read``）。
2. Windows 下若 Milvus 客户端**首次初始化**发生在帧处理链路之后，
   进程会在 milvus_lite 的 ``pa.RecordBatch`` 路径 SIGSEGV；预热把首次
   接触提到启动期（``test_api_search_uses_real_index`` 走 TestClient 的
   lifespan 预热，覆盖真实调用顺序）。
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4
import pytest

_URI = os.environ.get("AICG_RETRIEVAL_URI", "").strip()
pytestmark = pytest.mark.skipif(
    not _URI,
    reason="未设置 AICG_RETRIEVAL_URI：跳过真实 Milvus 集成测试",
)

pytest.importorskip("pymilvus", reason="pymilvus 未安装")

from aicg.retrieval import (  # noqa: E402
    CaseRecord,
    MilvusConfig,
    MilvusStore,
    StoreError,
)

# 集成测试专用集合，避免污染正式案例库
IT_COLLECTION = "cases_it"
ROOT = Path(__file__).resolve().parents[1]


def _it_store(name: str) -> MilvusStore:
    """每个用例一个**独立库文件**，天然隔离，无需 drop()。

    为什么不用 drop 来清理：Lite 在 Windows 上用 POSIX 语义做
    ``os.rename(tmp, manifest.json)``，目标已存在时抛 ``WinError 183``
    （实测：drop → close → flush → rename 失败）。用独立路径既避开这个
    裂缝，也让用例之间零耦合。
    """
    db_dir = ROOT / "outputs" / "index" / "_it"
    db_dir.mkdir(parents=True, exist_ok=True)
    cfg = MilvusConfig(uri=str(db_dir / f"{name}_{uuid4().hex[:8]}.db"), collection=IT_COLLECTION)
    st = MilvusStore(cfg)
    st.ensure_collection()
    return st


# ======================================================================
class TestRealMilvusStore:
    """对真实存储层的读写契约。"""

    def test_upsert_then_search_is_sorted_desc(self):
        """检索结果必须按相似度**降序**，且自反查询相似度为 1。"""
        st = _it_store("order")
        v_self = [0.10, 0.20, 0.30, 0.40, 0.50, 0.50, 0.60, 0.70]
        rows = [
            CaseRecord("far", [0.9] * 8, "assets/far.jpg", pattern="center"),
            CaseRecord("mid", [0.5] * 8, "assets/mid.jpg", pattern="center"),
            CaseRecord("near", v_self, "assets/near.jpg", pattern="center"),
        ]
        assert st.upsert(rows) == 3
        hits = st.search(v_self, top_k=3)
        assert [h["case_id"] for h in hits][0] == "near", f"最近邻不是自身: {hits}"
        sims = [h["similarity"] for h in hits]
        assert sims == sorted(sims, reverse=True), f"未按相似度降序: {sims}"
        assert sims[0] == pytest.approx(1.0, abs=1e-3), f"自反相似度应≈1，实际 {sims[0]}"

    def test_scalar_filter_is_respected(self):
        """pattern 标量过滤必须在真实库上生效。"""
        st = _it_store("filter")
        st.upsert(
            [
                CaseRecord("a", [0.1] * 8, "assets/a.jpg", pattern="center"),
                CaseRecord("b", [0.2] * 8, "assets/b.jpg", pattern="diagonal"),
            ]
        )
        hits = st.search([0.15] * 8, top_k=5, pattern="diagonal")
        assert [h["case_id"] for h in hits] == ["b"], hits

    def test_new_store_instance_can_read(self):
        """换新 store 实例（等价另一个进程）后仍能读——防 code=101 回归。

        背景：写入进程退出后 collection 处于 released，若没有 load 步骤，
        表现就是"库里有数据却永远搜不到"，且会被误判成 search_failed。
        """
        _w = _it_store("crossproc")
        st_uri = _w.cfg.uri
        _w.upsert([CaseRecord("x", [0.3] * 8, "assets/x.jpg", pattern="center")])
        # 新实例复用同一路径 = 等价另一个进程重新打开库
        fresh = MilvusStore(MilvusConfig(uri=st_uri, collection=IT_COLLECTION))
        try:
            assert fresh.count() == 1
            assert [h["case_id"] for h in fresh.search([0.3] * 8, top_k=1)] == ["x"]
        except StoreError as exc:  # 让失败信息带上原因枚举，便于定位
            pytest.fail(f"新实例读取失败（疑似 released 未 load）: [{exc.reason}] {exc}")

    def test_dimension_mismatch_is_rejected(self):
        """维度契约不一致必须显式报错，而不是静默写出错位向量。"""
        st = _it_store("dim")
        try:
            st.upsert([CaseRecord("bad", [0.1] * 4, "assets/bad.jpg", pattern="center")])
        except StoreError as exc:
            # 服务端可能直接拒绝，也可能被引擎截断——两种都必须落到这里或
            # 被服务端报错，唯一不可接受的是"静默成功"。
            assert exc.reason in {"upsert_failed", "search_failed"}
        else:
            pytest.fail("4 维向量竟写入了 8 维 collection，维度契约失效")


# ======================================================================
class TestApiWithRealIndex:
    """走完整 HTTP 栈：TestClient 触发 lifespan 预热 → 真实检索。"""

    @pytest.fixture(scope="class")
    def client(self):
        from aicg.api import create_app
        from aicg.settings import load_settings

        cfg = load_settings(overrides={"perception.backend": "rule"})
        from fastapi.testclient import TestClient

        app = create_app(cfg)
        with TestClient(app) as c:
            yield c

    def test_api_search_uses_real_index(self, client):
        """真实建库后，以图搜图应**不降级**且给出降序结果。"""
        img = ROOT / "assets" / "topic_photos" / "12_selfie_1280x853_cdn.jpg"
        if not img.exists():
            pytest.skip("缺少案例素材")
        r = client.post(
            "/v1/cases/search",
            json={"image_ref": str(img), "top_k": 3},
        )
        assert r.status_code == 200
        b = r.json()
        assert b["degraded"] is False, f"真实库应在却降级: {b['degrade_reason']}"
        assert b["total"] >= 1
        sims = [x["similarity"] for x in b["results"]]
        assert sims == sorted(sims, reverse=True), sims
        assert sims[0] > 0.9, f"自身素材应高度相似，实际 {sims}"
        assert (ROOT / b["results"][0]["image_ref"]).exists(), "image_ref 应可定位到素材"

    def test_text_query_still_degrades_honestly(self, client):
        """纯文本在真实库上依然诚实降级（无向量 → similarity 恒 0）。"""
        r = client.post("/v1/cases/search", json={"query_text": "居中构图", "top_k": 2})
        assert r.status_code == 200
        b = r.json()
        assert b["degraded"] is True
        assert b["degrade_reason"] == "text_embedding_unavailable"
        assert b["pattern_filter"] == "center"
        for row in b["results"]:
            assert row["similarity"] == 0.0, "标量过滤无排序语义，相似度必须诚实为 0"
