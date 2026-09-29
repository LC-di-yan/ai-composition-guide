"""检索层测试（FR-09）。

**测试策略**：
- 特征提取：构造合成 ``FrameSnapshot``，逐维核对契约（名称/次序/值域）；
- 降级链：monkeypatch 存储层抛 :class:`StoreError`，验证 reason 透传；
- 真实连接失败：本机未启动 Milvus 时的 ``connection_failed``（真实路径，
  非 mock——这是"降级承诺"在依赖缺失场景的直接证据）。

真实 Milvus 端到端（建库 + 检索 + 相似度排序）另有专门的
``tests/test_retrieval_integration.py``（需 ``AICG_RETRIEVAL_URI``，
默认跳过）——把"要真库"和"要 mock"的用例分开，是为了不让"CI 绿了"
这句话失真。下面的 :class:`TestEnsureLoaded` 用假客户端覆盖那些
**真跑才暴露、但必须常驻 CI** 的加载逻辑。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aicg.retrieval import (
    DIM,
    DIM_NAMES,
    CaseSearchService,
    MilvusConfig,
    MilvusStore,
    StoreError,
    features_from_snapshot,
)
from aicg.retrieval.search import pattern_from_text
from aicg.schemas.composition import ActionType, CompositionPattern, CompositionResult
from aicg.schemas.perception import PerceptionResult, Subject, SubjectSource
from aicg.schemas.snapshot import FrameSnapshot
from aicg.settings import load_settings
from aicg.stabilization import CommandDebouncer  # noqa: F401 — 保证导入链健康


# ----------------------------------------------------------------------
# 合成快照工厂
# ----------------------------------------------------------------------
def _make_snapshot(
    *,
    subject_bbox: tuple[float, float, float, float] | None = (0.3, 0.2, 0.5, 0.7),
    sub_scores: dict[str, float] | None = None,
) -> FrameSnapshot:
    subs = sub_scores if sub_scores is not None else {
        "thirds": 0.4,
        "balance": 0.6,
        "headroom": 0.5,
        "lead_room": 0.5,
        "subject_scale": 0.3,
        "subject_center": 0.7,
        "saliency_center": 0.5,
    }
    subjects = []
    if subject_bbox is not None:
        subjects = [
            Subject(
                subject_id="s0",
                label="person",
                bbox=subject_bbox,
                confidence=0.9,
                is_primary=True,
                source=SubjectSource.AUTO,
            )
        ]
    perception = PerceptionResult(
        frame_id=0,
        timestamp_ms=0,
        frame_size=(640, 480),
        subjects=subjects,
    )
    composition = CompositionResult(
        composition_score=60.0,
        sub_scores=subs,
        pattern=CompositionPattern.CENTER,
    )
    from aicg.schemas.composition import ActionCommand, StabilizedCommand

    cmd = ActionCommand(action=list(ActionType)[0], magnitude_text="")
    return FrameSnapshot(
        frame_id=0,
        timestamp_ms=0,
        frame_size=(640, 480),
        perception=perception,
        composition=composition,
        command=StabilizedCommand(command=cmd, raw_command=cmd, is_changed=False),
    )


# ----------------------------------------------------------------------
# 特征契约
# ----------------------------------------------------------------------
class TestFeatures:
    def test_dim_contract(self):
        assert DIM == 8
        assert len(DIM_NAMES) == 8
        assert len(set(DIM_NAMES)) == 8  # 名称不得重复

    def test_extraction_values_and_order(self):
        snap = _make_snapshot(subject_bbox=(0.3, 0.2, 0.5, 0.7))
        vec = features_from_snapshot(snap)
        assert vec is not None
        assert len(vec) == 8
        assert all(0.0 <= v <= 1.0 for v in vec)
        # 几何维独立核对：主体高度 = 0.7-0.2 = 0.5
        assert vec[6] == pytest.approx(0.5)
        # 中心 x = 0.4 → |0.4-0.5|*2 = 0.2
        assert vec[7] == pytest.approx(0.2)
        # 评分维从 sub_scores 直取
        assert vec[0] == pytest.approx(0.4)  # thirds

    def test_out_of_range_scores_clamped(self):
        snap = _make_snapshot(sub_scores={"thirds": 1.5, "balance": -0.2})
        vec = features_from_snapshot(snap)
        assert vec is not None
        assert vec[0] == 1.0  # 上界裁剪
        assert vec[2] == 0.0  # 下界裁剪

    def test_no_subject_returns_none(self):
        snap = _make_snapshot(subject_bbox=None)
        assert features_from_snapshot(snap) is None


# ----------------------------------------------------------------------
# query_text → pattern 标量过滤
# ----------------------------------------------------------------------
class TestPatternFromText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("想要三分法构图", "rule_of_thirds"),
            ("咖啡馆 居中 人像", "center"),
            ("对称构图", "symmetric"),
            ("对角线街拍", "diagonal"),
            ("前景框架感", "framing"),
        ],
    )
    def test_keyword_hits(self, text, expected):
        assert pattern_from_text(text) == expected

    def test_no_keyword(self):
        assert pattern_from_text("好看的日落") is None
        assert pattern_from_text("") is None
        assert pattern_from_text(None) is None


# ----------------------------------------------------------------------
# 降级链
# ----------------------------------------------------------------------
def _make_service(**overrides) -> CaseSearchService:
    cfg = load_settings(overrides={"perception.backend": "rule", **overrides})
    return CaseSearchService(cfg, processor_getter=lambda: None)


def _with_subject_processor(svc: CaseSearchService) -> CaseSearchService:
    """给 service 装一个"能检出主体"的假 processor。

    **必须注入才能测到存储层**：降级原因有优先级——
    ``感知失败 > 无主体 > 存储失败``。若 processor 为 None，
    永远只会得到 ``perception_failed``，测不到存储降级。
    这个次序本身是设计意图（先保证查询向量有意义，再谈检索），
    因此由本 helper 显式固化。
    """
    svc._processor_getter = lambda: SimpleNamespace(
        process=lambda frame, ctx: _make_snapshot()
    )
    return svc


class TestServiceDegradation:
    def test_store_error_reason_passthrough(self, monkeypatch):
        """存储层抛什么 reason，outcome 就带什么 reason（不吞、不改造）。"""
        svc = _with_subject_processor(_make_service())

        def _boom(*a, **kw):
            raise StoreError("search_failed", "boom")

        monkeypatch.setattr(svc, "_store_or_none", lambda: SimpleNamespace(search=_boom))
        outcome = svc.search(
            image=np.zeros((32, 32, 3), dtype=np.uint8), top_k=3
        )
        assert outcome.degraded is True
        assert outcome.degrade_reason == "search_failed"
        assert outcome.results == []

    def test_pure_text_with_healthy_store(self, monkeypatch):
        """纯文本 + Milvus 健康：标量结果 + 明确标注 text_embedding_unavailable。"""
        svc = _make_service()
        stub = SimpleNamespace(
            query_scalar=lambda *, limit, pattern: [
                {"case_id": "case-a", "image_ref": "assets/x.jpg", "similarity": 0.0,
                 "pattern": "center", "scene_tags": ["cafe"], "description": "d"}
            ],
            count=lambda: 42,
        )
        monkeypatch.setattr(svc, "_store_or_none", lambda: stub)
        outcome = svc.search(query_text="咖啡馆 居中", top_k=5)
        assert outcome.degraded is True
        assert outcome.degrade_reason == "text_embedding_unavailable"
        assert outcome.text_pattern_detected is True
        assert outcome.pattern_filter == "center"
        assert len(outcome.results) == 1
        assert outcome.results[0]["similarity"] == 0.0  # 无排序，诚实置 0
        assert outcome.index_size == 42

    def test_pure_text_without_pattern_hit(self, monkeypatch):
        """纯文本且无关键词：**库是通的**但没有过滤条件 → 空结果 + 降级原因不变。

        这里刻意用**健康的 store**（而不是 None）来隔离被测语义：若把 store
        设成 None，测到的其实是"存储不可用"那条路径（见上方
        test_pure_text_store_failure_wins），两者混在一起会让断言说不清在验什么。
        """
        svc = _make_service()
        stub = SimpleNamespace(
            query_scalar=lambda *, limit, pattern: [],
            count=lambda: 3,
        )
        monkeypatch.setattr(svc, "_store_or_none", lambda: stub)
        outcome = svc.search(query_text="好看的日落", top_k=3)
        assert outcome.degraded is True
        assert outcome.degrade_reason == "text_embedding_unavailable"
        assert outcome.results == []
        assert outcome.pattern_filter is None
        assert outcome.index_size == 3

    def test_pure_text_store_failure_wins(self, monkeypatch):
        """存储不可用优先于 text_embedding_unavailable（真实缺陷回归）。

        若把"store 返回 None"当成查无结果，Milvus 宕机这一更严重的事实会
        被"文本没有向量"这个次级原因盖住——调用方看到的是错的故障描述。
        """

        def _store_boom():
            raise StoreError("connection_failed", "down")

        svc = _make_service()
        monkeypatch.setattr(svc, "_store_or_none", _store_boom)
        outcome = svc.search(query_text="三分法", top_k=3)
        assert outcome.degraded is True
        assert outcome.degrade_reason == "connection_failed"

    def test_no_subject_degrades_before_store(self, monkeypatch):
        """无主体：在碰 Milvus **之前**就拒绝（几何维无真实来源）。"""
        svc = _make_service()
        # 故意放一个"一碰就炸"的 store——若被碰到测试就会失败，
        # 以此证明 no_subject 判定先于存储访问。
        def _boom(*a, **kw):
            raise AssertionError("no_subject 路径不应访问存储层")

        monkeypatch.setattr(svc, "_store_or_none", _boom)
        fake_processor = SimpleNamespace(
            process=lambda frame, ctx: _make_snapshot(subject_bbox=None)
        )
        svc._processor_getter = lambda: fake_processor
        outcome = svc.search(image=np.zeros((32, 32, 3), dtype=np.uint8))
        assert outcome.degraded is True
        assert outcome.degrade_reason == "no_subject"

    def test_perception_crash_degrades(self, monkeypatch):
        """感知层崩溃 → 200 + degraded（perception_failed），绝不抛出。"""
        svc = _make_service()
        def _crash(*a, **kw):
            raise RuntimeError("YOLO exploded")

        svc._processor_getter = lambda: SimpleNamespace(process=_crash)
        outcome = svc.search(image=np.zeros((32, 32, 3), dtype=np.uint8))
        assert outcome.degraded is True
        assert outcome.degrade_reason == "perception_failed"


class TestRealConnectionFailure:
    """真实连接失败（非 mock）：本机未启动 Milvus 时的行为。

    127.0.0.1 未监听端口 → TCP RST → 快速失败，不会拖慢套件。
    """

    def test_connection_refused_is_store_error(self):
        pytest.importorskip("pymilvus")
        svc = _make_service(**{"retrieval.uri": "http://127.0.0.1:59999"})
        with pytest.raises(StoreError) as excinfo:
            svc._store_or_none()
        assert excinfo.value.reason == "connection_failed"

    def test_service_swallows_connection_failure(self):
        pytest.importorskip("pymilvus")
        svc = _with_subject_processor(
            _make_service(**{"retrieval.uri": "http://127.0.0.1:59999"})
        )
        outcome = svc.search(
            image=np.zeros((32, 32, 3), dtype=np.uint8), top_k=2
        )
        assert outcome.degraded is True
        assert outcome.degrade_reason == "connection_failed"

    def test_failure_memory_avoids_retry(self):
        """首次失败后进程内不再重试（避免每请求叠加 5s 握手超时）。"""
        pytest.importorskip("pymilvus")
        svc = _with_subject_processor(
            _make_service(**{"retrieval.uri": "http://127.0.0.1:59999"})
        )
        svc.search(image=np.zeros((32, 32, 3), dtype=np.uint8))
        assert svc._store_failed is True
        # 第二次直接走降级，不再触碰网络
        outcome = svc.search(image=np.zeros((32, 32, 3), dtype=np.uint8))
        assert outcome.degrade_reason == "connection_failed"


class TestEnsureLoaded:
    """collection 的加载保证——**真跑才暴露的缺陷**，用假客户端常驻 CI。

    缺陷现场：写入进程退出后 collection 处于 ``released``，下一个进程只读时
    若不做 load，search 报 ``code=101: call load() before search``，表现是
    "库里有 40 条却永远搜不到"，且会被上层误判成 ``search_failed``。
    """

    @staticmethod
    def _store(client, monkeypatch) -> MilvusStore:
        st = MilvusStore(MilvusConfig(uri="http://stub:19530", collection="cases"))
        monkeypatch.setattr(st, "_get_client", lambda: client)
        return st

    @staticmethod
    def _existing_collection(state="NotLoad", load_impl=None):
        """构造"集合已存在"的假服务端。``load_impl`` 可注入异常。"""
        calls: list[str] = []

        class _Client:
            def has_collection(self, name):
                return True

            def describe_collection(self, name):
                return {"fields": [{"name": "vector", "params": {"dim": DIM}}]}

            def get_load_state(self, name):
                return SimpleNamespace(state=state)

            def load_collection(self, name):
                calls.append(name)
                if load_impl is not None:
                    load_impl(name)

        return _Client(), calls

    def test_loads_collection_when_released(self, monkeypatch):
        client, calls = self._existing_collection(state="NotLoad")
        self._store(client, monkeypatch).ensure_collection()
        assert calls == ["cases"], "released 状态必须触发 load，否则后续检索 code=101"

    def test_skips_load_when_already_loaded(self, monkeypatch):
        client, calls = self._existing_collection(state="Loaded")
        self._store(client, monkeypatch).ensure_collection()
        assert calls == [], "已加载时不应重复 load（多余 RPC）"

    def test_loads_when_load_state_unsupported(self, monkeypatch):
        """get_load_state 不可用时也要保底 load——保守好过漏。"""
        client, calls = self._existing_collection(state="NotLoad")

        def _unsupported(name):
            raise RuntimeError("unsupported by server")

        client.get_load_state = _unsupported  # type: ignore[method-assign]
        self._store(client, monkeypatch).ensure_collection()
        assert calls == ["cases"]

    def test_load_failure_is_store_error(self, monkeypatch):
        """load 失败要落到统一降级出口，而不是静默返回空结果。"""
        def _boom(name):
            raise RuntimeError("disk full")

        client, _ = self._existing_collection(load_impl=_boom)
        with pytest.raises(StoreError) as excinfo:
            self._store(client, monkeypatch).ensure_collection()
        assert excinfo.value.reason == "collection_load_failed"

    def test_dimension_mismatch_still_wins(self, monkeypatch):
        """维度不符优先于 load：宁可不加载，也不能加载一个口径错的库。"""

        class _Client:
            def has_collection(self, name):
                return True

            def describe_collection(self, name):
                return {"fields": [{"name": "vector", "params": {"dim": 512}}]}

            def get_load_state(self, name):
                raise AssertionError("维度校验应先失败，不应走到 load")

        with pytest.raises(StoreError) as excinfo:
            self._store(_Client(), monkeypatch).ensure_collection()
        assert excinfo.value.reason == "dimension_mismatch"


class TestWarmup:
    """启动期预热：把"首次接触 Milvus"钉在帧处理之前（崩溃规避）。"""

    def test_warmup_is_best_effort(self, monkeypatch):
        """预热失败只返回 False，不抛——启动绝不因连环依赖失败而中断。"""
        svc = _make_service(**{"retrieval.uri": "http://127.0.0.1:59998"})

        def _raise():
            raise StoreError("connection_failed", "down")

        monkeypatch.setattr(svc, "_store_or_none", _raise)
        assert svc.warmup() is False

    def test_warmup_returns_true_with_store(self, monkeypatch):
        svc = _make_service()
        monkeypatch.setattr(
            svc,
            "_store_or_none",
            lambda: SimpleNamespace(ensure_collection=lambda: None),
        )
        assert svc.warmup() is True
