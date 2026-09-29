"""案例检索服务（FR-09）：请求路由 + 降级链。

**降级链（调研文档 §4.4 定型，全部 200 + degraded，绝不 5xx）**：

========== ============================== =====================================
场景       处理                            degrade_reason
========== ============================== =====================================
有图       感知 → 8 维向量 → Milvus 检索   （正常路径，degraded=false）
图无主体   几何维无真实来源，拒绝编造       ``no_subject``
纯文本     无向量，标量过滤返回未排序行     ``text_embedding_unavailable``
pymilvus   依赖缺失，导入期即感知           ``pymilvus_unavailable``
Milvus     连接/集合/查询失败               ``connection_failed`` 等枚举
========== ============================== =====================================

**query_text 的角色**：仅做 pattern 标量过滤（关键词 → 构图模式），
不假装能做文本语义检索——那是 CLIP 双塔的活，本期明确不做（调研文档
§4.1，诚实降级优于假装能搜）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..observability import get_logger
from ..settings import Settings
from .features import features_from_snapshot
from .milvus_store import MilvusConfig, MilvusStore, StoreError

log = get_logger("retrieval.search")

# 合法 pattern 过滤值（与 schemas.CompositionPattern 对齐；unknown 不许
# 作为过滤条件——"搜未知模式"是语义矛盾，属于参数错误由路由层拦）
VALID_PATTERNS: frozenset[str] = frozenset(
    {"rule_of_thirds", "center", "symmetric", "diagonal", "framing"}
)

# query_text 关键词 → pattern。覆盖项目文案里实际使用的说法；
# 未命中关键词 = 文本不提供过滤信息（不是错误）。
_TEXT_PATTERN_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("三分", "三分法", "九宫格"), "rule_of_thirds"),
    (("居中", "中心构图", "正中"), "center"),
    (("对称", "镜像"), "symmetric"),
    (("对角", "斜线"), "diagonal"),
    (("框架", "前景框", "框式"), "framing"),
)


@dataclass
class CaseSearchOutcome:
    """检索结果 + 降级元数据。routes 层直接序列化为响应体。"""

    results: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    degrade_reason: str | None = None
    index_size: int | None = None
    """命中的案例库规模（None = 降级时连统计都拿不到）"""
    pattern_filter: str | None = None
    text_pattern_detected: bool = False
    """query_text 是否命中了 pattern 关键词（可观测，便于调试）"""

    def to_response(self) -> dict[str, Any]:
        return {
            "results": self.results,
            "total": len(self.results),
            "index_size": self.index_size,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "pattern_filter": self.pattern_filter,
        }


def pattern_from_text(query_text: str) -> str | None:
    """从查询文本提取 pattern 过滤条件；无命中返回 None。"""
    if not query_text:
        return None
    for keywords, pattern in _TEXT_PATTERN_KEYWORDS:
        if any(k in query_text for k in keywords):
            return pattern
    return None


class CaseSearchService:
    """对外的案例检索门面：把"图 / 文"两种请求翻译成向量或标量查询。

    依赖注入说明：``processor_getter`` 是**惰性工厂**——检索是低频接口，
    不应在应用启动时就把感知后端（可能含 YOLO 权重）拉起来；
    与 ``AppState`` 其它惰性 property 同一纪律。
    """

    def __init__(
        self,
        settings: Settings,
        processor_getter: Callable[[], Any],
    ) -> None:
        rcfg = settings.retrieval
        self.cfg = MilvusConfig(
            uri=rcfg.resolved_uri,
            collection=rcfg.collection,
            hnsw_m=rcfg.hnsw_m,
            ef_construction=rcfg.ef_construction,
            ef=rcfg.ef,
            timeout_s=rcfg.timeout_s,
        )
        self._store: MilvusStore | None = None
        self._store_failed: bool = False
        """上次连库失败标记：连不上的服务在进程内不再反复重试（每次请求
        都去握手会把 5s 超时叠加到响应延迟上）。"""
        self._last_store_reason: str | None = None
        """上次失败的原因枚举。降级记忆只返回 None 是不够的——调用方
        若拿 None 当 store 用会 AttributeError 并穿透成 500（真实缺陷，
        由 test_failure_memory_avoids_retry 捕获），因此必须连原因一起
        记住，走**同一条**降级出口。"""
        self._processor_getter = processor_getter

    # ------------------------------------------------------------------
    def _store_or_none(self) -> MilvusStore | None:
        """惰性建 store；pymilvus 缺失或曾连接失败 → None（走降级）。"""
        if self._store_failed:
            return None
        if self._store is None:
            store = MilvusStore(self.cfg)
            try:
                store._get_client()  # 立即握手，把失败提前暴露
            except StoreError as exc:
                log.warning("Milvus 不可用（降级）: %s", exc)
                self._store_failed = True
                self._last_store_reason = exc.reason
                raise
            self._store = store
        return self._store

    # ------------------------------------------------------------------
    def warmup(self) -> bool:
        """尽早完成"第一次接触 Milvus"。返回是否成功；失败不影响调用方启动。

        为什么必须有这一步（实测结论，见 ``测试与验收.md`` §4.3f）：
        在 Windows + Milvus Lite 组合下，**若 Milvus 客户端的首次初始化
        发生在帧处理链路（OpenCV + 感知/评分）之后**，进程会在 milvus_lite
        的 ``pa.RecordBatch`` 写入路径上触发 SIGSEGV（访问违规）。顺序反过来
        ——先握手/加载，再跑图像链路——则全程稳定。

        warmup 把"首次接触"钉死在启动阶段：代价是一次握手（连不上就降级，
        不拖垮启动），收益是消掉这个崩溃窗口。**这只是崩溃规避，不是
        功能保证**：预训练完后 Milvus 仍不可用时，所有请求照旧走降级链。
        """
        try:
            store = self._store_or_none()
            if store is None:
                return False
            store.ensure_collection()
        except Exception as exc:  # noqa: BLE001 — 预热失败一律降级，不冒泡
            log.warning("检索库预热失败（降级，不影响启动）: %s", exc)
            return False
        log.info("检索库预热完成: %s", self.cfg.uri)
        return True

    def _safe_index_size(self, store: MilvusStore | None) -> int | None:
        if store is None:
            return None
        try:
            return store.count()
        except StoreError:
            return None

    # ------------------------------------------------------------------
    def search(
        self,
        *,
        image: np.ndarray | None = None,
        query_text: str | None = None,
        top_k: int = 5,
        pattern: str | None = None,
    ) -> CaseSearchOutcome:
        """执行检索。调用方（routes）已保证 image/query_text 至少一项非空。

        image 是**已解码**的 BGR ndarray（routes 层负责解码与 400）。
        本方法内部一切失败都转成降级 outcome，不抛异常。
        """
        effective_pattern = pattern or pattern_from_text(query_text or "")
        text_hit = pattern is None and effective_pattern is not None

        # ---------- 纯文本：无向量，标量过滤降级路径 ----------
        if image is None:
            return self._scalar_only_outcome(
                query_text=query_text or "",
                top_k=top_k,
                pattern_filter=effective_pattern,
                text_hit=text_hit,
            )

        # ---------- 有图：感知 → 特征向量 ----------
        try:
            vector = self._features_from_image(image)
        except Exception as exc:  # noqa: BLE001 — 感知层任何异常都降级，不 5xx
            log.warning("检索查询帧感知失败（降级）: %s", exc)
            return CaseSearchOutcome(
                degraded=True,
                degrade_reason="perception_failed",
                pattern_filter=effective_pattern,
            )
        if vector is None:
            # 未检出主体：几何维无真实来源，拒绝用兜底框伪装相似度
            return CaseSearchOutcome(
                degraded=True,
                degrade_reason="no_subject",
                pattern_filter=effective_pattern,
            )

        # ---------- 向量检索（存储层失败 → 降级） ----------
        try:
            store = self._store_or_none()
            if store is None:
                # 降级记忆命中：仍走 StoreError 出口，绝不让 None 传下去
                raise StoreError(
                    self._last_store_reason or "pymilvus_unavailable",
                    "Milvus 不可用（进程内已判定，不再重试）",
                )
            results = store.search(vector, top_k=top_k, pattern=effective_pattern)
        except StoreError as exc:
            return CaseSearchOutcome(
                degraded=True,
                degrade_reason=exc.reason,
                pattern_filter=effective_pattern,
                text_pattern_detected=text_hit,
            )
        return CaseSearchOutcome(
            results=results,
            index_size=self._safe_index_size(store),
            pattern_filter=effective_pattern,
            text_pattern_detected=text_hit,
        )

    # ------------------------------------------------------------------
    def _features_from_image(self, image: np.ndarray) -> list[float] | None:
        """图像 → 8 维向量。复用帧管线（感知 + 评分一次成型）。

        用无副作用的临时 ``FrameContext``：检索是独立请求，不应污染
        任何会话的防抖状态。
        """
        from ..camera.base import Frame
        from ..pipeline.frame_processor import FrameContext

        processor = self._processor_getter()
        snapshot = processor.process(
            Frame(image=image, frame_id=0, timestamp_ms=0),
            FrameContext(),
        )
        return features_from_snapshot(snapshot)

    def _scalar_only_outcome(
        self,
        *,
        query_text: str,
        top_k: int,
        pattern_filter: str | None,
        text_hit: bool,
    ) -> CaseSearchOutcome:
        """纯文本请求：200 + degraded（text_embedding_unavailable）。

        若文本命中 pattern 关键词则做标量过滤查询（未排序行，similarity
        恒 0）；否则连过滤条件都没有，返回空结果。两种情况都明确标注
        降级原因——**绝不**让调用方误以为文本语义检索生效了。
        """
        results: list[dict[str, Any]] = []
        index_size: int | None = None
        try:
            store = self._store_or_none()
            if store is None:
                # 与 image 路径同一条纪律：存储不可用是**首要原因**，
                # 不能被 "文本无向量" 这个次级原因盖过去。
                #
                # 这段必须写在这里的原因（真实缺陷，第三次踩同一类坑）：
                # 启动期 warmup() 会先把连接失败记进 _store_failed，因此后续
                # _store_or_none() 直接返回 None（故意不再重试）。若这里把
                # None 当成"查无结果"，纯文本请求就会报
                # text_embedding_unavailable，把"Milvus 根本没起来"这个更严重
                # 的事实藏掉——由全量套件 test_pure_text_degrades_with_reason
                # 捕获。相关教训见 测试与验收.md §4.3g。
                raise StoreError(
                    self._last_store_reason or "pymilvus_unavailable",
                    "Milvus 不可用（进程内已判定，不再重试）",
                )
            # 库规模与"有没有关键词"无关：连得上就该如实报告，否则响应会在
            # 无命中时假称"不知道库有多大"。
            index_size = self._safe_index_size(store)
            if pattern_filter is not None:
                results = store.query_scalar(limit=top_k, pattern=pattern_filter)
        except StoreError as exc:
            return CaseSearchOutcome(
                degraded=True,
                degrade_reason=exc.reason,
                pattern_filter=pattern_filter,
                text_pattern_detected=text_hit,
            )
        return CaseSearchOutcome(
            results=results,
            degraded=True,
            degrade_reason="text_embedding_unavailable",
            index_size=index_size,
            pattern_filter=pattern_filter,
            text_pattern_detected=text_hit,
        )
