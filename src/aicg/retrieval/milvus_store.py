"""Milvus 存储层（FR-09）：collection 管理 + 写入 + 向量检索。

**pymilvus 是可选依赖**（requirements.txt 已注释说明）：未安装时本模块
**禁止在导入期失败**——``_import_pymilvus()`` 返回 None，由上层
:class:`~aicg.retrieval.search.CaseSearchService` 转成 ``200 + degraded``。
这是"运行类问题不 5xx"（NFR-R1）在依赖层面的第一道落实。

**度量契约**：HNSW + COSINE。Milvus 对 COSINE 返回的 ``distance =
1 - 相似度``（越小越相似），故 ``similarity = clamp(1 - distance, 0, 1)``，
与接口契约 ``CaseSearchResult.similarity ∈ [0,1]`` 对齐。

**为什么用 MilvusClient 而非 ORM 连接**：pymilvus 2.5 的 ``MilvusClient``
是官方推荐的新 API，单连接对象即含 collection CRUD 与 search，无需
再维护 ``connections.connect`` 全局态，测试也更容易隔离。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from ..observability import get_logger
from .features import DIM, DIM_NAMES

log = get_logger("retrieval.milvus_store")

# scene_tags 数组容量：场景标签是有限枚举（咖啡馆/街拍/旅行等），
# 8 个足够；超出直接截断而不是报错。
_MAX_TAGS = 8


def _import_pymilvus():
    """可选依赖探测。返回模块或 None，绝不抛异常。"""
    try:
        return importlib.import_module("pymilvus")
    except Exception:  # noqa: BLE001 — ImportError 之外还可能有依赖链损坏
        return None


@dataclass
class MilvusConfig:
    """连接与索引参数（来源：settings.retrieval，见 settings.py）。"""

    uri: str = "http://127.0.0.1:19530"
    collection: str = "composition_cases"
    # HNSW 参数（调研文档 §4.3）：案例库 ≤ 数百条，M=16/efC=200 稳妥，
    # ef=64 为查询侧精度-延迟平衡点。
    hnsw_m: int = 16
    ef_construction: int = 200
    ef: int = 64
    timeout_s: float = 5.0


@dataclass
class CaseRecord:
    """一条案例：向量 + 标量字段。image_ref 指向仓库内素材相对路径。"""

    case_id: str
    vector: list[float]
    image_ref: str
    pattern: str = "unknown"
    scene_tags: list[str] = field(default_factory=list)
    description: str | None = None


class MilvusStore:
    """对 pymilvus MilvusClient 的薄封装。所有方法失败都抛 :class:`StoreError`，
    由上层转降级——本层不做"吞错返回空结果"那种掩盖性处理。
    """

    def __init__(self, cfg: MilvusConfig) -> None:
        self.cfg = cfg
        self._client: Any | None = None
        self._checked: set[str] = set()
        """已确认存在的 collection 缓存（进程内，避免每次 search 探测）。"""

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """惰性建连。连接失败抛 StoreError（上层转降级）。"""
        if self._client is not None:
            return self._client
        pymilvus = _import_pymilvus()
        if pymilvus is None:
            raise StoreError("pymilvus_unavailable", "pymilvus 未安装，向量检索不可用")
        try:
            client = pymilvus.MilvusClient(
                uri=self.cfg.uri,
                timeout=self.cfg.timeout_s,
            )
            # 真正探测一次服务端（MilvusClient 构造不会立即握手）
            _ = client.list_collections()
        except Exception as exc:  # noqa: BLE001 — 网络/版本/鉴权问题统一归连接失败
            raise StoreError(
                "connection_failed", f"Milvus 连接失败（{self.cfg.uri}）: {exc}"
            ) from exc
        self._client = client
        log.info("Milvus 已连接: %s", self.cfg.uri)
        return client

    # ------------------------------------------------------------------
    # Collection 生命周期
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """集合不存在则按契约创建（含 HNSW 索引）。存在则校验维度。

        维度校验的意义：``DIM_NAMES`` 契约演进后（如加维）旧库不重建
        会导致向量错位——宁可让建库脚本显式失败，也不要静默搜出垃圾。
        """
        client = self._get_client()
        name = self.cfg.collection
        if name in self._checked:
            return
        if client.has_collection(name):
            desc = client.describe_collection(name)
            for f in desc.get("fields", []):
                if f.get("name") == "vector":
                    dim = (f.get("params") or {}).get("dim")
                    if dim is not None and int(dim) != DIM:
                        raise StoreError(
                            "dimension_mismatch",
                            f"既有 collection 维度 {dim} ≠ 契约维度 {DIM}，"
                            "请重建索引（scripts/build_case_index.py --rebuild）",
                        )
            self._checked.add(name)
            return

        pymilvus = _import_pymilvus()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("case_id", pymilvus.DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", pymilvus.DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("pattern", pymilvus.DataType.VARCHAR, max_length=32)
        schema.add_field(
            "scene_tags",
            pymilvus.DataType.ARRAY,
            element_type=pymilvus.DataType.VARCHAR,
            max_capacity=_MAX_TAGS,
            max_length=64,
        )
        schema.add_field("image_ref", pymilvus.DataType.VARCHAR, max_length=512)
        schema.add_field("description", pymilvus.DataType.VARCHAR, max_length=1024, nullable=True)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": self.cfg.hnsw_m, "efConstruction": self.cfg.ef_construction},
        )
        try:
            client.create_collection(name, schema=schema, index_params=index_params)
        except Exception as exc:  # noqa: BLE001
            raise StoreError(
                "collection_create_failed", f"创建 collection 失败: {exc}"
            ) from exc
        log.info("已创建 collection: %s（dim=%d, HNSW/COSINE）", name, DIM)
        self._checked.add(name)

    # ------------------------------------------------------------------
    # 写入 / 检索 / 统计
    # ------------------------------------------------------------------
    def upsert(self, records: list[CaseRecord]) -> int:
        """批量写入（case_id 相同即覆盖）。返回成功条数。"""
        if not records:
            return 0
        client = self._get_client()
        self.ensure_collection()
        rows = [
            {
                "case_id": r.case_id,
                "vector": r.vector,
                "pattern": r.pattern or "unknown",
                "scene_tags": list(r.scene_tags)[:_MAX_TAGS],
                "image_ref": r.image_ref,
                "description": r.description or "",
            }
            for r in records
        ]
        try:
            res = client.upsert(self.cfg.collection, rows)
        except Exception as exc:  # noqa: BLE001
            raise StoreError("upsert_failed", f"写入失败: {exc}") from exc
        # pymilvus upsert 返回 {'upsert_count': N, ...} 或 MutateResult
        count = int(res.get("upsert_count", len(rows))) if isinstance(res, dict) else len(rows)
        return count

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索。返回 [{case_id, image_ref, similarity, pattern, scene_tags, description}]。

        Args:
            vector: 8 维查询向量。
            top_k: 返回条数上限。
            pattern: 标量过滤（构图模式），None 不过滤。

        Raises:
            StoreError: 连接/集合/查询失败（上层转降级）。
        """
        client = self._get_client()
        self.ensure_collection()
        expr = f'pattern == "{pattern}"' if pattern else ""
        try:
            res = client.search(
                self.cfg.collection,
                data=[vector],
                limit=top_k,
                filter=expr or None,
                search_params={"metric_type": "COSINE", "params": {"ef": self.cfg.ef}},
                output_fields=["case_id", "image_ref", "pattern", "scene_tags", "description"],
                timeout=self.cfg.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError("search_failed", f"检索失败: {exc}") from exc

        out: list[dict[str, Any]] = []
        # MilvusClient.search 返回形如 [[{id, distance, entity}, ...]]（单查询 → 外层 1 条）
        for hit in (res[0] if res else []):
            entity = hit.get("entity") or {}
            distance = float(hit.get("distance", 0.0))
            similarity = 1.0 - distance
            # COSINE 的 1-sim 在 [-1, 2] 理论区间；契约只收 [0,1]，负值裁 0
            similarity = 0.0 if similarity < 0.0 else (1.0 if similarity > 1.0 else similarity)
            out.append(
                {
                    "case_id": entity.get("case_id", str(hit.get("id", ""))),
                    "image_ref": entity.get("image_ref", ""),
                    "similarity": round(similarity, 4),
                    "pattern": entity.get("pattern", "unknown"),
                    "scene_tags": list(entity.get("scene_tags") or []),
                    "description": entity.get("description") or None,
                }
            )
        return out

    def query_scalar(
        self,
        *,
        limit: int = 5,
        pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """纯标量过滤查询（无向量，不排序）——纯文本检索的降级实现。

        按插入序返回匹配行，**没有相似度语义**：这是"标量过滤"的诚实
        用法，调用方必须携带 degraded 标记，不得把结果伪装成排序结果。
        """
        client = self._get_client()
        self.ensure_collection()
        expr = f'pattern == "{pattern}"' if pattern else ""
        try:
            rows = client.query(
                self.cfg.collection,
                filter=expr or None,
                limit=limit,
                output_fields=["case_id", "image_ref", "pattern", "scene_tags", "description"],
                timeout=self.cfg.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError("search_failed", f"标量查询失败: {exc}") from exc
        return [
            {
                "case_id": r.get("case_id", ""),
                "image_ref": r.get("image_ref", ""),
                "similarity": 0.0,  # 无向量排序 → 无相似度，诚实置 0
                "pattern": r.get("pattern", "unknown"),
                "scene_tags": list(r.get("scene_tags") or []),
                "description": r.get("description") or None,
            }
            for r in (rows or [])
        ]

    def count(self) -> int:
        """库内案例总数（用于响应里诚实暴露索引规模）。"""
        client = self._get_client()
        self.ensure_collection()
        try:
            stats = client.get_collection_stats(self.cfg.collection)
        except Exception as exc:  # noqa: BLE001
            raise StoreError("stats_failed", f"获取统计失败: {exc}") from exc
        return int(stats.get("row_count", 0))

    def drop(self) -> None:
        """删除 collection（建库脚本 --rebuild 用）。"""
        client = self._get_client()
        if client.has_collection(self.cfg.collection):
            client.drop_collection(self.cfg.collection)
        self._checked.discard(self.cfg.collection)


class StoreError(Exception):
    """存储层统一异常。``reason`` 是面向调用方的降级原因枚举值。"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


__all__ = [
    "DIM_NAMES",
    "CaseRecord",
    "MilvusConfig",
    "MilvusStore",
    "StoreError",
]
