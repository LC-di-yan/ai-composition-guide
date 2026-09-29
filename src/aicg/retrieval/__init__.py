"""检索层（FR-09）：可解释构图特征 + Milvus 向量检索。

对外入口：:class:`~aicg.retrieval.search.CaseSearchService`。
设计决策与降级链见 ``docs/research/Milvus本地方案调研.md`` §4。
"""

from .features import DIM, DIM_NAMES, features_from_snapshot
from .milvus_store import CaseRecord, MilvusConfig, MilvusStore, StoreError
from .search import VALID_PATTERNS, CaseSearchOutcome, CaseSearchService

__all__ = [
    "DIM",
    "DIM_NAMES",
    "CaseRecord",
    "CaseSearchOutcome",
    "CaseSearchService",
    "MilvusConfig",
    "MilvusStore",
    "StoreError",
    "VALID_PATTERNS",
    "features_from_snapshot",
]
