"""会话状态存储（API-01/02 的支撑）。

对应需求：FR-05（实时引导循环）、NFR-R4（可恢复）
对应文档：《数据模型与接口.md》§4 —— "会话状态存储方式：待确认"

**设计决策（原文档留白，此处定型）**：

原文档把"会话状态存储"标为待确认。本实现选择 **内存字典 + TTL 淘汰**，
理由与代价如实记录：

1. **为什么不用 Redis**：本项目是单机 Demo，引入 Redis 会让"一条命令跑起来"
   变成"先装 Redis"。对作品集项目而言，可运行性 > 架构完备性。
2. **为什么必须有 TTL**：会话对象持有 `FrameContext`（含快照列表），
   长跑必然吃内存。TTL 是防止 Demo 长时间后台挂死的最低成本手段。
3. **代价（诚实标注）**：进程重启会话即丢；多副本部署无法共享。
   生产环境做法：把 `SessionStore` 换成 Redis 实现即可 —— 因此这里定义成
   **接口 + 内存实现**，替换点单一，不需要动路由层。

线程安全：FastAPI 同步路由跑在线程池里，可能并发访问，故加锁。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..observability import LatencyTracker, get_logger
from ..pipeline.frame_processor import FrameContext

log = get_logger("api.session")

DEFAULT_TTL_S = 1800.0
"""会话空闲存活时间（秒）。30 分钟无活动即回收。"""

MAX_SESSIONS = 256
"""上限保护：超过则淘汰最久未活动的会话。"""


@dataclass
class Session:
    """一个拍摄会话的运行时状态。"""

    session_id: str
    created_at_ms: int
    ctx: FrameContext = field(default_factory=FrameContext)
    latency: LatencyTracker = field(default_factory=lambda: LatencyTracker(window=300))
    last_active_ms: int = 0
    shot_count: int = 0
    language_calls: int = 0

    def touch(self, now_ms: int) -> None:
        self.last_active_ms = now_ms


class SessionStore(Protocol):
    """会话存储接口 —— 换 Redis 只需实现本协议。"""

    def create(self) -> Session: ...

    def get(self, session_id: str) -> Session | None: ...

    def delete(self, session_id: str) -> bool: ...

    def sweep(self) -> int: ...

    def count(self) -> int: ...


class InMemorySessionStore:
    """内存会话存储（带 TTL 与容量双保护）。"""

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        max_sessions: int = MAX_SESSIONS,
        clock_ms=lambda: int(time.time() * 1000),
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl_ms = int(ttl_s * 1000)
        self._max = max_sessions
        self._clock_ms = clock_ms
        self._counter = 0

    # ------------------------------------------------------------------
    def create(self) -> Session:
        """新建会话。返回的 ``session_id`` 形如 ``s-xxxxxxxx``，便于日志检索。"""
        with self._lock:
            self._counter += 1
            now = self._clock_ms()
            sid = f"s-{now:x}-{self._counter:04d}"
            sess = Session(session_id=sid, created_at_ms=now, last_active_ms=now)
            self._sessions[sid] = sess
            self._evict_if_needed_locked()
            log.info("会话创建: %s（当前 %d 个）", sid, len(self._sessions))
            return sess

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return None
            now = self._clock_ms()
            # 过期即视为不存在（惰性删除）
            if self._ttl_ms > 0 and now - sess.last_active_ms > self._ttl_ms:
                del self._sessions[session_id]
                log.info("会话过期回收: %s", session_id)
                return None
            sess.touch(now)
            return sess

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def sweep(self) -> int:
        """主动清理过期会话，返回清理数量。"""
        now = self._clock_ms()
        with self._lock:
            stale = [
                sid
                for sid, s in self._sessions.items()
                if self._ttl_ms > 0 and now - s.last_active_ms > self._ttl_ms
            ]
            for sid in stale:
                del self._sessions[sid]
        if stale:
            log.info("批量回收过期会话: %d 个", len(stale))
        return len(stale)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    def _evict_if_needed_locked(self) -> None:
        """容量超限时淘汰最久未活动的会话（LRU）。调用方必须持锁。"""
        while len(self._sessions) > self._max:
            victim = min(self._sessions.items(), key=lambda kv: kv[1].last_active_ms)
            del self._sessions[victim[0]]
            log.warning("会话容量超限，淘汰最久未活动: %s", victim[0])


__all__ = [
    "DEFAULT_TTL_S",
    "MAX_SESSIONS",
    "InMemorySessionStore",
    "Session",
    "SessionStore",
]
