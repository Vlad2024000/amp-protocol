"""
AMP MemoryStore — high-level facade

Single entry point for 95% of use cases.
Wraps PgVectorBackend with a friendlier API.

Usage:
    store = MemoryStore("./memory.db", agent_id="claude", user_id="alice")
    m = store.add("Alice prefers concise answers", memory_type=MemoryType.PREFERENCE)
    results = store.search("communication style")
    store.forget(m.id)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..memory_object import (
    MemoryObject, MemoryTrust, MemoryScope,
    MemoryType, ScopeType, SourceType,
)
from .pgvector_backend import PgVectorBackend


class MemoryStore:
    """
    High-level AMP memory store.

    Parameters
    ----------
    path : str
        Path to SQLite database file (or PostgreSQL DSN if psycopg2 installed).
        Pass ":memory:" for an in-process ephemeral store.
    agent_id : str
        Identifier for this agent (e.g. "agent-claude", "agent-gpt").
    user_id : str, optional
        User identifier. Memories with scope="user" are shared across all
        agents with the same user_id.
    session_id : str, optional
        Session identifier. Memories with scope="session" are shared across
        agents in the same session.
    api_key : str, optional
        OpenAI API key. If provided, uses text-embedding-3-small for embeddings.
        Otherwise uses offline LSA embeddings.
    embedding_dim : int
        Dimensionality for LSA embeddings (default: 256).
    """

    def __init__(
        self,
        path:          str            = ":memory:",
        *,
        agent_id:      str            = "amp-agent",
        user_id:       Optional[str]  = None,
        session_id:    Optional[str]  = None,
        api_key:       Optional[str]  = None,
        embedding_dim: int            = 256,
    ):
        import os
        dsn = None
        if path.startswith("postgresql://") or path.startswith("postgres://"):
            dsn  = path
            path = ":memory:"
        os.environ.setdefault("AMP_DB_PATH", path)

        self._backend = PgVectorBackend(
            dsn          = dsn,
            agent_id     = agent_id,
            user_id      = user_id,
            session_id   = session_id,
            api_key      = api_key,
            prefer_local = api_key is None,
            embedding_dim= embedding_dim,
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def add(
        self,
        content:     str,
        *,
        memory_type: MemoryType          = MemoryType.FACT,
        importance:  float               = 0.7,
        tags:        Optional[List[str]] = None,
        structured:  Optional[Dict]      = None,
        scope:       ScopeType           = ScopeType.USER,
        source:      SourceType          = SourceType.USER_INPUT,
        confidence:  float               = 0.85,
        model:       str                 = "unknown",
    ) -> MemoryObject:
        """
        Create and store a memory.

        Returns the stored MemoryObject (includes generated id and embedding).

        Example:
            m = store.add(
                "Alice is building AMP, an AI memory protocol",
                memory_type=MemoryType.FACT,
                importance=0.9,
                tags=["project", "identity"],
            )
            print(m.id, m.weight())
        """
        m = MemoryObject(
            content     = content,
            memory_type = memory_type,
            importance  = importance,
            tags        = tags or [],
            structured  = structured or {},
            trust = MemoryTrust(
                agent_id   = self._backend.agent_id,
                model      = model,
                confidence = confidence,
                source     = source,
            ),
            scope = MemoryScope(
                type       = scope,
                agent_id   = self._backend.agent_id,
                user_id    = self._backend.user_id,
                session_id = self._backend.session_id,
            ),
        )
        return self._backend.write(m)

    def write(self, memory: MemoryObject) -> MemoryObject:
        """Write a pre-constructed MemoryObject directly."""
        return self._backend.write(memory)

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> Optional[MemoryObject]:
        """Get a memory by its UUID."""
        return self._backend.get(memory_id)

    def forget(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted."""
        return self._backend.delete(memory_id)

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        *,
        top_k:          int                        = 10,
        memory_types:   Optional[List[MemoryType]] = None,
        tags:           Optional[List[str]]        = None,
        min_weight:     float                      = 0.0,
        include_shared: bool                       = True,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across memories.

        Uses three-signal RRF fusion:
          - Vector cosine similarity (LSA or OpenAI embeddings)
          - BM25 text relevance
          - AMP decay-adjusted weight

        With include_shared=True, also searches memories from other agents
        that have the same user_id (cross-agent recall).

        Returns list of dicts:
            {
                "memory":     MemoryObject,
                "rrf_score":  float,   # combined RRF score
                "vec_score":  float,   # semantic similarity
                "bm25_score": float,   # keyword relevance
                "amp_weight": float,   # decay-adjusted importance
                "shared_from": str | None,  # agent_id if from another agent
            }

        Example:
            results = store.search("where does Alice live?")
            for r in results:
                print(r["memory"].content, r["rrf_score"])
        """
        return self._backend.search(
            query          = query,
            top_k          = top_k,
            memory_types   = memory_types,
            tags           = tags,
            min_weight     = min_weight,
            include_shared = include_shared,
        )

    def all(self, limit: int = 100) -> List[MemoryObject]:
        """List all memories, sorted by importance descending."""
        return self._backend.list_all(limit=limit)

    # ── Maintenance ───────────────────────────────────────────────────────

    def prune(self, min_weight: float = 0.05) -> int:
        """Remove memories below min_weight. Returns count deleted."""
        return self._backend.prune(min_weight=min_weight)

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        return self._backend.stats()

    # ── Sync ──────────────────────────────────────────────────────────────

    def sync_from(self, agent_id: str) -> Dict[str, int]:
        """
        Pull and integrate memories from another agent.
        Returns {"pulled": N, "conflicts": N, "skipped": N}.
        """
        from ..sync.protocol import AgentSyncProtocol

        # AgentSyncProtocol expects a SQLiteBackend-compatible object
        # PgVectorBackend has the same interface for list_all/get/write
        class _CompatBackend:
            def __init__(self, backend):
                self._b = backend
                self.agent_id = backend.agent_id
                self.user_id  = backend.user_id
            def list_all(self, limit=500): return self._b.list_all(limit)
            def get(self, mid): return self._b.get(mid)
            def write(self, m): return self._b.write(m)
            def search(self, q, **kw): return self._b.search(q, **kw)

        syncer = AgentSyncProtocol(_CompatBackend(self._backend))
        result = syncer.pull_from_agent(agent_id)
        return {
            "pulled":    result.pulled,
            "conflicts": result.conflicts,
            "resolved":  result.resolved,
            "skipped":   result.skipped,
        }

    def export_snapshot(self) -> Dict[str, Any]:
        """Export all shareable memories as a JSON-serializable dict."""
        from ..sync.protocol import AgentSyncProtocol

        class _CompatBackend:
            def __init__(self, b):
                self._b = b
                self.agent_id = b.agent_id
                self.user_id  = b.user_id
            def list_all(self, limit=1000): return self._b.list_all(limit)
            def get(self, mid): return self._b.get(mid)
            def write(self, m): return self._b.write(m)

        return AgentSyncProtocol(_CompatBackend(self._backend)).export_snapshot()

    # ── Dunder ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.stats()["total"]

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryStore(agent={self._backend.agent_id!r}, "
            f"memories={s['total']}, "
            f"embedder={self._backend.embedder.mode})"
        )
