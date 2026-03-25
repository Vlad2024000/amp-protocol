"""
AMP SQLiteBackend — v0.2

Persistent storage for MemoryObjects using SQLite.
  - WAL journal mode (concurrent reads, safe writes)
  - JSON columns for structured/relations/trust/scope
  - BM25 + AMP hybrid search via HybridRetriever
  - Multi-agent scope filtering (private/session/user/public)
  - Embedding column (BLOB) — ready for pgvector upgrade

Schema:
  memories(id, content, embedding, structured, memory_type,
           tags, relations, created_at, accessed_at,
           importance, decay_rate, permanence, trust, scope,
           agent_id, user_id, session_id)

  sync_log(id, event_type, memory_id, agent_id, ts, payload)
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..memory_object import (
    MemoryObject, MemoryType, RelationType,
    ScopeType, SourceType, MemoryRelation,
    MemoryTrust, MemoryScope,
)
from .bm25 import HybridRetriever


# ── Serialization helpers ─────────────────────────────────────────────────

def _pack_embedding(vec: Optional[List[float]]) -> Optional[bytes]:
    if vec is None:
        return None
    return struct.pack(f"{len(vec)}f", *vec)

def _unpack_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))

def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ── Schema ────────────────────────────────────────────────────────────────

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    embedding    BLOB,
    structured   TEXT NOT NULL DEFAULT '{}',
    memory_type  TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '[]',
    relations    TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL,
    accessed_at  TEXT NOT NULL,
    importance   REAL NOT NULL DEFAULT 0.5,
    decay_rate   REAL NOT NULL DEFAULT 0.01,
    permanence   REAL NOT NULL DEFAULT 0.0,
    trust        TEXT,
    scope        TEXT,
    agent_id     TEXT,
    user_id      TEXT,
    session_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_agent   ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_user    ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_type    ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

CREATE TABLE IF NOT EXISTS sync_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    memory_id  TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    ts         TEXT NOT NULL,
    payload    TEXT
);

CREATE INDEX IF NOT EXISTS idx_sync_agent ON sync_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_sync_ts    ON sync_log(ts);
"""


# ── Backend ───────────────────────────────────────────────────────────────

class SQLiteBackend:
    """
    Thread-safe SQLite backend for AMP MemoryStore.

    One DB file per user. Multiple agents share the same DB
    (scoped by agent_id) — enabling cross-agent memory access.

    Usage:
        backend = SQLiteBackend("./amp_data/user-andrii.db",
                                agent_id="agent-claude",
                                user_id="user-andrii")
        backend.write(memory_object)
        results = backend.search("startup idea", top_k=10)
    """

    def __init__(
        self,
        db_path: str,
        agent_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.db_path    = str(db_path)
        self.agent_id   = agent_id
        self.user_id    = user_id
        self.session_id = session_id

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._retriever = HybridRetriever()
        self._lock = threading.Lock()

        self._init_db()
        self._warm_retriever()

    # ── Connection management ─────────────────────────────────────────────

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        self._conn.executescript(DDL)
        self._conn.commit()

    def _warm_retriever(self) -> None:
        """Load all memories into BM25 index on startup."""
        self._sync_retriever()

    def _sync_retriever(self, ids: Optional[List[str]] = None) -> None:
        """
        Sync BM25 retriever with DB.
        If ids provided, only sync those. Otherwise sync all missing.
        """
        if ids:
            known = set(ids)
        else:
            known = set(self._retriever._meta.keys())
            all_ids = {r[0] for r in self._conn.execute("SELECT id FROM memories").fetchall()}
            missing = all_ids - known
            if not missing:
                return
            known = missing  # only fetch missing

        if not known:
            return

        placeholders = ",".join("?" * len(known))
        rows = self._conn.execute(
            f"SELECT id, content, tags, importance, decay_rate, permanence, created_at FROM memories WHERE id IN ({placeholders})",
            list(known)
        ).fetchall()
        for r in rows:
            self._retriever.index(
                doc_id     = r["id"],
                text       = r["content"],
                importance = r["importance"],
                decay_rate = r["decay_rate"],
                permanence = r["permanence"],
                created_at = _str_to_dt(r["created_at"]),
                tags       = json.loads(r["tags"]),
            )

    # ── Write ─────────────────────────────────────────────────────────────

    def write(self, m: MemoryObject) -> MemoryObject:
        trust_json = None
        if m.trust:
            trust_json = json.dumps({
                "agent_id":    m.trust.agent_id,
                "model":       m.trust.model,
                "confidence":  m.trust.confidence,
                "source":      m.trust.source.value,
                "verified_by": m.trust.verified_by,
            })

        scope_json = None
        agent_id = session_id = user_id = None
        if m.scope:
            scope_json = json.dumps({
                "type":       m.scope.type.value,
                "agent_id":   m.scope.agent_id,
                "session_id": m.scope.session_id,
                "user_id":    m.scope.user_id,
            })
            agent_id   = m.scope.agent_id
            session_id = m.scope.session_id
            user_id    = m.scope.user_id

        relations_json = json.dumps([
            {"target_id": r.target_id, "relation_type": r.relation_type.value, "strength": r.strength}
            for r in m.relations
        ])

        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, embedding, structured, memory_type, tags, relations,
                 created_at, accessed_at, importance, decay_rate, permanence,
                 trust, scope, agent_id, user_id, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                m.id,
                m.content,
                _pack_embedding(m.embedding),
                json.dumps(m.structured),
                m.memory_type.value,
                json.dumps(m.tags),
                relations_json,
                _dt_to_str(m.created_at),
                _dt_to_str(m.accessed_at),
                m.importance,
                m.decay_rate,
                m.permanence,
                trust_json,
                scope_json,
                agent_id,
                user_id,
                session_id,
            ))
            self._conn.commit()

            # Update BM25 index
            self._retriever.index(
                doc_id     = m.id,
                text       = m.content,
                importance = m.importance,
                decay_rate = m.decay_rate,
                permanence = m.permanence,
                created_at = m.created_at,
                tags       = m.tags,
            )

            # Log sync event
            self._log_event("write", m.id, self.agent_id)

        return m

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> Optional[MemoryObject]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        m = self._row_to_obj(row)
        # Update accessed_at
        now = _dt_to_str(datetime.now(timezone.utc))
        self._conn.execute(
            "UPDATE memories SET accessed_at=? WHERE id=?", (now, memory_id)
        )
        self._conn.commit()
        return m

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id=?", (memory_id,)
            )
            self._conn.commit()
            deleted = cur.rowcount > 0
            if deleted:
                self._retriever.remove(memory_id)
                self._log_event("delete", memory_id, self.agent_id)
        return deleted

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_weight: float = 0.0,
        include_shared: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + AMP weight search with scope filtering.

        include_shared=True: return memories from other agents
        scoped to 'user' or 'public' for the same user_id.
        This is the core cross-agent feature.
        """
        # Build scope filter
        scope_clause, scope_params = self._scope_filter(include_shared)

        # Get candidate IDs from DB (scope + type filter)
        type_clause = ""
        type_params: List[str] = []
        if memory_types:
            placeholders = ",".join("?" * len(memory_types))
            type_clause = f"AND memory_type IN ({placeholders})"
            type_params = [mt.value for mt in memory_types]

        tag_clause  = ""
        tag_params: List[str] = []
        if tags:
            # SQLite JSON: check if any tag present
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            tag_clause = f"AND ({tag_conditions})"
            tag_params = [f'%"{t}"%' for t in tags]

        sql = f"""
            SELECT id FROM memories
            WHERE {scope_clause}
            {type_clause}
            {tag_clause}
        """
        params = scope_params + type_params + tag_params
        rows = self._conn.execute(sql, params).fetchall()
        candidate_ids = [r["id"] for r in rows]

        if not candidate_ids:
            return []

        # Sync BM25 index with any docs written by other agents
        self._sync_retriever()

        # Hybrid retrieval
        results = self._retriever.search(
            query        = query,
            top_k        = top_k,
            candidate_ids= candidate_ids,
            query_tags   = tags,
            min_weight   = min_weight,
        )

        # Fetch full objects for results
        if not results:
            return []

        id_to_score = {did: (fs, bs, aw) for did, fs, bs, aw in results}
        placeholders = ",".join("?" * len(id_to_score))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            list(id_to_score.keys())
        ).fetchall()

        output = []
        for row in rows:
            m = self._row_to_obj(row)
            fs, bs, aw = id_to_score[m.id]
            output.append({
                "memory":       m,
                "final_score":  round(fs, 4),
                "bm25_score":   round(bs, 4),
                "amp_weight":   round(aw, 4),
                "shared_from":  row["agent_id"] if row["agent_id"] != self.agent_id else None,
            })

        output.sort(key=lambda x: x["final_score"], reverse=True)
        return output

    def list_all(self, limit: int = 100) -> List[MemoryObject]:
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY importance DESC, created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_obj(r) for r in rows]

    # ── Sync log ──────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, memory_id: str, agent_id: str) -> None:
        self._conn.execute(
            "INSERT INTO sync_log (event_type, memory_id, agent_id, ts) VALUES (?,?,?,?)",
            (event_type, memory_id, agent_id, _dt_to_str(datetime.now(timezone.utc)))
        )
        self._conn.commit()

    def get_sync_log(self, since: Optional[str] = None, agent_id: Optional[str] = None) -> List[Dict]:
        clauses, params = [], []
        if since:
            clauses.append("ts > ?")
            params.append(since)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM sync_log {where} ORDER BY ts DESC LIMIT 200",
            params
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = {}
        for row in self._conn.execute(
            "SELECT memory_type, COUNT(*) as c FROM memories GROUP BY memory_type"
        ).fetchall():
            by_type[row["memory_type"]] = row["c"]

        # Agent breakdown
        by_agent = {}
        for row in self._conn.execute(
            "SELECT agent_id, COUNT(*) as c FROM memories GROUP BY agent_id"
        ).fetchall():
            by_agent[row["agent_id"] or "unknown"] = row["c"]

        avg_imp = self._conn.execute(
            "SELECT AVG(importance) FROM memories"
        ).fetchone()[0] or 0.0

        return {
            "total":        total,
            "by_type":      by_type,
            "by_agent":     by_agent,
            "avg_importance": round(avg_imp, 3),
            "db_path":      self.db_path,
            "bm25_indexed": len(self._retriever),
        }

    def prune(self, min_weight: float = 0.05) -> int:
        all_ids = [r["id"] for r in self._conn.execute("SELECT id FROM memories").fetchall()]
        to_delete = []
        from ..memory_object import MemoryObject as MO
        now = datetime.now(timezone.utc)
        for mid in all_ids:
            row = self._conn.execute("SELECT importance, decay_rate, permanence, created_at FROM memories WHERE id=?", (mid,)).fetchone()
            if not row:
                continue
            import math
            delta = (now - _str_to_dt(row["created_at"])).total_seconds() / 86400
            w = row["importance"] * math.exp(-row["decay_rate"] * delta) + row["permanence"]
            if w < min_weight:
                to_delete.append(mid)
        for mid in to_delete:
            self.delete(mid)
        return len(to_delete)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _scope_filter(self, include_shared: bool) -> Tuple[str, List]:
        """
        Build SQL WHERE clause for scope filtering.

        Own memories (private/session/user) + shared memories
        (other agents' user/public scope for same user_id).
        """
        if not include_shared or not self.user_id:
            # Only own agent's memories
            return "agent_id = ?", [self.agent_id]

        # Own + shared from same user
        clauses = [
            "agent_id = ?",                                   # own memories
            "(user_id = ? AND scope LIKE '%\"user\"%')",      # user-scoped from others
            "scope LIKE '%\"public\"%'",                      # public memories
        ]
        if self.session_id:
            clauses.append(
                "(session_id = ? AND scope LIKE '%\"session\"%')"
            )

        params = [self.agent_id, self.user_id]
        if self.session_id:
            params.append(self.session_id)

        return f"({' OR '.join(clauses)})", params

    def _row_to_obj(self, row: sqlite3.Row) -> MemoryObject:
        trust = None
        if row["trust"]:
            t = json.loads(row["trust"])
            trust = MemoryTrust(
                agent_id    = t["agent_id"],
                model       = t["model"],
                confidence  = t["confidence"],
                source      = SourceType(t.get("source", "inference")),
                verified_by = t.get("verified_by", []),
            )

        scope = None
        if row["scope"]:
            s = json.loads(row["scope"])
            scope = MemoryScope(
                type       = ScopeType(s["type"]),
                agent_id   = s.get("agent_id"),
                session_id = s.get("session_id"),
                user_id    = s.get("user_id"),
            )

        relations = [
            MemoryRelation(
                target_id     = r["target_id"],
                relation_type = RelationType(r["relation_type"]),
                strength      = r.get("strength", 1.0),
            )
            for r in json.loads(row["relations"])
        ]

        return MemoryObject(
            id          = row["id"],
            content     = row["content"],
            embedding   = _unpack_embedding(row["embedding"]),
            structured  = json.loads(row["structured"]),
            memory_type = MemoryType(row["memory_type"]),
            tags        = json.loads(row["tags"]),
            relations   = relations,
            created_at  = _str_to_dt(row["created_at"]),
            accessed_at = _str_to_dt(row["accessed_at"]),
            importance  = row["importance"],
            decay_rate  = row["decay_rate"],
            permanence  = row["permanence"],
            trust       = trust,
            scope       = scope,
        )
