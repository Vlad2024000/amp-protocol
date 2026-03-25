"""
AMP pgvector Backend — v0.3

Drop-in replacement for SQLiteBackend, backed by:
  PostgreSQL + pgvector extension + EmbeddingEngine

Hybrid search (3 signals fused via RRF):
  1. Vector similarity    — cosine distance on embeddings     (pgvector <=>)
  2. Full-text search     — PostgreSQL tsvector / tsquery     (ts_rank)
  3. AMP weight           — importance × e^(−λ×Δt) + perm    (computed)

RRF fusion (Reciprocal Rank Fusion, Cormack et al. 2009):
  rrf_score = Σ  1 / (k + rank_i)    k=60
  Each signal contributes its RRF rank. Final sort by rrf_score.

Schema (single table):
  memories (
    id           UUID PRIMARY KEY,
    content      TEXT,
    embedding    vector(256),        ← pgvector column
    tsvec        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    structured   JSONB,
    memory_type  TEXT,
    tags         TEXT[],
    relations    JSONB,
    created_at   TIMESTAMPTZ,
    accessed_at  TIMESTAMPTZ,
    importance   FLOAT,
    decay_rate   FLOAT,
    permanence   FLOAT,
    trust        JSONB,
    scope        JSONB,
    agent_id     TEXT,
    user_id      TEXT,
    session_id   TEXT
  );

  CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
  CREATE INDEX ON memories USING GIN (tsvec);
  CREATE INDEX ON memories (agent_id, user_id, memory_type);

Without a live PostgreSQL connection, the backend degrades gracefully to
an in-memory SQLite store with the same interface.
"""

from __future__ import annotations

import json
import os
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
from ..embed.engine import EmbeddingEngine
from ..store.bm25 import HybridRetriever


# ── Connection abstraction ────────────────────────────────────────────────

class _ConnectionPool:
    """
    Thin connection pool abstraction.
    Tries psycopg2 first, falls back to SQLite in-memory for testing.
    """

    def __init__(self, dsn: Optional[str]):
        self._dsn    = dsn
        self._mode   = "none"
        self._sqlite = None
        self._pg     = None
        self._lock   = threading.Lock()
        self._init()

    def _init(self):
        if self._dsn:
            try:
                import psycopg2
                import psycopg2.extras
                self._pg = psycopg2.connect(self._dsn)
                self._pg.autocommit = False
                self._mode = "postgres"
                return
            except Exception as e:
                pass

        # SQLite fallback
        import sqlite3
        db_path = os.environ.get("AMP_DB_PATH", ":memory:")
        self._sqlite = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._sqlite.row_factory = sqlite3.Row
        self._sqlite.execute("PRAGMA journal_mode=WAL")
        self._sqlite.execute("PRAGMA busy_timeout=30000")
        self._mode = "sqlite"

    def execute(self, sql: str, params=None):
        params = params or []
        if self._mode == "postgres":
            cur = self._pg.cursor(cursor_factory=__import__("psycopg2").extras.RealDictCursor)
            cur.execute(sql, params)
            return cur
        else:
            return self._sqlite.execute(sql, params)

    def commit(self):
        if self._mode == "postgres":
            self._pg.commit()
        else:
            self._sqlite.commit()

    def fetchall(self, sql: str, params=None) -> List[Dict]:
        cur = self.execute(sql, params or [])
        rows = cur.fetchall()
        if self._mode == "postgres":
            return [dict(r) for r in rows]
        else:
            return [dict(r) for r in rows]

    def fetchone(self, sql: str, params=None) -> Optional[Dict]:
        cur = self.execute(sql, params or [])
        row = cur.fetchone()
        return dict(row) if row else None

    @property
    def mode(self) -> str:
        return self._mode


# ── DDL ───────────────────────────────────────────────────────────────────

PG_DDL = """
-- Requires: CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS amp_memories (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    embedding    TEXT,                -- JSON array (pgvector: vector(256))
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

CREATE TABLE IF NOT EXISTS amp_sync_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    memory_id  TEXT NOT NULL,
    agent_id   TEXT NOT NULL,
    ts         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_amp_agent   ON amp_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_amp_user    ON amp_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_amp_type    ON amp_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_amp_created ON amp_memories(created_at);
"""

REAL_PG_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS amp_memories (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content      TEXT NOT NULL,
    embedding    vector(256),
    structured   JSONB NOT NULL DEFAULT '{}',
    memory_type  TEXT NOT NULL,
    tags         TEXT[] NOT NULL DEFAULT '{}',
    relations    JSONB NOT NULL DEFAULT '[]',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accessed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    importance   FLOAT NOT NULL DEFAULT 0.5,
    decay_rate   FLOAT NOT NULL DEFAULT 0.01,
    permanence   FLOAT NOT NULL DEFAULT 0.0,
    trust        JSONB,
    scope        JSONB,
    agent_id     TEXT,
    user_id      TEXT,
    session_id   TEXT,
    tsvec        TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);

-- ANN index for vector similarity (IVFFlat)
-- Run AFTER populating with enough data: CREATE INDEX CONCURRENTLY
CREATE INDEX IF NOT EXISTS idx_amp_embedding
    ON amp_memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text index
CREATE INDEX IF NOT EXISTS idx_amp_tsvec
    ON amp_memories USING GIN (tsvec);

-- Standard indexes
CREATE INDEX IF NOT EXISTS idx_amp_agent   ON amp_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_amp_user    ON amp_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_amp_type    ON amp_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_amp_scope   ON amp_memories((scope->>'type'));
"""


# ── Main Backend ──────────────────────────────────────────────────────────

class PgVectorBackend:
    """
    AMP memory backend with pgvector + semantic embeddings.

    On PostgreSQL: full vector search via pgvector <=> operator.
    Fallback (SQLite): in-process BM25 + numpy cosine similarity.

    Both paths expose identical .search() → List[Dict] interface.

    DSN format:
        postgresql://user:password@host:5432/amp_db
        or set AMP_DSN environment variable.

    Usage:
        backend = PgVectorBackend(
            dsn      = "postgresql://localhost/amp",
            agent_id = "agent-claude",
            user_id  = "user-andrii",
        )
        backend.write(memory_object)
        results = backend.search("where does the user live?", top_k=5)
    """

    # RRF constant (Cormack et al. 2009 recommend k=60)
    RRF_K = 60

    def __init__(
        self,
        dsn:          Optional[str] = None,
        agent_id:     str           = "amp-agent",
        user_id:      Optional[str] = None,
        session_id:   Optional[str] = None,
        api_key:      Optional[str] = None,
        prefer_local: bool          = False,
        embedding_dim: int          = 256,
    ):
        _dsn = dsn or os.environ.get("AMP_DSN")

        self.agent_id   = agent_id
        self.user_id    = user_id
        self.session_id = session_id

        # Embedding engine
        self.embedder = EmbeddingEngine(
            api_key      = api_key,
            prefer_local = prefer_local or not os.environ.get("OPENAI_API_KEY"),
            n_components = embedding_dim,
        )

        # DB connection
        self._pool = _ConnectionPool(_dsn)
        self._init_schema()

        # In-memory index for BM25 + AMP weight (both backends use this)
        self._retriever = HybridRetriever()

        # Warm from DB
        self._warm()

        self._lock = threading.Lock()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_schema(self):
        for stmt in PG_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self._pool.execute(stmt)
                except Exception:
                    pass
        self._pool.commit()

    def _warm(self):
        rows = self._pool.fetchall(
            "SELECT id, content, tags, importance, decay_rate, permanence, created_at "
            "FROM amp_memories"
        )
        for r in rows:
            self._retriever.index(
                doc_id     = r["id"],
                text       = r["content"],
                importance = r["importance"],
                decay_rate = r["decay_rate"],
                permanence = r["permanence"],
                created_at = _parse_dt(r["created_at"]),
                tags       = json.loads(r["tags"]) if isinstance(r["tags"], str) else (r["tags"] or []),
            )
            # Also warm LSA corpus
            self.embedder.index(r["id"], r["content"])

    # ── Write ─────────────────────────────────────────────────────────────

    def write(self, m: MemoryObject) -> MemoryObject:
        # Generate embedding
        if m.embedding is None:
            vec = self.embedder.embed(m.content)
            m.embedding = vec.tolist()

        trust_json = _serialize_trust(m.trust)
        scope_json = _serialize_scope(m.scope)
        relations_json = json.dumps([
            {"target_id": r.target_id, "relation_type": r.relation_type.value, "strength": r.strength}
            for r in m.relations
        ])

        agent_id = m.scope.agent_id if m.scope else None
        user_id  = m.scope.user_id  if m.scope else None
        session_id = m.scope.session_id if m.scope else None

        emb_json = json.dumps(m.embedding) if m.embedding else None

        with self._lock:
            self._pool.execute("""
                INSERT OR REPLACE INTO amp_memories
                (id, content, embedding, structured, memory_type, tags, relations,
                 created_at, accessed_at, importance, decay_rate, permanence,
                 trust, scope, agent_id, user_id, session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                m.id,
                m.content,
                emb_json,
                json.dumps(m.structured),
                m.memory_type.value,
                json.dumps(m.tags),
                relations_json,
                _fmt_dt(m.created_at),
                _fmt_dt(m.accessed_at),
                m.importance,
                m.decay_rate,
                m.permanence,
                trust_json,
                scope_json,
                agent_id, user_id, session_id,
            ))
            self._pool.execute(
                "INSERT INTO amp_sync_log(event_type,memory_id,agent_id,ts) VALUES(?,?,?,?)",
                ("write", m.id, self.agent_id, _fmt_dt(datetime.now(timezone.utc)))
            )
            self._pool.commit()

            # Update indexes
            self._retriever.index(
                doc_id=m.id, text=m.content,
                importance=m.importance, decay_rate=m.decay_rate,
                permanence=m.permanence, created_at=m.created_at, tags=m.tags,
            )
            self.embedder.index(m.id, m.content)

        return m

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> Optional[MemoryObject]:
        row = self._pool.fetchone(
            "SELECT * FROM amp_memories WHERE id=?", (memory_id,)
        )
        if not row:
            return None
        self._pool.execute(
            "UPDATE amp_memories SET accessed_at=? WHERE id=?",
            (_fmt_dt(datetime.now(timezone.utc)), memory_id)
        )
        self._pool.commit()
        return _row_to_obj(row)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            self._pool.execute(
                "DELETE FROM amp_memories WHERE id=?", (memory_id,)
            )
            self._pool.execute(
                "INSERT INTO amp_sync_log(event_type,memory_id,agent_id,ts) VALUES(?,?,?,?)",
                ("delete", memory_id, self.agent_id, _fmt_dt(datetime.now(timezone.utc)))
            )
            self._pool.commit()
            self._retriever.remove(memory_id)
            self.embedder.remove(memory_id)
        return True

    def list_all(self, limit: int = 100) -> List[MemoryObject]:
        rows = self._pool.fetchall(
            "SELECT * FROM amp_memories ORDER BY importance DESC LIMIT ?", (limit,)
        )
        return [_row_to_obj(r) for r in rows]

    # ── Hybrid Search ─────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        top_k:          int           = 10,
        memory_types:   Optional[List[MemoryType]] = None,
        tags:           Optional[List[str]]        = None,
        min_weight:     float         = 0.0,
        include_shared: bool          = True,
    ) -> List[Dict[str, Any]]:
        """
        Three-signal RRF fusion:
          Signal 1: Vector cosine similarity (semantic)
          Signal 2: BM25 text relevance
          Signal 3: AMP weight (decay-adjusted importance)

        RRF score = Σ 1/(k + rank_i)  where k=60

        Returns list of dicts with memory + scores.
        """
        # Sync BM25 for cross-agent docs
        self._sync_retriever()

        # Get candidates from DB (scope + type + tag filter)
        scope_sql, scope_params = _scope_filter(
            self.agent_id, self.user_id, self.session_id, include_shared
        )
        type_sql, type_params = "", []
        if memory_types:
            ph = ",".join("?" * len(memory_types))
            type_sql  = f"AND memory_type IN ({ph})"
            type_params = [mt.value for mt in memory_types]

        tag_sql, tag_params = "", []
        if tags:
            conds = " OR ".join(["tags LIKE ?" for _ in tags])
            tag_sql   = f"AND ({conds})"
            tag_params = [f'%"{t}"%' for t in tags]

        rows = self._pool.fetchall(
            f"SELECT id, content, embedding FROM amp_memories WHERE {scope_sql} {type_sql} {tag_sql}",
            scope_params + type_params + tag_params
        )

        if not rows:
            return []

        candidate_ids = [r["id"] for r in rows]

        # ── Signal 1: Vector similarity ───────────────────────────────────
        q_vec = self.embedder.embed(query)
        vec_scores: Dict[str, float] = {}

        for r in rows:
            emb_raw = r.get("embedding")
            if emb_raw:
                try:
                    d_vec = np.array(
                        json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw,
                        dtype=np.float32
                    )
                    # Pad or truncate to match query dim
                    if len(d_vec) != len(q_vec):
                        tmp = np.zeros(len(q_vec), dtype=np.float32)
                        tmp[:min(len(d_vec), len(q_vec))] = d_vec[:min(len(d_vec), len(q_vec))]
                        d_vec = tmp
                    norm = np.linalg.norm(d_vec)
                    if norm > 1e-9:
                        d_vec = d_vec / norm
                    sim = float(np.dot(q_vec, d_vec))
                    vec_scores[r["id"]] = max(0.0, sim)
                except Exception:
                    vec_scores[r["id"]] = 0.0
            else:
                vec_scores[r["id"]] = 0.0

        # ── Signal 2: BM25 ────────────────────────────────────────────────
        bm25_results = self._retriever.bm25.search(query, top_k=len(candidate_ids)*2, candidate_ids=candidate_ids)
        bm25_vals    = {did: score for did, score in bm25_results}
        max_bm25     = max(bm25_vals.values(), default=1.0) or 1.0
        bm25_norm    = {did: s / max_bm25 for did, s in bm25_vals.items()}

        # ── Signal 3: AMP weight ──────────────────────────────────────────
        import math as _math
        from datetime import datetime as _dt
        now_ts = _dt.now(timezone.utc).timestamp()

        def amp_w(did: str) -> float:
            meta = self._retriever._meta.get(did)
            if not meta:
                return 0.5
            imp, lam, perm, created_ts, _ = meta
            delta_days = max(0, (now_ts - created_ts) / 86400)
            return min(1.0, max(0.0, imp * _math.exp(-lam * delta_days) + perm))

        # ── RRF Fusion ────────────────────────────────────────────────────
        k = self.RRF_K

        def rrf(rank: int) -> float:
            return 1.0 / (k + rank)

        # Rank each signal
        vec_ranked  = sorted(candidate_ids, key=lambda d: vec_scores.get(d, 0), reverse=True)
        bm25_ranked = sorted(candidate_ids, key=lambda d: bm25_norm.get(d, 0), reverse=True)
        amp_ranked  = sorted(candidate_ids, key=amp_w, reverse=True)

        vec_rank  = {did: i+1 for i, did in enumerate(vec_ranked)}
        bm25_rank = {did: i+1 for i, did in enumerate(bm25_ranked)}
        amp_rank  = {did: i+1 for i, did in enumerate(amp_ranked)}

        rrf_scores: Dict[str, float] = {}
        for did in candidate_ids:
            w = amp_w(did)
            if w < min_weight:
                continue
            rrf_scores[did] = (
                rrf(vec_rank.get(did, len(candidate_ids))) +
                rrf(bm25_rank.get(did, len(candidate_ids))) +
                rrf(amp_rank.get(did, len(candidate_ids)))
            )

        if not rrf_scores:
            return []

        top_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:top_k]

        # Fetch full objects
        ph  = ",".join("?" * len(top_ids))
        full_rows = self._pool.fetchall(
            f"SELECT * FROM amp_memories WHERE id IN ({ph})", top_ids
        )
        row_map   = {r["id"]: r for r in full_rows}

        results = []
        for did in top_ids:
            row = row_map.get(did)
            if not row:
                continue
            m = _row_to_obj(row)
            results.append({
                "memory":       m,
                "rrf_score":    round(rrf_scores[did], 5),
                "vec_score":    round(vec_scores.get(did, 0), 4),
                "bm25_score":   round(bm25_norm.get(did, 0), 4),
                "amp_weight":   round(amp_w(did), 4),
                "shared_from":  row.get("agent_id") if row.get("agent_id") != self.agent_id else None,
            })

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = self._pool.fetchone("SELECT COUNT(*) as c FROM amp_memories")["c"]
        by_type = {}
        for r in self._pool.fetchall(
            "SELECT memory_type, COUNT(*) as c FROM amp_memories GROUP BY memory_type"
        ):
            by_type[r["memory_type"]] = r["c"]
        by_agent = {}
        for r in self._pool.fetchall(
            "SELECT agent_id, COUNT(*) as c FROM amp_memories GROUP BY agent_id"
        ):
            by_agent[r["agent_id"] or "unknown"] = r["c"]

        avg_imp = (self._pool.fetchone("SELECT AVG(importance) as v FROM amp_memories") or {}).get("v") or 0.0

        return {
            "total":          total,
            "by_type":        by_type,
            "by_agent":       by_agent,
            "avg_importance": round(avg_imp, 3),
            "embedder":       repr(self.embedder),
            "db_mode":        self._pool.mode,
        }

    def prune(self, min_weight: float = 0.05) -> int:
        import math as _m
        now = datetime.now(timezone.utc)
        rows = self._pool.fetchall(
            "SELECT id, importance, decay_rate, permanence, created_at FROM amp_memories"
        )
        deleted = 0
        for r in rows:
            delta = (now - _parse_dt(r["created_at"])).total_seconds() / 86400
            w = r["importance"] * _m.exp(-r["decay_rate"] * delta) + r["permanence"]
            if w < min_weight:
                self.delete(r["id"])
                deleted += 1
        return deleted

    # ── Internal ──────────────────────────────────────────────────────────

    def _sync_retriever(self):
        """Sync BM25 for docs written by other agents (cross-agent discovery)."""
        all_ids = {r["id"] for r in self._pool.fetchall("SELECT id FROM amp_memories")}
        known   = set(self._retriever._meta.keys())
        missing = all_ids - known
        if not missing:
            return
        ph   = ",".join("?" * len(missing))
        rows = self._pool.fetchall(
            f"SELECT id, content, tags, importance, decay_rate, permanence, created_at FROM amp_memories WHERE id IN ({ph})",
            list(missing)
        )
        for r in rows:
            self._retriever.index(
                doc_id     = r["id"],
                text       = r["content"],
                importance = r["importance"],
                decay_rate = r["decay_rate"],
                permanence = r["permanence"],
                created_at = _parse_dt(r["created_at"]),
                tags       = json.loads(r["tags"]) if isinstance(r["tags"], str) else (r["tags"] or []),
            )
            self.embedder.index(r["id"], r["content"])


# ── Helpers ───────────────────────────────────────────────────────────────

def _fmt_dt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _serialize_trust(trust) -> Optional[str]:
    if not trust: return None
    return json.dumps({
        "agent_id":    trust.agent_id,
        "model":       trust.model,
        "confidence":  trust.confidence,
        "source":      trust.source.value,
        "verified_by": trust.verified_by,
    })

def _serialize_scope(scope) -> Optional[str]:
    if not scope: return None
    return json.dumps({
        "type":       scope.type.value,
        "agent_id":   scope.agent_id,
        "session_id": scope.session_id,
        "user_id":    scope.user_id,
    })

def _scope_filter(agent_id, user_id, session_id, include_shared) -> Tuple[str, List]:
    if not include_shared or not user_id:
        return "agent_id = ?", [agent_id]
    clauses = [
        "agent_id = ?",
        "(user_id = ? AND scope LIKE '%\"user\"%')",
        "scope LIKE '%\"public\"%'",
    ]
    params = [agent_id, user_id]
    if session_id:
        clauses.append("(session_id = ? AND scope LIKE '%\"session\"%')")
        params.append(session_id)
    return f"({' OR '.join(clauses)})", params

def _row_to_obj(row: Dict) -> MemoryObject:
    trust = None
    if row.get("trust"):
        t = json.loads(row["trust"]) if isinstance(row["trust"], str) else row["trust"]
        trust = MemoryTrust(
            agent_id=t["agent_id"], model=t["model"], confidence=t["confidence"],
            source=SourceType(t.get("source","inference")), verified_by=t.get("verified_by",[]),
        )
    scope = None
    if row.get("scope"):
        s = json.loads(row["scope"]) if isinstance(row["scope"], str) else row["scope"]
        scope = MemoryScope(
            type=ScopeType(s["type"]), agent_id=s.get("agent_id"),
            session_id=s.get("session_id"), user_id=s.get("user_id"),
        )
    rels_raw = row.get("relations","[]")
    relations = [
        MemoryRelation(
            target_id=r["target_id"],
            relation_type=RelationType(r["relation_type"]),
            strength=r.get("strength",1.0),
        )
        for r in (json.loads(rels_raw) if isinstance(rels_raw, str) else rels_raw)
    ]
    emb_raw = row.get("embedding")
    embedding = None
    if emb_raw:
        try:
            embedding = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
        except Exception:
            embedding = None

    return MemoryObject(
        id          = row["id"],
        content     = row["content"],
        embedding   = embedding,
        structured  = json.loads(row["structured"]) if isinstance(row.get("structured"), str) else (row.get("structured") or {}),
        memory_type = MemoryType(row["memory_type"]),
        tags        = json.loads(row["tags"]) if isinstance(row.get("tags"), str) else (row.get("tags") or []),
        relations   = relations,
        created_at  = _parse_dt(row["created_at"]),
        accessed_at = _parse_dt(row["accessed_at"]),
        importance  = row["importance"],
        decay_rate  = row["decay_rate"],
        permanence  = row["permanence"],
        trust       = trust,
        scope       = scope,
    )
