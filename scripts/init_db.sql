-- AMP Database Initialization
-- Runs once on first postgres container start

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- for fuzzy text search

-- ── Main memories table ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS amp_memories (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content      TEXT NOT NULL,
    embedding    vector(256),          -- LSA/OpenAI embedding (adjustable dim)
    structured   JSONB NOT NULL DEFAULT '{}',
    memory_type  TEXT NOT NULL CHECK (memory_type IN ('fact','event','skill','preference','context')),
    tags         TEXT[] NOT NULL DEFAULT '{}',
    relations    JSONB NOT NULL DEFAULT '[]',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accessed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    importance   FLOAT NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    decay_rate   FLOAT NOT NULL DEFAULT 0.01 CHECK (decay_rate >= 0),
    permanence   FLOAT NOT NULL DEFAULT 0.0  CHECK (permanence BETWEEN 0 AND 1),
    trust        JSONB,
    scope        JSONB,
    agent_id     TEXT,
    user_id      TEXT,
    session_id   TEXT,

    -- Generated column: AMP weight at creation time (decays in application layer)
    tsvec        TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);

-- ── Indexes ───────────────────────────────────────────────────────────────

-- Vector similarity (IVFFlat — build AFTER initial data load for best quality)
-- lists = sqrt(N) roughly. For 10k docs: 100, for 1M: ~1000
-- CREATE INDEX CONCURRENTLY ON amp_memories
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_amp_tsvec
    ON amp_memories USING GIN (tsvec);

-- Structured queries
CREATE INDEX IF NOT EXISTS idx_amp_agent   ON amp_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_amp_user    ON amp_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_amp_session ON amp_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_amp_type    ON amp_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_amp_created ON amp_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_amp_importance ON amp_memories(importance DESC);

-- Scope filtering (common in cross-agent queries)
CREATE INDEX IF NOT EXISTS idx_amp_scope_type
    ON amp_memories ((scope->>'type'));

-- Tags (GIN for array containment)
CREATE INDEX IF NOT EXISTS idx_amp_tags
    ON amp_memories USING GIN (tags);

-- ── Sync log ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS amp_sync_log (
    id         BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    memory_id  UUID NOT NULL,
    agent_id   TEXT NOT NULL,
    ts         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload    JSONB
);

CREATE INDEX IF NOT EXISTS idx_sync_agent ON amp_sync_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_sync_ts    ON amp_sync_log(ts DESC);

-- ── Helper functions ──────────────────────────────────────────────────────

-- AMP decay weight (mirrors Python formula)
CREATE OR REPLACE FUNCTION amp_weight(
    importance  FLOAT,
    decay_rate  FLOAT,
    permanence  FLOAT,
    created_at  TIMESTAMPTZ
) RETURNS FLOAT AS $$
    SELECT LEAST(1.0, GREATEST(0.0,
        importance * EXP(-decay_rate * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0)
        + permanence
    ))
$$ LANGUAGE SQL IMMUTABLE;

-- Hybrid RRF search function (callable from application or psql)
-- Usage: SELECT * FROM amp_rrf_search('alice location', 'user-alice', 10);
CREATE OR REPLACE FUNCTION amp_rrf_search(
    query_text  TEXT,
    p_user_id   TEXT,
    p_limit     INT DEFAULT 10,
    rrf_k       INT DEFAULT 60
)
RETURNS TABLE (
    id          UUID,
    content     TEXT,
    memory_type TEXT,
    importance  FLOAT,
    weight      FLOAT,
    tags        TEXT[],
    agent_id    TEXT,
    rrf_score   FLOAT,
    ts_score    FLOAT
) AS $$
WITH
-- Full-text signal
fts AS (
    SELECT m.id,
           ts_rank(m.tsvec, websearch_to_tsquery('english', query_text)) AS score,
           ROW_NUMBER() OVER (ORDER BY ts_rank(m.tsvec, websearch_to_tsquery('english', query_text)) DESC) AS rank
    FROM amp_memories m
    WHERE (m.user_id = p_user_id OR scope->>'type' = 'public')
      AND m.tsvec @@ websearch_to_tsquery('english', query_text)
),
-- Weight signal (recency × importance)
wts AS (
    SELECT m.id,
           amp_weight(m.importance, m.decay_rate, m.permanence, m.created_at) AS w,
           ROW_NUMBER() OVER (ORDER BY amp_weight(m.importance, m.decay_rate, m.permanence, m.created_at) DESC) AS rank
    FROM amp_memories m
    WHERE m.user_id = p_user_id OR scope->>'type' = 'public'
),
-- RRF fusion
rrf AS (
    SELECT COALESCE(fts.id, wts.id) AS id,
           (1.0 / (rrf_k + COALESCE(fts.rank, 999)))::FLOAT +
           (1.0 / (rrf_k + COALESCE(wts.rank, 999)))::FLOAT AS rrf_score,
           COALESCE(fts.score, 0) AS ts_score
    FROM fts FULL OUTER JOIN wts ON fts.id = wts.id
)
SELECT m.id, m.content, m.memory_type, m.importance,
       amp_weight(m.importance, m.decay_rate, m.permanence, m.created_at),
       m.tags, m.agent_id,
       r.rrf_score, r.ts_score
FROM rrf r JOIN amp_memories m ON r.id = m.id
ORDER BY r.rrf_score DESC
LIMIT p_limit;
$$ LANGUAGE SQL;

-- ── Maintenance ───────────────────────────────────────────────────────────

-- Prune expired context memories (cron: every hour)
-- DELETE FROM amp_memories
-- WHERE memory_type = 'context'
--   AND amp_weight(importance, decay_rate, permanence, created_at) < 0.01;

-- Auto-create IVFFlat index when enough data exists
-- (run once after initial bulk load, not on every start)
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_amp_embedding
--     ON amp_memories USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

COMMENT ON TABLE  amp_memories IS 'AMP v0.3 — Agent Memory Protocol memory store';
COMMENT ON COLUMN amp_memories.embedding    IS 'L2-normalized float32 vector. Dim=256 (LSA) or 1536 (OpenAI).';
COMMENT ON COLUMN amp_memories.decay_rate   IS 'λ in weight(t) = I·e^(−λ·days) + permanence';
COMMENT ON COLUMN amp_memories.permanence   IS 'Minimum weight floor — memory never falls below this.';
COMMENT ON COLUMN amp_memories.tsvec        IS 'Auto-generated for PostgreSQL full-text search.';
