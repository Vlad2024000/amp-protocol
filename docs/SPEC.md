# AMP — Agent Memory Protocol Specification
## Version 0.1 — Draft RFC

**Status:** Draft  
**Authors:** AMP Protocol Contributors  
**Repository:** https://github.com/amp-protocol/amp-python  
**Schema:** https://amp-protocol.org/schemas/v0.1/memory-object.json

---

## Abstract

The Agent Memory Protocol (AMP) defines an open standard for persistent, structured memory shared between AI agents. AMP enables any AI agent — regardless of vendor, model, or deployment — to read from and write to a shared memory store, preserving context across sessions and enabling collaboration between agents.

AMP is to agent memory what TCP/IP is to data transmission: a foundational protocol that any implementation can adopt.

---

## 1. Motivation

Every AI agent in production today starts each session from zero. There is no standard way for Claude to remember what GPT told it about a user, no protocol for persisting facts across sessions, and no mechanism for resolving conflicting beliefs between agents.

This creates three concrete problems:

**Problem 1 — Session amnesia.** Users re-explain their context in every conversation. An agent that helped you build a product last week has no memory of it today.

**Problem 2 — Agent silos.** When multiple agents collaborate on a task, each operates on its own context window. There is no shared understanding.

**Problem 3 — No trust model.** When two agents hold contradictory beliefs about the same fact, there is no principled way to resolve the conflict.

AMP solves all three by defining a standard memory object, a wire format, and a protocol for cross-agent access and conflict resolution.

---

## 2. Core Concepts

### 2.1 MemoryObject

The fundamental unit of AMP. Every memory is a `MemoryObject`:

```
M = (content, relations, time, trust, scope)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Globally unique identifier |
| `content` | string | Human-readable memory content |
| `embedding` | float[] \| null | Semantic vector (optional) |
| `structured` | object | Key-value facts extracted from content |
| `memory_type` | enum | Category: `fact` `event` `skill` `preference` `context` |
| `tags` | string[] | Labels for filtering |
| `relations` | Relation[] | Graph edges to other memories |
| `created_at` | ISO 8601 | Creation timestamp with timezone |
| `accessed_at` | ISO 8601 | Last access timestamp |
| `importance` | float [0,1] | Subjective importance set by writing agent |
| `decay_rate` | float ≥ 0 | λ in the decay function |
| `permanence` | float [0,1] | Minimum weight floor |
| `trust` | Trust \| null | Provenance metadata |
| `scope` | Scope \| null | Access control |

### 2.2 Memory Types and Default Decay

Each memory type has a default decay rate (λ) and permanence floor:

| Type | λ | Permanence | Half-life | Use case |
|------|---|------------|-----------|----------|
| `fact` | 0.001 | 0.30 | ~693 days | Names, locations, stable facts |
| `preference` | 0.002 | 0.20 | ~347 days | User preferences, styles |
| `skill` | 0.003 | 0.25 | ~231 days | Procedural knowledge |
| `event` | 0.050 | 0.00 | ~14 days | Specific past events |
| `context` | 0.500 | 0.00 | ~1.4 days | Current session state |

### 2.3 The Decay Function

Memory weight decreases over time following an exponential decay with a floor:

```
weight(t) = importance × e^(−λ × Δt_days) + permanence
```

Where:
- `importance` ∈ [0, 1] — set by the writing agent at creation
- `λ` — decay rate (higher = faster forgetting)
- `Δt_days` — days since `created_at`
- `permanence` ∈ [0, 1] — minimum weight; memory never falls below this

This mirrors the Ebbinghaus forgetting curve (1885), extended with an importance scalar and a permanence floor.

### 2.4 Relations (Memory Graph)

MemoryObjects form a directed graph:

| Relation type | Semantics |
|---------------|-----------|
| `derived_from` | This memory was inferred from another |
| `contradicts` | Conflicts with another memory |
| `supports` | Corroborates another memory |
| `updates` | Supersedes an older memory |
| `related_to` | Loose semantic connection |

### 2.5 Trust

Every memory carries provenance:

```json
{
  "agent_id":    "agent-claude",
  "model":       "claude-sonnet-4-6",
  "confidence":  0.95,
  "source":      "user_input",
  "verified_by": ["agent-gpt-4o"]
}
```

Source priority for conflict resolution: `user_input (4) > tool_result (3) > inference (2) > agent_share (1) > system (0)`

### 2.6 Scope

Access control determines which agents can read or write a memory:

| Scope | Visibility |
|-------|------------|
| `private` | Only the writing agent |
| `session` | All agents in the same `session_id` |
| `user` | All agents with the same `user_id`, across sessions |
| `public` | Any agent |

---

## 3. Wire Format

AMP uses JSON over HTTP (REST) and JSON-RPC 2.0 over stdio (MCP).

### 3.1 MemoryObject JSON Example

```json
{
  "id":          "a1b2c3d4-0000-0000-0000-000000000001",
  "content":     "Alice's name is Alice Chen",
  "embedding":   null,
  "structured":  { "entity": "user", "attribute": "name", "value": "Alice Chen" },
  "memory_type": "fact",
  "tags":        ["identity", "name"],
  "relations":   [],
  "created_at":  "2025-03-25T10:00:00+00:00",
  "accessed_at": "2025-03-25T10:00:00+00:00",
  "importance":  0.95,
  "decay_rate":  0.001,
  "permanence":  0.30,
  "trust": {
    "agent_id":    "agent-claude",
    "model":       "claude-sonnet-4-6",
    "confidence":  0.95,
    "source":      "user_input",
    "verified_by": []
  },
  "scope": {
    "type":       "user",
    "agent_id":   "agent-claude",
    "session_id": null,
    "user_id":    "user-alice"
  }
}
```

### 3.2 REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/memories` | Write a memory |
| `GET` | `/v1/memories/search?q=...` | Semantic search |
| `GET` | `/v1/memories/{id}` | Get by ID |
| `DELETE` | `/v1/memories/{id}` | Delete |
| `GET` | `/v1/memories` | List all |
| `GET` | `/v1/stats` | Store statistics |
| `POST` | `/v1/sync/export` | Export snapshot |
| `POST` | `/v1/sync/import` | Import snapshot |
| `GET` | `/health` | Health check |

### 3.3 MCP Tools

AMP exposes six MCP tools via the stdio JSON-RPC transport:

| Tool | Description |
|------|-------------|
| `amp_remember` | Write a memory |
| `amp_recall` | Semantic search |
| `amp_reflect` | Store overview |
| `amp_forget` | Delete by ID |
| `amp_sync` | Pull from another agent |
| `amp_export` | Export snapshot |
| `amp_relate` | Create graph relation |

---

## 4. Conflict Resolution

When two memories contradict each other (same `memory_type`, same topic, incompatible content), AMP resolves using:

```
score = confidence × 0.5 + weight(now) × 0.3 + source_priority × 0.2
```

The higher-scoring memory wins. The loser receives a `contradicts` relation pointing to the winner.

Implementations MAY override this algorithm; the default is defined here.

---

## 5. Retrieval

AMP RECOMMENDS three-signal RRF fusion for retrieval:

**Signal 1 — Vector similarity** (semantic)  
Cosine similarity between query embedding and stored embeddings.

**Signal 2 — BM25 text relevance**  
Okapi BM25 over tokenized content + tags (k1=1.5, b=0.75).

**Signal 3 — AMP weight**  
Decay-adjusted importance score.

**RRF fusion** (Cormack et al. 2009, k=60):
```
rrf_score = Σ  1 / (60 + rank_i)
```

Implementations MAY use any retrieval algorithm that returns relevant results.

---

## 6. Implementation Requirements

An AMP-compliant implementation MUST:

1. Persist `MemoryObject` with all required fields.
2. Implement the decay function `weight(t) = I × e^(−λ×t) + P`.
3. Enforce scope access control before returning search results.
4. Expose at minimum: write, read-by-id, delete, search.
5. Return results in descending score order.

An AMP-compliant implementation SHOULD:

6. Implement cross-agent access via shared `user_id` scope.
7. Implement conflict resolution per Section 4.
8. Support the MCP tool interface per Section 3.3.
9. Include `trust` provenance on all written memories.

---

## 7. Versioning

This document describes AMP v0.1. The `$id` URI in the JSON Schema serves as the canonical version identifier:

```
https://amp-protocol.org/schemas/v0.1/memory-object.json
```

Breaking changes increment the major version. Additive changes increment the minor version.

---

## 8. License

This specification is released under CC0 1.0 Universal (Public Domain).  
Reference implementations are MIT licensed.

---

## Appendix A — Reference Implementations

| Language | Package | Source |
|----------|---------|--------|
| Python | `pip install amp-memory` | github.com/amp-protocol/amp-python |
| TypeScript | `npm install @amp-protocol/sdk` | github.com/amp-protocol/amp-python/sdk/typescript |

## Appendix B — Prior Art

- Ebbinghaus forgetting curve (1885) — decay model
- Okapi BM25 (Robertson & Zaragoza, 2009) — text ranking
- RRF (Cormack, Clarke & Buettcher, 2009) — rank fusion
- Model Context Protocol (Anthropic, 2024) — agent tool interface
- pgvector (PostgreSQL extension) — vector similarity
