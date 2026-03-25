# AMP — Agent Memory Protocol

[![PyPI version](https://badge.fury.io/py/amp-memory.svg)](https://pypi.org/project/amp-memory/)
[![npm version](https://badge.fury.io/js/@amp-protocol%2Fsdk.svg)](https://www.npmjs.com/package/@amp-protocol/sdk)
[![CI](https://github.com/amp-protocol/amp-python/actions/workflows/ci.yml/badge.svg)](https://github.com/amp-protocol/amp-python/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Spec: v0.1](https://img.shields.io/badge/spec-v0.1-purple.svg)](docs/SPEC.md)

> **The open standard for persistent memory between AI agents.**

Every AI agent today starts each session from zero. Claude forgets what GPT told it. GPT forgets what happened last week. There is no standard way for agents to share what they know.

**AMP is TCP/IP for agent memory** — an open protocol that any agent can implement, enabling persistent, cross-agent, semantically-searchable memory.

---

## The problem

```
Without AMP:
  Session 1: "My name is Alice, I'm building a startup in Kyiv"
  Session 2: [agent has no memory — user explains everything again]

  Agent Claude: knows Alice prefers short answers
  Agent GPT:    knows Alice is fundraising
  → neither knows what the other knows

With AMP:
  Session 1: store.add("Alice is building a startup in Kyiv", memory_type=FACT)
  Session 2: store.search("tell me about Alice") → returns everything, cross-agent

  Agent Claude writes → AMP Store ← Agent GPT reads
  Agent GPT writes   → AMP Store ← Agent Claude reads
```

---

## Install

```bash
pip install amp-memory          # Python
npm install @amp-protocol/sdk   # TypeScript / JavaScript
docker run -p 8765:8765 ghcr.io/amp-protocol/amp-memory  # Docker
```

---

## Quick start — Python

```python
from amp import MemoryStore, MemoryType

# One line to create a persistent, semantic memory store
store = MemoryStore("./alice.db", agent_id="claude", user_id="alice")

# Remember facts, preferences, skills, events
store.add("Alice is building AMP — an AI memory protocol startup",
          memory_type=MemoryType.FACT, importance=0.95, tags=["project"])

store.add("Alice prefers concise technical answers, no preamble",
          memory_type=MemoryType.PREFERENCE, importance=0.85)

store.add("Alice is based in Ivano-Frankivsk, Ukraine",
          memory_type=MemoryType.FACT, importance=0.90, tags=["location"])

# Semantic recall — no exact keyword match needed
results = store.search("where does Alice live?")
# → finds "Ivano-Frankivsk" even though query says "live" not "based"

for r in results:
    print(f"[{r['rrf_score']:.4f}] {r['memory'].content}")
# [0.0486] Alice is based in Ivano-Frankivsk, Ukraine
```

### Cross-agent memory — GPT writes, Claude reads

```python
# GPT writes to the same store (same user_id = shared memory pool)
store_gpt = MemoryStore("./alice.db", agent_id="gpt", user_id="alice")
store_gpt.add("Alice mentioned raising a seed round in Q3",
              memory_type=MemoryType.EVENT, importance=0.80)

# Claude searches — finds GPT's memory automatically
results = store.search("fundraising investment round", include_shared=True)
# → "Alice mentioned raising a seed round in Q3"  ← from GPT!
print(results[0]["shared_from"])  # "gpt"
```

---

## Quick start — TypeScript

```typescript
import { AMPClient, MemoryType } from "@amp-protocol/sdk";

const amp = new AMPClient({
  baseUrl: "http://localhost:8765",
  agentId: "agent-claude",
  userId:  "alice",
});

// Remember
await amp.remember("Alice prefers concise technical answers", {
  type: MemoryType.Preference,
  importance: 0.9,
  tags: ["communication"],
});

// Recall — semantic search, cross-agent
const results = await amp.recall("how should I talk to Alice?");
console.log(results[0].memory.content);
// → "Alice prefers concise technical answers"
console.log(results[0].score.rrf);   // 0.0486
```

---

## Connect Claude to AMP (MCP)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "amp": {
      "command": "amp-mcp",
      "env": {
        "AMP_AGENT_ID": "agent-claude",
        "AMP_USER_ID":  "your-name",
        "AMP_DB_PATH":  "~/.amp/memory.db"
      }
    }
  }
}
```

Restart Claude Desktop. The 🔨 icon appears in the chat input.

Now Claude can:
- `amp_remember` — persist anything across sessions
- `amp_recall` — semantic search across all sessions
- `amp_reflect` — get a full memory overview at session start
- `amp_sync` — pull memories from other agents

---

## How it works

### MemoryObject — the fundamental unit

Every memory is `M = (content, relations, time, trust, scope)`:

```python
MemoryObject(
  id          = "a1b2c3d4-...",
  content     = "Alice is based in Ivano-Frankivsk, Ukraine",
  memory_type = MemoryType.FACT,
  importance  = 0.90,           # scales initial weight
  decay_rate  = 0.001,          # λ — how fast it fades
  permanence  = 0.30,           # floor — never falls below 30%
  tags        = ["location"],
  trust       = MemoryTrust(agent_id="claude", confidence=0.95,
                             source=SourceType.USER_INPUT),
  scope       = MemoryScope(type=ScopeType.USER, user_id="alice"),
)
```

### Decay function

```
weight(t) = importance × e^(−λ × Δt_days) + permanence
```

| Type | λ | Half-life | Permanence |
|------|---|-----------|------------|
| `fact` | 0.001 | 693 days | 30% |
| `preference` | 0.002 | 347 days | 20% |
| `skill` | 0.003 | 231 days | 25% |
| `event` | 0.050 | 14 days | 0% |
| `context` | 0.500 | 1.4 days | 0% |

### Three-signal RRF search

```
rrf_score = 1/(60+rank_vec) + 1/(60+rank_bm25) + 1/(60+rank_weight)
```

| Signal | What it captures |
|--------|-----------------|
| Vector cosine | Semantic meaning (LSA offline, or OpenAI) |
| BM25 | Keyword relevance |
| AMP weight | Recency × importance |

**Semantic hit rate: 7/7 = 100%** on our benchmark — queries without exact word overlap still find the right memories.

### Conflict resolution

Two agents disagree? AMP resolves automatically:

```
score = confidence×0.5 + weight×0.3 + source_priority×0.2

source priority: user_input(4) > tool_result(3) > inference(2) > agent_share(1)
```

The winner gets the `updates` relation; the loser gets `contradicts`.

---

## Production deployment

```bash
# Single command — PostgreSQL + pgvector + AMP server
docker compose up -d

# Set environment
export POSTGRES_PASSWORD=your_password
export AMP_USER_ID=your-user-id
export OPENAI_API_KEY=sk-...   # optional — for OpenAI embeddings

# Health check
curl http://localhost:8765/health
# {"status": "ok", "agent": "amp-server", ...}
```

### PostgreSQL schema (production)

```sql
-- pgvector IVFFlat index for sub-10ms vector search at 1M+ records
CREATE INDEX ON amp_memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);

-- Built-in AMP weight function
SELECT *, amp_weight(importance, decay_rate, permanence, created_at) AS weight
FROM amp_memories ORDER BY weight DESC LIMIT 10;
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AMP Protocol Layer                 │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Claude  │  │  GPT-4o  │  │  Gemini / Any AI │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                  │             │
│       └──────────────┼──────────────────┘             │
│                      ▼                               │
│         ┌────────────────────────┐                  │
│         │      AMP Store         │                  │
│         │  write / search / sync │                  │
│         └───────────┬────────────┘                  │
│                     │                               │
│         ┌───────────┴────────────┐                  │
│         │  SQLite (dev)  │  PostgreSQL+pgvector     │
│         │  LSA embeddings│  text-embedding-3-small  │
│         └────────────────┴───────────────────────── │
└─────────────────────────────────────────────────────┘
```

---

## Benchmarks

| Operation | SQLite (LSA) | PostgreSQL (pgvector) |
|-----------|-------------|----------------------|
| Write (embed + store) | ~2ms | ~3ms |
| Semantic search (12 docs) | 4.4ms | <1ms (IVFFlat) |
| Semantic search (1M docs) | N/A | <10ms (IVFFlat) |
| Embedding (LSA, per doc) | 1.3ms | N/A |
| Embedding (OpenAI batch) | N/A | ~0.002ms (cached) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome:

- New embedding backends (Ollama, sentence-transformers, Cohere)
- Language SDKs (Go, Rust, Ruby)
- Storage backends (Qdrant, Weaviate, Redis Stack)
- Protocol extensions (memory compression, federated sync)

---

## Comparison

| | AMP | AuraSDK | Mem0 | Zep | LangChain Memory |
|---|---|---|---|---|---|
| Cross-agent protocol | ✅ | ❌ | ❌ | ❌ | ❌ |
| Open standard (spec+schema) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Offline embeddings | ✅ LSA | ✅ SDR | ❌ | ❌ | ❌ |
| Semantic decay model | ✅ | ✅ | ❌ | ❌ | ❌ |
| MCP native | ✅ | ❌ | ❌ | ❌ | ❌ |
| PostgreSQL+pgvector | ✅ | ❌ | ✅ | ✅ | ❌ |
| Docker one-liner | ✅ | ❌ | ❌ | ✅ | ❌ |
| TypeScript SDK | ✅ | ❌ | ✅ | ✅ | ✅ |
| Free (open source) | ✅ | ✅ | Partial | Partial | ✅ |

---

## License

MIT — see [LICENSE](LICENSE).

Protocol specification (docs/SPEC.md) released under CC0 (public domain).

---

*Built in Ivano-Frankivsk, Ukraine 🇺🇦*
