# Changelog

All notable changes to AMP are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [0.3.0] — 2025-03-25

### Added
- **LSA Embedding Engine** — Latent Semantic Analysis (TF-IDF + Truncated SVD via scipy). 256-dim vectors. Runs 100% offline, zero API calls. Semantic hit rate 7/7 = 100% on benchmark.
- **`PgVectorBackend`** — PostgreSQL + pgvector backend with three-signal RRF fusion search: vector cosine similarity + BM25 + AMP weight. Falls back to SQLite for development.
- **`EmbeddingEngine` facade** — Auto-selects LSA (offline) or OpenAI (`text-embedding-3-small`). Pluggable: bring your own embedding function.
- **RRF fusion** (Reciprocal Rank Fusion, Cormack et al. 2009) — combines all three search signals without calibration.
- **`MemoryStore` high-level API** — single entry point: `store.add()`, `store.search()`, `store.forget()`, `store.sync_from()`, `store.export_snapshot()`.
- **PostgreSQL DDL** with pgvector IVFFlat index, GIN full-text index, `amp_weight()` function, `amp_rrf_search()` stored procedure.
- **Docker + docker-compose** — production-ready multi-stage build. `docker run amp-memory` works in one command.
- **`@amp-protocol/sdk`** — TypeScript/JavaScript SDK. Full type safety. Works in Node.js, Deno, Bun, browsers.
- **CI/CD pipeline** — GitHub Actions: test (Python 3.9–3.12 × 3 OS), lint, type check, security audit, PyPI publish (OIDC), npm publish, GitHub Release.

### Changed
- `MCP server` upgraded to v0.2.0: 7 tools (added `amp_sync`, `amp_export`), SQLite persistence.
- Search results now include `rrf_score`, `vec_score`, `bm25_score`, `amp_weight` breakdown.

---

## [0.2.0] — 2025-03-24

### Added
- **SQLite persistent backend** — WAL mode, multi-agent scope filtering, BM25 warm-start on restart.
- **BM25 engine** — custom implementation with unigrams + bigrams, incremental add/remove.
- **Hybrid retriever** — `0.5×BM25 + 0.35×AMP_weight + 0.15×tag_boost`.
- **Cross-agent sync protocol** — `pull_from_agent()`, `export_snapshot()`, `import_snapshot()`, `who_knows_what()`.
- **HTTP REST server** — stdlib-only, `POST/GET/DELETE /v1/memories`, `/v1/sync/*`.
- **MCP server v0.1** — 6 tools, JSON-RPC 2.0 over stdio.

### Fixed
- SQLite `_log_event` uncommitted transaction causing `database is locked` errors.
- BM25 index not syncing cross-agent writes from shared DB file.

---

## [0.1.0] — 2025-03-23

### Added
- **`MemoryObject`** — core data model with 5 dimensions: content, relations, time, trust, scope.
- **Decay function** `weight(t) = importance × e^(−λ×t) + permanence`.
- **5 memory types** with preset λ and permanence values.
- **Graph relations**: `derived_from`, `contradicts`, `supports`, `updates`, `related_to`.
- **Conflict resolution** — source priority × confidence × weight scoring.
- **`MemoryStore`** — in-memory reference implementation with search and prune.
- **JSON Schema** `schema_v0.1.json` — canonical wire format specification.
- **Serialization** — `to_json()`, `from_json()`, full round-trip fidelity.

---

[0.3.0]: https://github.com/amp-protocol/amp-python/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/amp-protocol/amp-python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/amp-protocol/amp-python/releases/tag/v0.1.0
