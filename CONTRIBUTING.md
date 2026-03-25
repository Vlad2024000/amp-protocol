# Contributing to AMP

First: thank you. AMP is only as strong as its community.

## What we need most

The highest-impact contributions right now:

| Area | What to build | Difficulty |
|------|---------------|------------|
| **Embedding backends** | Ollama, sentence-transformers, Cohere | Medium |
| **Storage backends** | Qdrant, Weaviate, Redis Stack | Medium |
| **Language SDKs** | Go, Rust, Ruby, Java | Medium–Hard |
| **Protocol extensions** | Memory compression, federated sync | Hard |
| **Benchmarks** | Retrieval quality vs Mem0/Zep/AuraSDK | Easy |
| **Integrations** | LangChain, LlamaIndex, AutoGen | Easy–Medium |

## Getting started

```bash
git clone https://github.com/amp-protocol/amp-python
cd amp-python
pip install -e ".[dev]"
pytest tests/ -v
```

## Code standards

- **Python 3.9+** compatible. No walrus operator (`:=`) without 3.8 fallback.
- **Type hints everywhere.** `mypy amp/` must pass clean.
- **Docstrings** on all public functions, classes, modules.
- **Tests** for every new feature. We target ≥70% coverage.
- **`ruff check amp/ tests/`** must pass before PR.

## Adding an embedding backend

Create `amp/embed/your_backend.py`:

```python
class YourEmbedder:
    """Docstring: what model, what dim, any requirements."""

    @property
    def dim(self) -> int:
        return 384  # your dimension

    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalized float32 vector."""
        ...

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        ...

    def index(self, doc_id: str, text: str) -> None:
        """Optional: for models needing pre-indexing (like LSA)."""
        pass

    def remove(self, doc_id: str) -> None:
        pass
```

Then pass it to `EmbeddingEngine.set_embedder(YourEmbedder())`.

## Adding a storage backend

Implement the interface in `amp/store/your_backend.py`:

```python
class YourBackend:
    agent_id:   str
    user_id:    Optional[str]
    session_id: Optional[str]
    embedder:   EmbeddingEngine

    def write(self, m: MemoryObject) -> MemoryObject: ...
    def get(self, memory_id: str) -> Optional[MemoryObject]: ...
    def delete(self, memory_id: str) -> bool: ...
    def search(self, query: str, **kwargs) -> List[Dict]: ...
    def list_all(self, limit: int = 100) -> List[MemoryObject]: ...
    def stats(self) -> Dict: ...
    def prune(self, min_weight: float) -> int: ...
```

## Protocol extensions

Protocol changes (new fields, new relation types, behavior changes) require:

1. Update `amp/schema_v0.1.json` (or create `schema_v0.2.json`)
2. Update `docs/SPEC.md` with RFC-style language (MUST/SHOULD/MAY)
3. Update `amp/memory_object.py`
4. Migration path for existing data
5. PR description explaining the design rationale

Breaking changes require a major version bump and deprecation period.

## PR checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Linting: `ruff check amp/ tests/`
- [ ] Types: `mypy amp/`
- [ ] CHANGELOG.md updated (under `[Unreleased]`)
- [ ] Docs updated if behavior changed
- [ ] No new required dependencies without discussion

## Reporting bugs

Use GitHub Issues. Include:
- Python version, OS
- Minimal reproducible example
- Expected vs actual behavior
- Error traceback if applicable

## Discussion

For design discussions, protocol proposals, or questions:
- GitHub Discussions (preferred)
- Issues for concrete bugs/features

## Code of Conduct

Be kind. Build things. Ship code.  
This project is built in Ukraine. We appreciate patience and understanding.

---

*"The standard that everyone adopts makes everyone richer."*
