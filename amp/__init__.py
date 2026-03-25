"""
AMP — Agent Memory Protocol v0.3.0
The open standard for persistent, cross-agent AI memory.

    from amp import MemoryStore, MemoryType
    store = MemoryStore("./memory.db", agent_id="claude", user_id="alice")
    store.add("Alice prefers concise answers", memory_type=MemoryType.PREFERENCE)
    results = store.search("communication style")
"""

try:
    from importlib.metadata import version
    __version__ = version("amp-memory")
except Exception:
    __version__ = "0.3.0-dev"

from .memory_object import (
    MemoryObject, MemoryRelation, MemoryScope, MemoryTrust,
    MemoryType, RelationType, ScopeType, SourceType,
    DECAY_PRESETS, PERMANENCE_PRESETS,
)
from .store.memory_store import MemoryStore
from .store.pgvector_backend import PgVectorBackend

__all__ = [
    "__version__",
    "MemoryObject", "MemoryRelation", "MemoryTrust", "MemoryScope",
    "MemoryType", "RelationType", "ScopeType", "SourceType",
    "DECAY_PRESETS", "PERMANENCE_PRESETS",
    "MemoryStore", "PgVectorBackend",
]
