"""
AMP — Agent Memory Protocol
Version: 0.1.0

MemoryObject: the fundamental unit of agent memory.

M = (content, relations, time, trust, scope)

Every memory an agent holds is represented as a MemoryObject.
The protocol defines how these objects are created, stored,
retrieved, decayed, and shared between agents.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Enums — the vocabulary of AMP
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    FACT        = "fact"        # Verifiable: "User's name is Andrii"
    EVENT       = "event"       # Happened once: "User asked about TCP/IP on 2025-03-01"
    SKILL       = "skill"       # Procedural: "User prefers code in Python"
    PREFERENCE  = "preference"  # Stable taste: "User dislikes verbose explanations"
    CONTEXT     = "context"     # Ephemeral: current session state


class RelationType(str, Enum):
    DERIVED_FROM  = "derived_from"  # This memory was inferred from another
    CONTRADICTS   = "contradicts"   # Conflicts with another memory
    SUPPORTS      = "supports"      # Corroborates another memory
    UPDATES       = "updates"       # Supersedes an older memory
    RELATED_TO    = "related_to"    # Loose semantic connection


class ScopeType(str, Enum):
    PRIVATE = "private"   # Only this agent
    SESSION = "session"   # All agents in this session
    USER    = "user"      # All agents for this user, across sessions
    PUBLIC  = "public"    # Any agent (read-only)


class SourceType(str, Enum):
    USER_INPUT   = "user_input"    # Directly stated by user
    INFERENCE    = "inference"     # Derived by the agent
    TOOL_RESULT  = "tool_result"   # Returned by an external tool
    AGENT_SHARE  = "agent_share"   # Written by another agent
    SYSTEM       = "system"        # Written by AMP infrastructure


# ---------------------------------------------------------------------------
# Sub-objects
# ---------------------------------------------------------------------------

@dataclass
class MemoryRelation:
    """
    A directed edge in the memory graph.
    target_id → another MemoryObject's id.
    strength ∈ [0, 1] — how strongly this relation holds.
    """
    target_id: str
    relation_type: RelationType
    strength: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")


@dataclass
class MemoryTrust:
    """
    Provenance of a memory.
    Who wrote it, with what model, and how confident.
    Multiple agents can verify a memory, raising its trust.
    """
    agent_id: str
    model: str
    confidence: float                          # ∈ [0, 1]
    source: SourceType = SourceType.INFERENCE
    verified_by: List[str] = field(default_factory=list)  # agent_ids

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def effective_confidence(self) -> float:
        """
        Boost confidence slightly for each independent verifying agent.
        Trust grows with consensus but is capped at 1.0.
        Formula: min(1.0, confidence + 0.05 * len(verified_by))
        """
        return min(1.0, self.confidence + 0.05 * len(self.verified_by))


@dataclass
class MemoryScope:
    """
    Access control for a memory.
    Determines which agents can read or write this memory.
    """
    type: ScopeType
    agent_id: Optional[str]   = None
    session_id: Optional[str] = None
    user_id: Optional[str]    = None

    def allows_read(self, requesting_agent_id: str, session_id: Optional[str] = None) -> bool:
        if self.type == ScopeType.PUBLIC:
            return True
        if self.type == ScopeType.USER and self.user_id:
            return True  # caller is assumed same-user; enforce at API layer
        if self.type == ScopeType.SESSION:
            return session_id is not None and session_id == self.session_id
        if self.type == ScopeType.PRIVATE:
            return requesting_agent_id == self.agent_id
        return False


# ---------------------------------------------------------------------------
# Decay presets — λ values for common memory types
# ---------------------------------------------------------------------------

DECAY_PRESETS: Dict[MemoryType, float] = {
    MemoryType.FACT:        0.001,   # Very slow — names, facts persist
    MemoryType.PREFERENCE:  0.002,   # Slow — preferences are stable
    MemoryType.SKILL:       0.003,   # Slow — skills persist
    MemoryType.EVENT:       0.05,    # Medium — events fade over weeks
    MemoryType.CONTEXT:     0.5,     # Fast — session context fades in hours
}

PERMANENCE_PRESETS: Dict[MemoryType, float] = {
    MemoryType.FACT:        0.3,   # Facts keep 30% weight floor
    MemoryType.PREFERENCE:  0.2,
    MemoryType.SKILL:       0.25,
    MemoryType.EVENT:       0.0,
    MemoryType.CONTEXT:     0.0,
}


# ---------------------------------------------------------------------------
# The core object
# ---------------------------------------------------------------------------

@dataclass
class MemoryObject:
    """
    The fundamental unit of AMP.

    A MemoryObject represents a single piece of agent memory
    across five dimensions:

      content    — what is remembered (text + optional vector + structured fields)
      relations  — how this memory connects to others (graph edges)
      time       — when created, last accessed, and how it decays
      trust      — who wrote it, how confident, who verified
      scope      — who can read or write it

    Every MemoryObject has a weight that decays over time:

      weight(t) = importance × e^(−λ × Δt_days) + permanence

    where:
      importance  — how significant this memory is (0–1)
      λ           — decay rate (higher = faster forgetting)
      Δt_days     — days since creation
      permanence  — minimum weight floor (memory never fully disappears)
    """

    # ── Identity ──────────────────────────────────────────────────────────
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Content ───────────────────────────────────────────────────────────
    content: str = ""
    embedding: Optional[List[float]] = None          # Semantic vector
    structured: Dict[str, Any] = field(default_factory=dict)  # Key-value facts
    memory_type: MemoryType = MemoryType.CONTEXT
    tags: List[str] = field(default_factory=list)

    # ── Relations ─────────────────────────────────────────────────────────
    relations: List[MemoryRelation] = field(default_factory=list)

    # ── Time ──────────────────────────────────────────────────────────────
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5       # Set by the writing agent. ∈ [0, 1]
    decay_rate: float = 0.01      # λ — overridden by preset if not set explicitly
    permanence: float = 0.0       # Floor weight

    # ── Trust ─────────────────────────────────────────────────────────────
    trust: Optional[MemoryTrust] = None

    # ── Scope ─────────────────────────────────────────────────────────────
    scope: Optional[MemoryScope] = None

    def __post_init__(self):
        # Auto-apply decay presets if using defaults
        if self.decay_rate == 0.01:
            self.decay_rate = DECAY_PRESETS.get(self.memory_type, 0.01)
        if self.permanence == 0.0:
            self.permanence = PERMANENCE_PRESETS.get(self.memory_type, 0.0)

    # ── Weight function ───────────────────────────────────────────────────

    def weight(self, at: Optional[datetime] = None) -> float:
        """
        Current memory weight using the AMP decay function:

          weight(t) = importance × e^(−λ × Δt_days) + permanence

        Args:
            at: Point in time to evaluate weight. Defaults to now.

        Returns:
            Float in [permanence, importance + permanence]
        """
        now = at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        delta_days = (now - created).total_seconds() / 86400.0
        w = self.importance * math.exp(-self.decay_rate * delta_days) + self.permanence
        return round(min(1.0, max(0.0, w)), 6)

    def touch(self) -> None:
        """Update access time (relevant for LRU eviction strategies)."""
        self.accessed_at = datetime.now(timezone.utc)

    # ── Graph helpers ─────────────────────────────────────────────────────

    def add_relation(
        self,
        target_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
    ) -> None:
        self.relations.append(
            MemoryRelation(target_id=target_id, relation_type=relation_type, strength=strength)
        )

    def contradicts(self, other: "MemoryObject", strength: float = 1.0) -> None:
        self.add_relation(other.id, RelationType.CONTRADICTS, strength)
        other.add_relation(self.id, RelationType.CONTRADICTS, strength)

    def updates(self, old: "MemoryObject") -> None:
        """Mark this memory as superseding an older one."""
        self.add_relation(old.id, RelationType.UPDATES, 1.0)

    # ── Conflict resolution ───────────────────────────────────────────────

    @staticmethod
    def resolve_conflict(
        a: "MemoryObject",
        b: "MemoryObject",
    ) -> "MemoryObject":
        """
        Given two conflicting memories, return the one that should be trusted.

        Resolution priority (highest wins):
          1. Higher effective trust confidence
          2. Higher current weight (recency × importance)
          3. Source hierarchy: user_input > tool_result > inference > agent_share

        The losing memory gets a CONTRADICTS relation pointing to the winner.
        """
        source_priority = {
            SourceType.USER_INPUT:  4,
            SourceType.TOOL_RESULT: 3,
            SourceType.INFERENCE:   2,
            SourceType.AGENT_SHARE: 1,
            SourceType.SYSTEM:      0,
        }

        conf_a = a.trust.effective_confidence if a.trust else 0.5
        conf_b = b.trust.effective_confidence if b.trust else 0.5

        src_a = source_priority.get(a.trust.source, 0) if a.trust else 0
        src_b = source_priority.get(b.trust.source, 0) if b.trust else 0

        score_a = conf_a * 0.5 + a.weight() * 0.3 + (src_a / 4) * 0.2
        score_b = conf_b * 0.5 + b.weight() * 0.3 + (src_b / 4) * 0.2

        winner, loser = (a, b) if score_a >= score_b else (b, a)
        loser.add_relation(winner.id, RelationType.CONTRADICTS, 1.0)
        return winner

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialize(v) for k, v in asdict(obj).items()}
            return obj

        return _serialize(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryObject":
        d = dict(data)

        # Parse datetimes
        for tf in ("created_at", "accessed_at"):
            if isinstance(d.get(tf), str):
                d[tf] = datetime.fromisoformat(d[tf])

        # Parse enums
        if "memory_type" in d:
            d["memory_type"] = MemoryType(d["memory_type"])

        # Parse relations
        if "relations" in d and d["relations"]:
            d["relations"] = [
                MemoryRelation(
                    target_id=r["target_id"],
                    relation_type=RelationType(r["relation_type"]),
                    strength=r.get("strength", 1.0),
                )
                for r in d["relations"]
            ]

        # Parse trust
        if d.get("trust"):
            t = d["trust"]
            d["trust"] = MemoryTrust(
                agent_id=t["agent_id"],
                model=t["model"],
                confidence=t["confidence"],
                source=SourceType(t.get("source", "inference")),
                verified_by=t.get("verified_by", []),
            )

        # Parse scope
        if d.get("scope"):
            s = d["scope"]
            d["scope"] = MemoryScope(
                type=ScopeType(s["type"]),
                agent_id=s.get("agent_id"),
                session_id=s.get("session_id"),
                user_id=s.get("user_id"),
            )

        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryObject":
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        w = self.weight()
        return (
            f"MemoryObject(id={self.id[:8]}…, "
            f"type={self.memory_type.value}, "
            f"weight={w:.3f}, "
            f"content={self.content[:60]!r})"
        )
