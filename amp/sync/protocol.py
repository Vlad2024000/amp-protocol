"""
AMP Multi-Agent Sync Protocol — v0.2

This is what AuraSDK doesn't have.
This is why AMP is a protocol, not a library.

The sync protocol defines:
  1. How agents discover each other's memories
  2. How they pull/push shared memories
  3. How conflicts between agents are resolved
  4. How trust propagates across agent boundaries

Architecture:
  All agents for the same user_id share one SQLite file.
  Scope controls visibility:
    private  → only the writing agent
    session  → all agents in this session_id
    user     → all agents for this user_id (cross-session)
    public   → any agent

  When Agent B calls sync(), it pulls all user-scoped memories
  written by Agent A, resolves conflicts, and stamps them
  with AGENT_SHARE provenance.

  This is how Claude remembers what GPT told it about the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..memory_object import (
    MemoryObject, MemoryRelation, MemoryTrust, MemoryScope,
    MemoryType, RelationType, ScopeType, SourceType,
)
from ..store.sqlite_backend import SQLiteBackend


@dataclass
class SyncResult:
    pulled:   int = 0   # Memories pulled from other agents
    conflicts: int = 0   # Conflicts detected
    resolved: int = 0   # Conflicts resolved (winner kept)
    skipped:  int = 0   # Already up-to-date
    errors:   List[str] = field(default_factory=list)


class AgentSyncProtocol:
    """
    Implements the AMP cross-agent memory synchronization protocol.

    Two modes:
      passive  — Agent reads shared memories on demand (default)
      active   — Agent explicitly pulls and stamps shared memories

    The passive mode is already built into SQLiteBackend._scope_filter().
    This class implements the active sync for conflict resolution and
    trust propagation.
    """

    def __init__(self, backend: SQLiteBackend):
        self.backend = backend

    def pull_from_agent(
        self,
        source_agent_id: str,
        since: Optional[str] = None,
        auto_resolve_conflicts: bool = True,
    ) -> SyncResult:
        """
        Pull user-scoped memories written by source_agent_id
        and integrate them into this agent's view.

        Conflict detection: if we already have a memory with similar
        content (high BM25 score) but different content — resolve.
        """
        result = SyncResult()

        # Get all user-scoped memories from source agent
        all_memories = self.backend.list_all(limit=500)
        source_memories = [
            m for m in all_memories
            if m.trust and m.trust.agent_id == source_agent_id
            and m.scope and m.scope.type in (ScopeType.USER, ScopeType.PUBLIC)
        ]

        for foreign_mem in source_memories:
            # Check if we already have this exact memory
            existing = self.backend.get(foreign_mem.id)
            if existing:
                result.skipped += 1
                continue

            # Search for semantically similar memory we own
            similar = self.backend.search(
                query          = foreign_mem.content,
                top_k          = 3,
                include_shared = False,
                min_weight     = 0.1,
            )

            conflict_found = False
            if similar:
                best = similar[0]
                # High BM25 score + same type = likely conflict
                if best["bm25_score"] > 0.6 and best["memory"].memory_type == foreign_mem.memory_type:
                    conflict_found = True
                    result.conflicts += 1

                    if auto_resolve_conflicts:
                        winner = MemoryObject.resolve_conflict(
                            best["memory"], foreign_mem
                        )
                        # If foreign wins, write it in
                        if winner.id == foreign_mem.id:
                            foreign_mem.trust = MemoryTrust(
                                agent_id    = self.backend.agent_id,
                                model       = "amp-sync",
                                confidence  = foreign_mem.trust.confidence if foreign_mem.trust else 0.7,
                                source      = SourceType.AGENT_SHARE,
                                verified_by = [source_agent_id],
                            )
                            self.backend.write(foreign_mem)
                            result.resolved += 1
                        # else: our memory wins, do nothing

            if not conflict_found:
                # Stamp with agent_share provenance
                foreign_mem.trust = MemoryTrust(
                    agent_id    = self.backend.agent_id,
                    model       = "amp-sync",
                    confidence  = (foreign_mem.trust.confidence if foreign_mem.trust else 0.7) * 0.9,
                    source      = SourceType.AGENT_SHARE,
                    verified_by = [source_agent_id],
                )
                self.backend.write(foreign_mem)
                result.pulled += 1

        return result

    def push_to_shared(
        self,
        memory_ids: List[str],
        scope: ScopeType = ScopeType.USER,
    ) -> int:
        """
        Promote private memories to user/public scope.
        Returns count of promoted memories.
        """
        promoted = 0
        for mid in memory_ids:
            m = self.backend.get(mid)
            if not m:
                continue
            if m.scope:
                m.scope.type = scope
                if scope == ScopeType.USER:
                    m.scope.user_id = self.backend.user_id
            self.backend.write(m)
            promoted += 1
        return promoted

    def export_snapshot(self, scope: ScopeType = ScopeType.USER) -> Dict[str, Any]:
        """
        Export all shareable memories as a JSON snapshot.
        This is the AMP wire format for cross-instance sync.
        """
        memories = self.backend.list_all(limit=1000)
        shareable = [
            m for m in memories
            if m.scope and m.scope.type in (ScopeType.USER, ScopeType.PUBLIC)
        ]
        return {
            "amp_version":  "0.2",
            "exported_at":  datetime.now(timezone.utc).isoformat(),
            "agent_id":     self.backend.agent_id,
            "user_id":      self.backend.user_id,
            "memory_count": len(shareable),
            "memories":     [m.to_dict() for m in shareable],
        }

    def import_snapshot(self, snapshot: Dict[str, Any]) -> SyncResult:
        """
        Import memories from an AMP JSON snapshot.
        Used for cross-instance sync (e.g., Claude Desktop → Claude API).
        """
        result = SyncResult()
        source_agent = snapshot.get("agent_id", "unknown")

        for mem_dict in snapshot.get("memories", []):
            try:
                m = MemoryObject.from_dict(mem_dict)
                existing = self.backend.get(m.id)
                if existing:
                    result.skipped += 1
                    continue
                # Re-stamp trust
                m.trust = MemoryTrust(
                    agent_id    = self.backend.agent_id,
                    model       = "amp-import",
                    confidence  = (m.trust.confidence if m.trust else 0.7) * 0.85,
                    source      = SourceType.AGENT_SHARE,
                    verified_by = [source_agent],
                )
                self.backend.write(m)
                result.pulled += 1
            except Exception as e:
                result.errors.append(str(e))

        return result

    def who_knows_what(self) -> Dict[str, List[str]]:
        """
        Returns a map of {agent_id: [top memory summaries]}.
        Shows what each agent contributes to the shared memory pool.
        """
        all_mems = self.backend.list_all(limit=200)
        by_agent: Dict[str, List[str]] = {}
        for m in all_mems:
            aid = m.trust.agent_id if m.trust else "unknown"
            if aid not in by_agent:
                by_agent[aid] = []
            if len(by_agent[aid]) < 5:
                by_agent[aid].append(m.content[:80])
        return by_agent
