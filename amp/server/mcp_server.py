"""
AMP MCP Server — v0.2

Same 6 tools as v0.1, now backed by:
  - SQLite persistence (memories survive restarts)
  - BM25 + AMP hybrid search (semantic without embeddings)
  - Multi-agent sync (amp_sync tool added)

New tools in v0.2:
  amp_sync    — pull memories from another agent
  amp_export  — export snapshot for cross-instance sync
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from amp.memory_object import (
    MemoryObject, MemoryType, ScopeType, SourceType,
    MemoryTrust, MemoryScope, RelationType,
)
from amp.store.sqlite_backend import SQLiteBackend
from amp.sync.protocol import AgentSyncProtocol

# ── Init ──────────────────────────────────────────────────────────────────

DB_PATH    = os.environ.get("AMP_DB_PATH",    "./amp_data/amp.db")
AGENT_ID   = os.environ.get("AMP_AGENT_ID",   "amp-mcp-v2")
USER_ID    = os.environ.get("AMP_USER_ID",    "default-user")
SESSION_ID = os.environ.get("AMP_SESSION_ID", None)

BACKEND = SQLiteBackend(DB_PATH, AGENT_ID, USER_ID, SESSION_ID)
SYNCER  = AgentSyncProtocol(BACKEND)

# ── JSON-RPC helpers ──────────────────────────────────────────────────────

def ok(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}

def err(req_id, code, message, data=None):
    e = {"code": code, "message": message}
    if data: e["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": e}

def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()

# ── Tools ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "amp_remember",
        "description": "Store a persistent memory. Survives across sessions and agents.",
        "inputSchema": {
            "type": "object", "required": ["content"],
            "properties": {
                "content":     {"type": "string"},
                "memory_type": {"type": "string", "enum": ["fact","event","skill","preference","context"], "default": "fact"},
                "importance":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7},
                "tags":        {"type": "array", "items": {"type": "string"}},
                "scope":       {"type": "string", "enum": ["private","session","user","public"], "default": "user"},
                "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.85},
            }
        }
    },
    {
        "name": "amp_recall",
        "description": "Semantic search across memories using BM25 + AMP weight scoring. Finds relevant memories even without exact word match.",
        "inputSchema": {
            "type": "object", "required": ["query"],
            "properties": {
                "query":       {"type": "string"},
                "top_k":       {"type": "integer", "default": 10},
                "memory_type": {"type": "string", "enum": ["fact","event","skill","preference","context","any"], "default": "any"},
                "min_weight":  {"type": "number", "default": 0.02},
                "shared":      {"type": "boolean", "default": True, "description": "Include memories from other agents (cross-agent recall)"},
            }
        }
    },
    {
        "name": "amp_reflect",
        "description": "Full memory store overview: stats, top memories by weight. Use at session start.",
        "inputSchema": {
            "type": "object",
            "properties": {"top_k": {"type": "integer", "default": 5}}
        }
    },
    {
        "name": "amp_forget",
        "description": "Delete a memory by ID.",
        "inputSchema": {
            "type": "object", "required": ["memory_id"],
            "properties": {"memory_id": {"type": "string"}}
        }
    },
    {
        "name": "amp_sync",
        "description": "Pull shared memories from another agent into this agent's view. Use after another agent has written memories you want to access.",
        "inputSchema": {
            "type": "object", "required": ["from_agent_id"],
            "properties": {
                "from_agent_id": {"type": "string", "description": "Agent ID to pull from"},
                "resolve":       {"type": "boolean", "default": True, "description": "Auto-resolve conflicts"},
            }
        }
    },
    {
        "name": "amp_export",
        "description": "Export all shareable memories as AMP snapshot JSON. Use to transfer memory between instances.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "amp_relate",
        "description": "Create a relation between two memories in the memory graph.",
        "inputSchema": {
            "type": "object", "required": ["from_id", "to_id", "relation_type"],
            "properties": {
                "from_id":       {"type": "string"},
                "to_id":         {"type": "string"},
                "relation_type": {"type": "string", "enum": ["derived_from","contradicts","supports","updates","related_to"]},
                "strength":      {"type": "number", "default": 1.0},
            }
        }
    },
]

# ── Handlers ──────────────────────────────────────────────────────────────

def handle_amp_remember(args):
    content    = args["content"]
    mtype      = MemoryType(args.get("memory_type", "fact"))
    importance = float(args.get("importance", 0.7))
    tags       = args.get("tags", [])
    scope_type = ScopeType(args.get("scope", "user"))
    confidence = float(args.get("confidence", 0.85))

    m = MemoryObject(
        content     = content,
        memory_type = mtype,
        importance  = importance,
        tags        = tags,
        trust = MemoryTrust(
            agent_id   = AGENT_ID,
            model      = "claude",
            confidence = confidence,
            source     = SourceType.USER_INPUT,
        ),
        scope = MemoryScope(
            type       = scope_type,
            agent_id   = AGENT_ID,
            user_id    = USER_ID,
            session_id = SESSION_ID,
        ),
    )
    BACKEND.write(m)
    return {
        "memory_id":   m.id,
        "memory_type": m.memory_type.value,
        "weight":      round(m.weight(), 4),
        "decay_rate":  m.decay_rate,
        "permanence":  m.permanence,
        "persisted":   True,
    }


def handle_amp_recall(args):
    query      = args["query"]
    top_k      = int(args.get("top_k", 10))
    mtype_str  = args.get("memory_type", "any")
    min_weight = float(args.get("min_weight", 0.02))
    shared     = bool(args.get("shared", True))

    mtypes = [MemoryType(mtype_str)] if mtype_str != "any" else None

    results = BACKEND.search(
        query          = query,
        top_k          = top_k,
        memory_types   = mtypes,
        min_weight     = min_weight,
        include_shared = shared,
    )

    return {
        "query":       query,
        "total_found": len(results),
        "memories": [
            {
                "id":          r["memory"].id,
                "content":     r["memory"].content,
                "type":        r["memory"].memory_type.value,
                "weight":      r["amp_weight"],
                "importance":  r["memory"].importance,
                "tags":        r["memory"].tags,
                "final_score": r["final_score"],
                "bm25_score":  r["bm25_score"],
                "shared_from": r["shared_from"],
                "created_at":  r["memory"].created_at.isoformat(),
            }
            for r in results
        ],
    }


def handle_amp_reflect(args):
    top_k = int(args.get("top_k", 5))
    stats = BACKEND.stats()
    top   = BACKEND.list_all(limit=top_k)
    return {
        "store_stats":  stats,
        "agent_id":     BACKEND.agent_id,
        "user_id":      BACKEND.user_id,
        "top_memories": [
            {
                "id":      m.id,
                "content": m.content,
                "type":    m.memory_type.value,
                "weight":  round(m.weight(), 4),
                "tags":    m.tags,
            }
            for m in top
        ],
    }


def handle_amp_forget(args):
    mid     = args["memory_id"]
    deleted = BACKEND.delete(mid)
    return {"deleted": deleted, "memory_id": mid}


def handle_amp_sync(args):
    source  = args["from_agent_id"]
    resolve = bool(args.get("resolve", True))
    result  = SYNCER.pull_from_agent(source, auto_resolve_conflicts=resolve)
    return {
        "pulled":    result.pulled,
        "conflicts": result.conflicts,
        "resolved":  result.resolved,
        "skipped":   result.skipped,
        "errors":    result.errors,
    }


def handle_amp_export(args):
    return SYNCER.export_snapshot()


def handle_amp_relate(args):
    from_id = args["from_id"]
    to_id   = args["to_id"]
    rtype   = RelationType(args["relation_type"])
    strength = float(args.get("strength", 1.0))
    m = BACKEND.get(from_id)
    if not m:
        return {"error": f"Memory {from_id} not found"}
    m.add_relation(to_id, rtype, strength)
    BACKEND.write(m)
    return {"from_id": from_id, "to_id": to_id, "relation": rtype.value, "strength": strength}


HANDLERS = {
    "amp_remember": handle_amp_remember,
    "amp_recall":   handle_amp_recall,
    "amp_reflect":  handle_amp_reflect,
    "amp_forget":   handle_amp_forget,
    "amp_sync":     handle_amp_sync,
    "amp_export":   handle_amp_export,
    "amp_relate":   handle_amp_relate,
}

# ── Router ────────────────────────────────────────────────────────────────

def route(msg):
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "amp-memory-server", "version": "0.2.0"},
        })
    if method in ("initialized", "ping"):
        return ok(req_id, {}) if req_id else None
    if method == "tools/list":
        return ok(req_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name not in HANDLERS:
            return err(req_id, -32601, f"Unknown tool: {name}")
        try:
            result = HANDLERS[name](args)
            return ok(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str, indent=2)}],
                "isError": "error" in result,
            })
        except Exception as e:
            return err(req_id, -32000, str(e), traceback.format_exc())
    if method in ("resources/list", "prompts/list"):
        return ok(req_id, {"resources": [], "prompts": []})
    if req_id:
        return err(req_id, -32601, f"Method not found: {method}")
    return None

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log = open(os.environ.get("AMP_LOG", "/tmp/amp-mcp-v2.log"), "a")
    log.write(f"[START] AMP MCP v0.2 agent={AGENT_ID} db={DB_PATH}\n")
    log.flush()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg  = json.loads(raw)
            resp = route(msg)
        except Exception as e:
            resp = err(None, -32700, str(e))

        if resp:
            send(resp)
            log.write(f"[OUT] {json.dumps(resp)[:120]}\n")
            log.flush()

    log.close()

if __name__ == "__main__":
    main()
