"""
AMP HTTP Server — v0.2

REST API for AMP memory store.
Pure stdlib: http.server + threading. Zero dependencies.

Endpoints:
  POST   /v1/memories              — write memory
  GET    /v1/memories/search?q=... — search
  GET    /v1/memories/{id}         — get by id
  DELETE /v1/memories/{id}         — delete
  GET    /v1/memories              — list all (top 50)
  GET    /v1/stats                 — store statistics
  POST   /v1/sync/export           — export snapshot
  POST   /v1/sync/import           — import snapshot
  GET    /v1/sync/who              — who knows what
  GET    /health                   — health check

Auth: Bearer token via AMP_API_KEY env var (optional for MVP)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from amp.memory_object import (
    MemoryObject, MemoryType, ScopeType,
    MemoryTrust, MemoryScope, SourceType,
)
from amp.store.sqlite_backend import SQLiteBackend
from amp.sync.protocol import AgentSyncProtocol


# ── Globals ───────────────────────────────────────────────────────────────

DB_PATH    = os.environ.get("AMP_DB_PATH",    "./amp_data/amp.db")
AGENT_ID   = os.environ.get("AMP_AGENT_ID",   "amp-http-server")
USER_ID    = os.environ.get("AMP_USER_ID",    "default-user")
SESSION_ID = os.environ.get("AMP_SESSION_ID", None)
API_KEY    = os.environ.get("AMP_API_KEY",    None)   # None = no auth
HOST       = os.environ.get("AMP_HOST",       "0.0.0.0")
PORT       = int(os.environ.get("AMP_PORT",   "8765"))

BACKEND: Optional[SQLiteBackend] = None
SYNCER:  Optional[AgentSyncProtocol] = None


def init_server():
    global BACKEND, SYNCER
    BACKEND = SQLiteBackend(DB_PATH, AGENT_ID, USER_ID, SESSION_ID)
    SYNCER  = AgentSyncProtocol(BACKEND)


# ── HTTP handler ──────────────────────────────────────────────────────────

class AMPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # Silence default logging; use our own

    def _auth(self) -> bool:
        if not API_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {API_KEY}"

    def _read_body(self) -> Optional[Dict]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _send(self, status: int, body: Any) -> None:
        data = json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _ok(self, body: Any) -> None:
        self._send(200, body)

    def _err(self, status: int, message: str) -> None:
        self._send(status, {"error": message})

    # ── Routing ───────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if not self._auth():
            return self._err(401, "Unauthorized")
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = parse_qs(parsed.query)

        try:
            if path == "/health":
                return self._ok({"status": "ok", "agent": AGENT_ID, "ts": datetime.now(timezone.utc).isoformat()})

            if path == "/v1/stats":
                return self._ok(BACKEND.stats())

            if path == "/v1/memories":
                memories = BACKEND.list_all(limit=50)
                return self._ok({
                    "memories": [self._mem_summary(m) for m in memories],
                    "total": len(memories),
                })

            if path == "/v1/memories/search":
                q          = qs.get("q", [""])[0]
                top_k      = int(qs.get("top_k", ["10"])[0])
                min_weight = float(qs.get("min_weight", ["0.0"])[0])
                mtype      = qs.get("type", [None])[0]
                shared     = qs.get("shared", ["true"])[0].lower() != "false"

                if not q:
                    return self._err(400, "Missing query parameter 'q'")

                mtypes = [MemoryType(mtype)] if mtype else None
                results = BACKEND.search(
                    query          = q,
                    top_k          = top_k,
                    memory_types   = mtypes,
                    min_weight     = min_weight,
                    include_shared = shared,
                )
                return self._ok({
                    "query":   q,
                    "total":   len(results),
                    "results": [
                        {
                            **self._mem_summary(r["memory"]),
                            "final_score": r["final_score"],
                            "bm25_score":  r["bm25_score"],
                            "amp_weight":  r["amp_weight"],
                            "shared_from": r["shared_from"],
                        }
                        for r in results
                    ],
                })

            if path.startswith("/v1/memories/"):
                mid = path[len("/v1/memories/"):]
                m   = BACKEND.get(mid)
                if not m:
                    return self._err(404, f"Memory {mid} not found")
                return self._ok(m.to_dict())

            if path == "/v1/sync/who":
                return self._ok(SYNCER.who_knows_what())

            self._err(404, f"Not found: {path}")

        except Exception as e:
            self._err(500, f"Internal error: {e}\n{traceback.format_exc()}")

    def do_POST(self):
        if not self._auth():
            return self._err(401, "Unauthorized")
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        body   = self._read_body()
        if body is None:
            return self._err(400, "Invalid JSON body")

        try:
            if path == "/v1/memories":
                return self._handle_write(body)

            if path == "/v1/sync/export":
                snapshot = SYNCER.export_snapshot()
                return self._ok(snapshot)

            if path == "/v1/sync/import":
                if not body:
                    return self._err(400, "Missing snapshot body")
                result = SYNCER.import_snapshot(body)
                return self._ok({
                    "pulled":    result.pulled,
                    "conflicts": result.conflicts,
                    "resolved":  result.resolved,
                    "skipped":   result.skipped,
                    "errors":    result.errors,
                })

            if path == "/v1/sync/pull":
                source = body.get("from_agent_id")
                if not source:
                    return self._err(400, "Missing from_agent_id")
                result = SYNCER.pull_from_agent(source)
                return self._ok({
                    "pulled":    result.pulled,
                    "conflicts": result.conflicts,
                    "resolved":  result.resolved,
                    "skipped":   result.skipped,
                })

            self._err(404, f"Not found: {path}")

        except Exception as e:
            self._err(500, f"Internal error: {e}\n{traceback.format_exc()}")

    def do_DELETE(self):
        if not self._auth():
            return self._err(401, "Unauthorized")
        path = urlparse(self.path).path.rstrip("/")

        if path.startswith("/v1/memories/"):
            mid     = path[len("/v1/memories/"):]
            deleted = BACKEND.delete(mid)
            return self._ok({"deleted": deleted, "id": mid})

        self._err(404, "Not found")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _handle_write(self, body: Dict) -> None:
        content    = body.get("content", "").strip()
        if not content:
            return self._err(400, "Missing 'content'")

        mtype      = MemoryType(body.get("memory_type", "fact"))
        importance = float(body.get("importance", 0.7))
        tags       = body.get("tags", [])
        structured = body.get("structured", {})
        scope_str  = body.get("scope", "user")
        scope_type = ScopeType(scope_str)

        m = MemoryObject(
            content     = content,
            memory_type = mtype,
            importance  = importance,
            tags        = tags,
            structured  = structured,
            trust       = MemoryTrust(
                agent_id   = AGENT_ID,
                model      = body.get("model", "unknown"),
                confidence = float(body.get("confidence", 0.8)),
                source     = SourceType(body.get("source", "user_input")),
            ),
            scope       = MemoryScope(
                type       = scope_type,
                agent_id   = AGENT_ID,
                user_id    = USER_ID,
                session_id = SESSION_ID,
            ),
        )
        BACKEND.write(m)
        self._ok({
            "id":          m.id,
            "memory_type": m.memory_type.value,
            "weight":      round(m.weight(), 4),
            "decay_rate":  m.decay_rate,
            "permanence":  m.permanence,
        })

    def _mem_summary(self, m: MemoryObject) -> Dict:
        return {
            "id":          m.id,
            "content":     m.content,
            "type":        m.memory_type.value,
            "weight":      round(m.weight(), 4),
            "importance":  m.importance,
            "tags":        m.tags,
            "created_at":  m.created_at.isoformat(),
            "relations":   len(m.relations),
            "agent_id":    m.trust.agent_id if m.trust else None,
        }


# ── Server runner ─────────────────────────────────────────────────────────

def run(host: str = HOST, port: int = PORT, threaded: bool = True):
    init_server()

    server = HTTPServer((host, port), AMPHandler)
    if threaded:
        server.socket.setsockopt = lambda *a: None  # no-op SO_REUSEADDR

    print(f"AMP HTTP Server v0.2")
    print(f"  Listening : http://{host}:{port}")
    print(f"  Agent     : {AGENT_ID}")
    print(f"  User      : {USER_ID}")
    print(f"  DB        : {DB_PATH}")
    print(f"  Auth      : {'Bearer token' if API_KEY else 'none (dev mode)'}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    run()
