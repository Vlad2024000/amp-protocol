/**
 * AMP — Agent Memory Protocol
 * TypeScript / JavaScript SDK v0.3.0
 *
 * Works in Node.js, Deno, Bun, and browsers (via fetch).
 *
 * Install:
 *   npm install @amp-protocol/sdk
 *   # or
 *   bun add @amp-protocol/sdk
 *
 * Usage:
 *   import { AMPClient, MemoryType } from "@amp-protocol/sdk";
 *
 *   const amp = new AMPClient({
 *     baseUrl: "http://localhost:8765",
 *     agentId: "agent-gpt",
 *     userId:  "alice",
 *   });
 *
 *   // Remember
 *   const mem = await amp.remember(
 *     "Alice prefers concise technical answers",
 *     { type: MemoryType.Preference, importance: 0.9, tags: ["comms"] }
 *   );
 *
 *   // Recall — semantic, cross-agent
 *   const results = await amp.recall("how should I talk to Alice?", { shared: true });
 *   for (const r of results) {
 *     console.log(r.content, r.score.rrf);
 *   }
 */

// ── Types ──────────────────────────────────────────────────────────────────

export enum MemoryType {
  Fact       = "fact",
  Event      = "event",
  Skill      = "skill",
  Preference = "preference",
  Context    = "context",
}

export enum ScopeType {
  Private = "private",
  Session = "session",
  User    = "user",
  Public  = "public",
}

export enum SourceType {
  UserInput  = "user_input",
  Inference  = "inference",
  ToolResult = "tool_result",
  AgentShare = "agent_share",
  System     = "system",
}

export interface MemoryTrust {
  agent_id:    string;
  model:       string;
  confidence:  number;
  source:      SourceType;
  verified_by: string[];
}

export interface MemoryScope {
  type:        ScopeType;
  agent_id?:   string | null;
  session_id?: string | null;
  user_id?:    string | null;
}

export interface MemoryRelation {
  target_id:     string;
  relation_type: string;
  strength:      number;
}

export interface Memory {
  id:          string;
  content:     string;
  memory_type: MemoryType;
  importance:  number;
  decay_rate:  number;
  permanence:  number;
  weight:      number;
  tags:        string[];
  relations:   MemoryRelation[];
  created_at:  string;
  accessed_at: string;
  trust?:      MemoryTrust;
  scope?:      MemoryScope;
  agent_id?:   string;
}

export interface SearchResult {
  memory:      Memory;
  score: {
    rrf:     number;
    vector:  number;
    bm25:    number;
    weight:  number;
  };
  shared_from: string | null;
}

export interface WriteResult {
  id:          string;
  memory_type: string;
  weight:      number;
  decay_rate:  number;
  permanence:  number;
}

export interface StoreStats {
  total:           number;
  by_type:         Record<string, number>;
  by_agent:        Record<string, number>;
  avg_importance:  number;
  embedder:        string;
  db_mode:         string;
}

export interface SyncResult {
  pulled:    number;
  conflicts: number;
  resolved:  number;
  skipped:   number;
  errors:    string[];
}

export interface AMPClientOptions {
  /** Base URL of the AMP HTTP server. Default: http://localhost:8765 */
  baseUrl?: string;
  /** Agent ID (e.g. "agent-gpt", "agent-claude"). */
  agentId?: string;
  /** User ID for cross-agent memory sharing. */
  userId?: string;
  /** Session ID for session-scoped memories. */
  sessionId?: string;
  /** Bearer token for auth (set AMP_API_KEY on server). */
  apiKey?: string;
  /** Request timeout in milliseconds. Default: 10000 */
  timeoutMs?: number;
}

export interface RememberOptions {
  type?:       MemoryType;
  importance?: number;
  tags?:       string[];
  scope?:      ScopeType;
  confidence?: number;
  model?:      string;
}

export interface RecallOptions {
  topK?:       number;
  memoryType?: MemoryType;
  minWeight?:  number;
  shared?:     boolean;
}

// ── AMPClient ──────────────────────────────────────────────────────────────

export class AMPClient {
  private readonly baseUrl:   string;
  private readonly agentId:   string;
  private readonly userId?:   string;
  private readonly sessionId?: string;
  private readonly apiKey?:   string;
  private readonly timeoutMs: number;

  constructor(options: AMPClientOptions = {}) {
    this.baseUrl   = (options.baseUrl   ?? "http://localhost:8765").replace(/\/$/, "");
    this.agentId   = options.agentId   ?? "amp-agent";
    this.userId    = options.userId;
    this.sessionId = options.sessionId;
    this.apiKey    = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 10_000;
  }

  // ── Write ────────────────────────────────────────────────────────────────

  /**
   * Store a new memory.
   *
   * @example
   * const mem = await amp.remember("Alice is based in Kyiv", {
   *   type: MemoryType.Fact,
   *   importance: 0.9,
   *   tags: ["location"],
   * });
   */
  async remember(content: string, options: RememberOptions = {}): Promise<WriteResult> {
    return this._post<WriteResult>("/v1/memories", {
      content,
      memory_type: options.type      ?? MemoryType.Fact,
      importance:  options.importance ?? 0.7,
      tags:        options.tags       ?? [],
      scope:       options.scope      ?? ScopeType.User,
      confidence:  options.confidence ?? 0.85,
      model:       options.model      ?? "unknown",
    });
  }

  // ── Read ─────────────────────────────────────────────────────────────────

  /**
   * Get a memory by its UUID.
   */
  async get(memoryId: string): Promise<Memory> {
    return this._get<Memory>(`/v1/memories/${memoryId}`);
  }

  /**
   * Semantic + keyword search across memories.
   * With shared=true (default), searches across all agents for the same user.
   *
   * @example
   * const results = await amp.recall("communication style preferences");
   * for (const r of results) {
   *   console.log(`[${r.score.rrf.toFixed(4)}] ${r.memory.content}`);
   * }
   */
  async recall(query: string, options: RecallOptions = {}): Promise<SearchResult[]> {
    const params = new URLSearchParams({
      q:          query,
      top_k:      String(options.topK      ?? 10),
      min_weight: String(options.minWeight ?? 0.0),
      shared:     String(options.shared    ?? true),
    });
    if (options.memoryType) params.set("type", options.memoryType);

    const data = await this._get<{
      results: Array<{
        id: string; content: string; memory_type: string; weight: number;
        importance: number; tags: string[]; created_at: string;
        final_score: number; bm25_score: number; vec_score?: number; amp_weight: number;
        shared_from: string | null; agent_id?: string;
      }>;
    }>(`/v1/memories/search?${params}`);

    return (data.results ?? []).map(r => ({
      memory: {
        id:          r.id,
        content:     r.content,
        memory_type: r.memory_type as MemoryType,
        importance:  r.importance,
        decay_rate:  0,
        permanence:  0,
        weight:      r.weight,
        tags:        r.tags,
        relations:   [],
        created_at:  r.created_at,
        accessed_at: r.created_at,
        agent_id:    r.agent_id,
      },
      score: {
        rrf:    r.final_score  ?? r.amp_weight,
        vector: r.vec_score   ?? 0,
        bm25:   r.bm25_score  ?? 0,
        weight: r.amp_weight  ?? 0,
      },
      shared_from: r.shared_from,
    }));
  }

  /**
   * List all memories (top N by importance).
   */
  async list(limit = 50): Promise<Memory[]> {
    const data = await this._get<{ memories: Memory[] }>("/v1/memories");
    return (data.memories ?? []).slice(0, limit);
  }

  // ── Delete ───────────────────────────────────────────────────────────────

  /**
   * Delete a memory by its UUID.
   */
  async forget(memoryId: string): Promise<boolean> {
    const data = await this._delete<{ deleted: boolean }>(`/v1/memories/${memoryId}`);
    return data.deleted;
  }

  // ── Stats ────────────────────────────────────────────────────────────────

  async stats(): Promise<StoreStats> {
    return this._get<StoreStats>("/v1/stats");
  }

  async health(): Promise<{ status: string; agent: string; ts: string }> {
    return this._get("/health");
  }

  // ── Sync ─────────────────────────────────────────────────────────────────

  /**
   * Export all shareable memories as a snapshot.
   * Use to transfer memory between instances or to back up.
   */
  async exportSnapshot(): Promise<Record<string, unknown>> {
    return this._post("/v1/sync/export", {});
  }

  /**
   * Import an AMP snapshot from another instance.
   */
  async importSnapshot(snapshot: Record<string, unknown>): Promise<SyncResult> {
    return this._post<SyncResult>("/v1/sync/import", snapshot);
  }

  /**
   * Pull memories from another agent (server-side sync).
   */
  async syncFrom(fromAgentId: string): Promise<SyncResult> {
    return this._post<SyncResult>("/v1/sync/pull", { from_agent_id: fromAgentId });
  }

  // ── HTTP helpers ──────────────────────────────────────────────────────────

  private _headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  private async _get<T>(path: string): Promise<T> {
    const resp = await this._fetch(`${this.baseUrl}${path}`, { method: "GET" });
    return resp.json() as Promise<T>;
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    const resp = await this._fetch(`${this.baseUrl}${path}`, {
      method:  "POST",
      headers: this._headers(),
      body:    JSON.stringify(body),
    });
    return resp.json() as Promise<T>;
  }

  private async _delete<T>(path: string): Promise<T> {
    const resp = await this._fetch(`${this.baseUrl}${path}`, { method: "DELETE" });
    return resp.json() as Promise<T>;
  }

  private async _fetch(url: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const resp = await fetch(url, { ...init, headers: { ...this._headers(), ...((init.headers as Record<string, string>) ?? {}) }, signal: controller.signal });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new AMPError(`AMP API error ${resp.status}: ${body}`, resp.status);
      }
      return resp;
    } finally {
      clearTimeout(timer);
    }
  }
}

// ── Error ──────────────────────────────────────────────────────────────────

export class AMPError extends Error {
  constructor(message: string, public readonly statusCode?: number) {
    super(message);
    this.name = "AMPError";
  }
}

// ── MCP tool definitions (for use in MCP server adapters) ──────────────────

export const AMP_MCP_TOOLS = [
  {
    name: "amp_remember",
    description: "Store a persistent memory in the AMP store. Survives across sessions and agents.",
    inputSchema: {
      type: "object", required: ["content"],
      properties: {
        content:     { type: "string" },
        memory_type: { type: "string", enum: ["fact","event","skill","preference","context"], default: "fact" },
        importance:  { type: "number", minimum: 0, maximum: 1, default: 0.7 },
        tags:        { type: "array", items: { type: "string" } },
        scope:       { type: "string", enum: ["private","session","user","public"], default: "user" },
      },
    },
  },
  {
    name: "amp_recall",
    description: "Semantic + keyword search across memories. Finds relevant memories even without exact word match. Searches across all agents for the same user.",
    inputSchema: {
      type: "object", required: ["query"],
      properties: {
        query:       { type: "string" },
        top_k:       { type: "integer", default: 10 },
        memory_type: { type: "string", enum: ["fact","event","skill","preference","context","any"], default: "any" },
        min_weight:  { type: "number", default: 0.0 },
        shared:      { type: "boolean", default: true },
      },
    },
  },
  {
    name: "amp_reflect",
    description: "Get memory store overview: stats, top memories by weight. Call at session start to orient yourself.",
    inputSchema: {
      type: "object",
      properties: { top_k: { type: "integer", default: 5 } },
    },
  },
  {
    name: "amp_forget",
    description: "Delete a specific memory by ID.",
    inputSchema: {
      type: "object", required: ["memory_id"],
      properties: { memory_id: { type: "string" } },
    },
  },
] as const;

// ── Default export ─────────────────────────────────────────────────────────

export default AMPClient;
