"""
AMP BM25 Engine — v0.2

Hybrid retrieval:
  score = α × BM25(query, doc) + β × AMP_weight(memory) + γ × tag_boost

BM25 parameters (Robertson & Zaragoza 2009):
  k1 = 1.5  — term frequency saturation
  b  = 0.75 — length normalization

AMP_weight:
  weight(t) = importance × e^(−λ × Δt_days) + permanence

Combined score ranks by relevance AND recency+importance simultaneously.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Tokenizer ─────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
    "who", "how", "when", "where", "why", "not", "no", "s", "t",
}


def tokenize(text: str) -> List[str]:
    """
    Lowercase, split on non-alphanumeric, remove stop words, 2+ char tokens.
    Also includes bigrams for better phrase matching.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    unigrams = [w for w in words if len(w) >= 2 and w not in _STOP_WORDS]
    bigrams  = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
    return unigrams + bigrams


# ── BM25 index ────────────────────────────────────────────────────────────

class BM25Index:
    """
    Inverted index with BM25 scoring.

    Supports:
      - Incremental add/remove (no full rebuild)
      - Per-document length normalization
      - IDF with +1 smoothing (no log(0))
      - Numpy vectorized scoring for top-k
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        # doc_id → token counts
        self._docs:   Dict[str, Counter] = {}
        # doc_id → token list length
        self._lengths: Dict[str, int]    = {}
        # token → {doc_id: count}
        self._index:  Dict[str, Dict[str, int]] = defaultdict(dict)
        # running average document length
        self._total_tokens: int = 0

    @property
    def N(self) -> int:
        return len(self._docs)

    @property
    def avgdl(self) -> float:
        return self._total_tokens / max(1, self.N)

    # ── Index management ──────────────────────────────────────────────────

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)

        if doc_id in self._docs:
            self.remove(doc_id)

        self._docs[doc_id]   = counts
        self._lengths[doc_id] = len(tokens)
        self._total_tokens   += len(tokens)

        for token, count in counts.items():
            self._index[token][doc_id] = count

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        for token in self._docs[doc_id]:
            self._index[token].pop(doc_id, None)
        self._total_tokens -= self._lengths.pop(doc_id, 0)
        del self._docs[doc_id]

    # ── Scoring ───────────────────────────────────────────────────────────

    def idf(self, token: str) -> float:
        df = len(self._index.get(token, {}))
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, doc_id: str, query_tokens: List[str]) -> float:
        if doc_id not in self._docs:
            return 0.0
        dl   = self._lengths[doc_id]
        norm = 1 - self.b + self.b * (dl / max(1, self.avgdl))
        total = 0.0
        counts = self._docs[doc_id]
        for token in query_tokens:
            tf  = counts.get(token, 0)
            idf = self.idf(token)
            total += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return total

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return (doc_id, bm25_score) sorted descending, max top_k results.
        candidate_ids: restrict search to this subset (for filtering).
        """
        if self.N == 0:
            return []

        qtokens = tokenize(query)
        if not qtokens:
            return []

        # Collect candidate docs that contain at least one query token
        candidates: Dict[str, float] = {}
        for token in qtokens:
            for doc_id in self._index.get(token, {}):
                if candidate_ids is not None and doc_id not in candidate_ids:
                    continue
                candidates[doc_id] = 0.0

        if not candidates:
            # Fallback: score all docs (or candidate_ids subset)
            pool = candidate_ids if candidate_ids is not None else list(self._docs.keys())
            for doc_id in pool:
                candidates[doc_id] = 0.0

        # Score
        for doc_id in candidates:
            candidates[doc_id] = self.score(doc_id, qtokens)

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ── Hybrid retrieval ──────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 relevance with AMP memory weight.

    final_score = α×bm25_norm + β×amp_weight + γ×tag_boost

    Coefficients (tunable):
      α = 0.50  — semantic relevance
      β = 0.35  — memory weight (recency × importance)
      γ = 0.15  — tag exact match bonus
    """

    ALPHA = 0.50
    BETA  = 0.35
    GAMMA = 0.15

    def __init__(self):
        self.bm25 = BM25Index()
        # doc_id → (importance, decay_rate, permanence, created_at_ts, tags)
        self._meta: Dict[str, Tuple] = {}

    def index(
        self,
        doc_id: str,
        text: str,
        importance: float,
        decay_rate: float,
        permanence: float,
        created_at: datetime,
        tags: List[str],
    ) -> None:
        # BM25 indexes text + tags together
        full_text = text + " " + " ".join(tags)
        self.bm25.add(doc_id, full_text)
        self._meta[doc_id] = (importance, decay_rate, permanence, created_at.timestamp(), tags)

    def remove(self, doc_id: str) -> None:
        self.bm25.remove(doc_id)
        self._meta.pop(doc_id, None)

    def _amp_weight(self, doc_id: str, now_ts: float) -> float:
        if doc_id not in self._meta:
            return 0.5
        imp, lam, perm, created_ts, _ = self._meta[doc_id]
        delta_days = max(0, (now_ts - created_ts) / 86400.0)
        w = imp * math.exp(-lam * delta_days) + perm
        return min(1.0, max(0.0, w))

    def _tag_boost(self, doc_id: str, query_tags: List[str]) -> float:
        if not query_tags or doc_id not in self._meta:
            return 0.0
        doc_tags = set(self._meta[doc_id][4])
        matches  = sum(1 for t in query_tags if t in doc_tags)
        return min(1.0, matches / max(1, len(query_tags)))

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_ids: Optional[List[str]] = None,
        query_tags: Optional[List[str]] = None,
        min_weight: float = 0.0,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Returns list of (doc_id, final_score, bm25_score, amp_weight).
        """
        now_ts = datetime.now(timezone.utc).timestamp()
        query_tags = query_tags or []

        # Filter by min_weight first (cheap operation)
        if candidate_ids is not None:
            pool = [
                did for did in candidate_ids
                if self._amp_weight(did, now_ts) >= min_weight
            ]
        else:
            pool = [
                did for did in self._meta
                if self._amp_weight(did, now_ts) >= min_weight
            ]

        if not pool:
            return []

        # BM25 scores
        bm25_results = dict(self.bm25.search(query, top_k=len(pool) * 2, candidate_ids=pool))

        # Normalize BM25 scores to [0, 1]
        bm25_vals = np.array(list(bm25_results.values()), dtype=np.float64)
        if bm25_vals.max() > 0:
            bm25_vals = bm25_vals / bm25_vals.max()
        bm25_norm = {did: float(s) for did, s in zip(bm25_results.keys(), bm25_vals)}

        # Compute final scores for all pool members
        scored: List[Tuple[str, float, float, float]] = []
        for did in pool:
            b_score = bm25_norm.get(did, 0.0)
            a_score = self._amp_weight(did, now_ts)
            t_score = self._tag_boost(did, query_tags)
            final   = self.ALPHA * b_score + self.BETA * a_score + self.GAMMA * t_score
            scored.append((did, final, b_score, a_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._meta)
