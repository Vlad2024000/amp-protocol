"""
AMP Embedding Engine — v0.3

Three layers:

  1. LSAEmbedder  — Latent Semantic Analysis (TF-IDF + Truncated SVD via scipy).
                    Runs 100% offline. Dimensions: 256.
                    Real semantic search — "startup idea" finds "company concept".

  2. OpenAIEmbedder — Calls text-embedding-3-small. Drop-in for production.
                       Dimensions: 1536 (small) or 3072 (large).

  3. EmbeddingEngine — Facade. Auto-selects LSA → OpenAI.
                        Thread-safe. Batches requests. Caches results (LRU).

LSA theory:
  TF-IDF matrix M (terms × docs) → SVD: M = U Σ Vᵀ
  Take top-k left singular vectors (U[:, :k]) as term embeddings.
  New doc embedding = TF-IDF(doc) @ U[:, :k]  (project into latent space)
  Cosine similarity in this space ≈ semantic similarity.

  This is the same math as early Google LSI. No GPU, no API, no internet.
  Works well for a memory store with < 100k documents.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import Counter, OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


# ── Tokenizer (shared with BM25) ──────────────────────────────────────────

_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","is","was","are","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might",
    "this","that","these","those","i","you","he","she","it","we","they",
    "what","which","who","how","when","where","why","not","no","s","t",
}

def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 2 and w not in _STOP]


# ── LRU cache ─────────────────────────────────────────────────────────────

class _LRUCache:
    def __init__(self, maxsize: int = 4096):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, val: np.ndarray) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = val
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


# ── LSA Embedder ──────────────────────────────────────────────────────────

class LSAEmbedder:
    """
    Latent Semantic Analysis embedder.

    Pipeline:
      1. Build vocabulary from all indexed documents.
      2. Compute TF-IDF matrix (docs × terms).
      3. Truncated SVD: TF-IDF ≈ U Σ Vᵀ  (keep top `n_components` dims).
      4. Term matrix T = U[:, :k] × diag(Σ[:k])  — term embedding matrix.
      5. New doc → TF-IDF vector → @ T → L2-normalized embedding.

    Incremental indexing:
      Adding/removing docs triggers a re-fit when the corpus changes by
      more than `refit_threshold` fraction (default 20%).
      Between refits, new docs are projected using the existing term matrix.

    Thread-safety: read-write lock (writers wait for readers to finish).
    """

    DIM = 256  # Embedding dimensions. 256 is sweet spot: quality vs speed.

    def __init__(
        self,
        n_components: int = DIM,
        refit_threshold: float = 0.2,
    ):
        self.n_components    = n_components
        self.refit_threshold = refit_threshold

        # Corpus
        self._docs:   Dict[str, str]           = {}   # doc_id → text
        self._vocab:  Dict[str, int]           = {}   # term → index
        self._idf:    Optional[np.ndarray]     = None # shape (vocab_size,)
        self._T:      Optional[np.ndarray]     = None # shape (vocab_size, k) — term embedding matrix
        self._fitted  = False
        self._docs_at_last_fit = 0

        self._cache = _LRUCache()
        self._lock  = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────

    def index(self, doc_id: str, text: str) -> None:
        """Add or update a document."""
        with self._lock:
            self._docs[doc_id] = text
            self._cache = _LRUCache()  # invalidate on update

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            self._fitted = False

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a text string. Returns L2-normalized float32 vector of dim `n_components`.
        If corpus too small (<= 3 docs), returns a TF-IDF hash vector as fallback.
        """
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._lock:
            self._maybe_refit()
            vec = self._project(text)

        self._cache.set(cache_key, vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        return self.n_components

    @property
    def is_fitted(self) -> bool:
        return self._fitted and self._T is not None

    @property
    def corpus_size(self) -> int:
        return len(self._docs)

    # ── Internal ──────────────────────────────────────────────────────────

    def _maybe_refit(self) -> None:
        """Re-fit if corpus changed enough, or not yet fitted."""
        n = len(self._docs)
        if n < 4:
            return  # fallback to hash-based
        delta = abs(n - self._docs_at_last_fit) / max(1, self._docs_at_last_fit)
        if not self._fitted or delta >= self.refit_threshold:
            self._fit()

    def _fit(self) -> None:
        """Full TF-IDF → SVD pipeline."""
        docs     = list(self._docs.values())
        doc_ids  = list(self._docs.keys())
        n_docs   = len(docs)
        if n_docs < 4:
            return

        # Step 1: Tokenize all docs, build vocabulary
        tokenized = [_tokenize(d) for d in docs]
        all_terms = sorted({t for toks in tokenized for t in toks})
        vocab     = {t: i for i, t in enumerate(all_terms)}
        V         = len(vocab)
        if V < 10:
            return

        # Step 2: TF matrix (docs × terms)
        rows, cols, data = [], [], []
        for di, toks in enumerate(tokenized):
            counts = Counter(toks)
            total  = max(1, len(toks))
            for term, cnt in counts.items():
                if term in vocab:
                    rows.append(di)
                    cols.append(vocab[term])
                    data.append(cnt / total)   # TF

        tf_matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, V), dtype=np.float32)

        # Step 3: IDF weights
        df  = np.array((tf_matrix > 0).sum(axis=0), dtype=np.float32).flatten()
        idf = np.log(1 + (n_docs + 1) / (df + 1)).astype(np.float32)

        # TF-IDF
        tfidf = tf_matrix.multiply(idf)   # broadcast IDF across rows

        # Step 4: Truncated SVD
        k = min(self.n_components, n_docs - 1, V - 1)
        if k < 2:
            return

        try:
            U, S, Vt = svds(tfidf.T.astype(np.float64), k=k)
            # svds returns in ascending order — reverse
            U  = U[:, ::-1].astype(np.float32)
            S  = S[::-1].astype(np.float32)
        except Exception:
            return

        # Term embedding matrix T: project a TF-IDF vector into latent space
        # T = U * diag(S)  shape: (V, k)
        self._T      = U * S[np.newaxis, :]
        self._vocab  = vocab
        self._idf    = idf
        self._fitted = True
        self._docs_at_last_fit = n_docs

    def _project(self, text: str) -> np.ndarray:
        """Project a text into the latent embedding space."""
        # Fallback: hash-based vector when corpus too small
        if not self._fitted or self._T is None:
            return self._hash_embed(text)

        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(self.n_components, dtype=np.float32)

        counts = Counter(tokens)
        total  = max(1, len(tokens))
        V      = len(self._vocab)

        # Build sparse TF-IDF vector
        tfidf_vec = np.zeros(V, dtype=np.float32)
        for term, cnt in counts.items():
            idx = self._vocab.get(term)
            if idx is not None:
                tf = cnt / total
                tfidf_vec[idx] = tf * self._idf[idx]

        # Project: tfidf_vec @ T  →  shape (k,)
        embedded = tfidf_vec @ self._T   # (V,) @ (V, k) → (k,)

        # Pad to n_components if SVD returned fewer dims
        if len(embedded) < self.n_components:
            pad = np.zeros(self.n_components, dtype=np.float32)
            pad[:len(embedded)] = embedded
            embedded = pad

        # L2 normalize
        norm = np.linalg.norm(embedded)
        if norm > 1e-9:
            embedded = embedded / norm

        return embedded.astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        Fallback embedding for small corpora.
        Deterministic hash-based sparse vector.
        Better than zeros — preserves some term signal.
        """
        tokens = _tokenize(text)
        vec = np.zeros(self.n_components, dtype=np.float32)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx  = h % self.n_components
            sign = 1 if (h >> 128) & 1 else -1
            vec[idx] += sign * math.log1p(1)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def __repr__(self) -> str:
        return (f"LSAEmbedder(dim={self.n_components}, "
                f"docs={self.corpus_size}, fitted={self.is_fitted})")


# ── OpenAI Embedder ───────────────────────────────────────────────────────

class OpenAIEmbedder:
    """
    Calls OpenAI text-embedding-3-small (or -large).
    Drop-in replacement for LSAEmbedder in production.

    Usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        vec = embedder.embed("Andrii is building AMP")
    """

    MODELS = {
        "small": ("text-embedding-3-small", 1536),
        "large": ("text-embedding-3-large", 3072),
    }

    def __init__(
        self,
        api_key: str,
        model:   str = "small",
        base_url: str = "https://api.openai.com/v1",
    ):
        self._api_key = api_key
        self._model, self._dim = self.MODELS.get(model, self.MODELS["small"])
        self._base_url = base_url
        self._cache = _LRUCache()

    def embed(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        vec = self._call_api([text])[0]
        self._cache.set(key, vec)
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs  = self._call_api(batch)
            results.extend(vecs)
        return results

    def _call_api(self, texts: List[str]) -> List[np.ndarray]:
        import urllib.request
        payload = json.dumps({
            "model": self._model,
            "input": texts,
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        vecs = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            vecs.append(vec)
        return vecs

    @property
    def dim(self) -> int:
        return self._dim

    def index(self, doc_id: str, text: str) -> None:
        pass  # OpenAI doesn't need pre-indexing

    def remove(self, doc_id: str) -> None:
        pass

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self._model}, dim={self._dim})"


# ── Embedding Engine (facade) ─────────────────────────────────────────────

class EmbeddingEngine:
    """
    Unified embedding engine for AMP.

    Auto-selects backend:
      - If OPENAI_API_KEY is set → OpenAIEmbedder
      - Otherwise → LSAEmbedder (offline, no API calls)

    Pluggable:
        engine = EmbeddingEngine()
        engine.set_embedder(MyCustomEmbedder())

    Thread-safe. Handles batching and caching transparently.
    """

    def __init__(
        self,
        api_key:      Optional[str] = None,
        prefer_local: bool          = False,
        n_components: int           = 256,
    ):
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY")

        if key and not prefer_local:
            self._embedder = OpenAIEmbedder(api_key=key)
            self._mode     = "openai"
        else:
            self._embedder = LSAEmbedder(n_components=n_components)
            self._mode     = "lsa"

    def set_embedder(self, embedder) -> None:
        """Plug in any embedder with .embed(text) → np.ndarray."""
        self._embedder = embedder

    def embed(self, text: str) -> np.ndarray:
        return self._embedder.embed(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self._embedder.embed_batch(texts)

    def index(self, doc_id: str, text: str) -> None:
        """For LSA: add doc to corpus for re-fit."""
        if hasattr(self._embedder, "index"):
            self._embedder.index(doc_id, text)

    def remove(self, doc_id: str) -> None:
        if hasattr(self._embedder, "remove"):
            self._embedder.remove(doc_id)

    @property
    def dim(self) -> int:
        return self._embedder.dim

    @property
    def mode(self) -> str:
        return self._mode

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors (fast path)."""
        # Assumes both are already L2-normalized → dot product = cosine
        dot = float(np.dot(a, b))
        return max(-1.0, min(1.0, dot))

    def __repr__(self) -> str:
        return f"EmbeddingEngine(mode={self._mode}, embedder={self._embedder})"
