"""
Optional dense (embedding) channel backed by Gemini embeddings.

* Corpus vectors are computed once and cached on disk, keyed by
  (model, dimensionality, corpus fingerprint) - a KB edit invalidates the cache.
* The index builds in a background thread at startup; until it is ready the
  retriever silently runs on its lexical channels (graceful degradation).
* Query embeddings are LRU-cached; the retriever only asks for one when the
  lexical channels are unsure (adaptive escalation), so most turns pay nothing.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from backend.rag.corpus import Corpus

log = logging.getLogger("calorai.rag.dense")


class GeminiEmbedder:
    """Thin wrapper over ``google-genai`` embeddings with task-type hints."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dims: int = 768, batch: int = 100):
        from google import genai  # imported lazily: the offline stack must not need it

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model, self.dims, self.batch = model, dims, batch

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[i : i + self.batch],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dims),
            )
            out.extend(e.values for e in resp.embeddings)
        mat = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)  # non-3072-d Gemini vectors must be re-normalised

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")


class DenseIndex:
    def __init__(self, corpus: Corpus, embedder, cache_dir: Path | str = ".cache/dense", query_cache: int = 2048):
        self.corpus = corpus
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.matrix: np.ndarray | None = None
        self.status = "idle"  # idle | building | ready | failed
        self.error: str | None = None
        self._qcache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._qcache_size = query_cache
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.matrix is not None

    @property
    def cache_path(self) -> Path:
        model = getattr(self.embedder, "model", "custom").replace("/", "_")
        dims = getattr(self.embedder, "dims", 0)
        return self.cache_dir / f"{model}-{dims}-{self.corpus.fingerprint}.npy"

    def build(self) -> None:
        self.status = "building"
        try:
            path = self.cache_path
            if path.exists():
                matrix = np.load(path)
                if matrix.shape[0] != len(self.corpus):
                    raise ValueError("stale dense cache")
            else:
                matrix = self.embedder.embed_documents([d.dense_text for d in self.corpus.docs])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, matrix)
            self.matrix = matrix
            self.status = "ready"
            log.info("dense index ready: %s vectors (%s)", matrix.shape[0], path.name)
        except Exception as exc:  # network/quota errors must never take the app down
            self.status, self.error = "failed", str(exc)
            log.warning("dense index unavailable, continuing lexical-only: %s", exc)

    def build_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.build, name="dense-index", daemon=True)
        thread.start()
        return thread

    def embed_query(self, text: str) -> np.ndarray | None:
        key = text.strip().lower()
        with self._lock:
            if key in self._qcache:
                self._qcache.move_to_end(key)
                return self._qcache[key]
        try:
            vec = self.embedder.embed_queries([key])[0]
        except Exception as exc:
            log.warning("query embedding failed: %s", exc)
            return None
        with self._lock:
            self._qcache[key] = vec
            if len(self._qcache) > self._qcache_size:
                self._qcache.popitem(last=False)
        return vec

    def warm_queries(self, texts: list[str]) -> None:
        """Embed several queries in one round trip (used once per chat turn)."""
        todo = [t.strip().lower() for t in texts if t.strip() and t.strip().lower() not in self._qcache]
        if not todo or not self.ready:
            return
        try:
            vecs = self.embedder.embed_queries(todo)
        except Exception as exc:
            log.warning("batch query embedding failed: %s", exc)
            return
        with self._lock:
            for t, v in zip(todo, vecs, strict=True):
                self._qcache[t] = v

    def search(self, vec: np.ndarray, top_k: int = 25) -> list[tuple[int, float]]:
        if self.matrix is None:
            return []
        sims = self.matrix @ vec
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]
