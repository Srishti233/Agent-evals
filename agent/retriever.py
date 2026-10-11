"""Chunking + local vector index (FAISS flat inner-product; numpy fallback)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)

try:  # FAISS is optional at runtime; numpy brute force is equivalent for this scale.
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def load_docs(docs_dir: Path) -> dict[str, str]:
    """Read all .md/.txt files; doc id = filename stem."""
    docs: dict[str, str] = {}
    for p in sorted(Path(docs_dir).glob("*")):
        if p.suffix.lower() in {".md", ".txt"} and p.is_file():
            docs[p.stem] = p.read_text(encoding="utf-8").strip()
    if not docs:
        raise FileNotFoundError(f"No .md/.txt documents found in {docs_dir}")
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split into windows of `chunk_size` words with `overlap` words shared."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = max(1, chunk_size - max(0, min(overlap, chunk_size - 1)))
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return out


def build_chunks(docs: dict[str, str], chunk_size: int, overlap_ratio: float) -> list[Chunk]:
    overlap = int(chunk_size * overlap_ratio)
    return [Chunk(doc_id, i, t) for doc_id, text in docs.items()
            for i, t in enumerate(chunk_text(text, chunk_size, overlap))]


def fingerprint(docs: dict[str, str], chunk_size: int, overlap_ratio: float) -> str:
    h = hashlib.sha1(f"{chunk_size}|{overlap_ratio}".encode())
    for k in sorted(docs):
        h.update(k.encode())
        h.update(docs[k].encode())
    return h.hexdigest()


class Retriever:
    """Top-k semantic search over chunked docs. Index persisted under paths.index_dir."""

    def __init__(self, cfg: Any, chunk_size: int, embedder: Embedder | None = None) -> None:
        self.cfg = cfg
        self.chunk_size = int(chunk_size)
        self.overlap_ratio = float(cfg.agent.get("chunk_overlap_ratio", 0.2))
        self.embedder = embedder or get_embedder(cfg)
        self.docs_dir = cfg.resolve("paths.docs_dir")
        self.dir = cfg.resolve("paths.index_dir") / f"{self.embedder.name}_cs{self.chunk_size}"
        self.chunks: list[Chunk] = []
        self._vecs: np.ndarray | None = None
        self._index: Any = None
        self._lock = threading.Lock()

    # ---- build / load ----------------------------------------------------------------
    def build(self, force: bool = False) -> int:
        """Build (or load a fresh) index. Returns the number of chunks."""
        with self._lock:
            docs = load_docs(self.docs_dir)
            fp = fingerprint(docs, self.chunk_size, self.overlap_ratio)
            meta_p, vec_p = self.dir / "meta.json", self.dir / "vectors.npy"
            if not force and meta_p.exists() and vec_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == fp:
                    self.chunks = [Chunk(**c) for c in meta["chunks"]]
                    self._set_vectors(np.load(vec_p))
                    return len(self.chunks)
            self.chunks = build_chunks(docs, self.chunk_size, self.overlap_ratio)
            vecs = self.embedder.encode([c.text for c in self.chunks])
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(vec_p, vecs)
            meta_p.write_text(json.dumps({
                "fingerprint": fp, "chunk_size": self.chunk_size, "embedder": self.embedder.name,
                "chunks": [c.__dict__ for c in self.chunks]}, ensure_ascii=False), encoding="utf-8")
            self._set_vectors(vecs)
            log.info("built index %s: %d docs -> %d chunks", self.dir.name, len(docs), len(self.chunks))
            return len(self.chunks)

    def _set_vectors(self, vecs: np.ndarray) -> None:
        self._vecs = np.ascontiguousarray(vecs, dtype="float32")
        if faiss is not None:
            self._index = faiss.IndexFlatIP(self._vecs.shape[1])
            self._index.add(self._vecs)

    def ensure_ready(self) -> None:
        if self._vecs is None:
            self.build()

    # ---- query -----------------------------------------------------------------------
    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        self.ensure_ready()
        assert self._vecs is not None
        k = max(1, min(int(k), len(self.chunks)))
        q = self.embedder.encode([query]).astype("float32")
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            pairs = list(zip(ids[0].tolist(), scores[0].tolist()))
        else:
            sims = (self._vecs @ q[0])
            top = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in top]
        return [{"doc_id": self.chunks[i].doc_id, "chunk_id": self.chunks[i].chunk_id,
                 "text": self.chunks[i].text, "score": round(float(s), 4)} for i, s in pairs if i >= 0]


_RETRIEVERS: dict[str, Retriever] = {}
_LOCK = threading.Lock()


def get_retriever(cfg: Any, chunk_size: int) -> Retriever:
    """Process-wide cached retriever per (index dir, chunk size)."""
    emb = get_embedder(cfg)
    key = f"{cfg['_path']}|{emb.name}|{chunk_size}"
    with _LOCK:
        if key not in _RETRIEVERS:
            r = Retriever(cfg, chunk_size, emb)
            r.ensure_ready()
            _RETRIEVERS[key] = r
        return _RETRIEVERS[key]
