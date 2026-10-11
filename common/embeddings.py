"""Embedding backends: sentence-transformers (default) and a hashing stand-in for offline tests."""
from __future__ import annotations

import logging
import re
import threading
import zlib
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer
        self.device = resolve_device(device)
        log.info("loading embedding model %s on %s", model_name, self.device)
        self._model = SentenceTransformer(model_name, device=self.device)
        self.name = "st-" + model_name.split("/")[-1]
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        with self._lock:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                      convert_to_numpy=True, batch_size=32)
        return np.asarray(vecs, dtype="float32")


_HASH_STOP = frozenset("a an the of in on at to for is are was were be by and or with what which who whom how many "
                       "much does do did it its this that from as per can will would should if has have had your their "
                       "his her they them you i we our us not no than then there these those".split())


class HashEmbedder:
    """Hashing bag-of-words (stopwords removed, light plural stemming, sublinear tf) into 2048 dims.

    Offline stand-in for CI/smoke tests: lexical overlap only, no semantic generalisation."""

    name = "hash"
    dim = 2048

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
        toks = [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in _HASH_STOP]
        return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        out = np.where(out > 0, 1.0 + np.log(np.maximum(out, 1.0)), 0.0).astype("float32")
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


_EMBEDDERS: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def get_embedder(cfg: Any) -> Embedder:
    """Process-wide cached embedder built from config."""
    c = cfg.embeddings
    key = (c.backend, c.get("model", ""))
    with _LOCK:
        if key not in _EMBEDDERS:
            if c.backend == "hash":
                _EMBEDDERS[key] = HashEmbedder()
            elif c.backend == "sentence-transformers":
                _EMBEDDERS[key] = SentenceTransformerEmbedder(c.model, c.get("device", "auto"))
            else:
                raise ValueError(f"Unknown embeddings.backend: {c.backend!r}")
        return _EMBEDDERS[key]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (already normalised or not) vectors."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
