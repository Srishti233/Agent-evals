"""Claim-vs-context entailment checkers: HF cross-encoder NLI (default) and a lexical stand-in."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from common.embeddings import resolve_device

log = logging.getLogger(__name__)


@dataclass
class ClaimVerdict:
    claim: str
    entailed: bool
    score: float            # best entailment probability (or lexical support) over premises
    supported_by: str = ""  # "nli" | "tool" | ""


class NLIChecker(Protocol):
    def check(self, claims: list[str], premises: list[str]) -> list[ClaimVerdict]: ...


class TransformersNLI:
    """cross-encoder/nli-deberta-v3-small. Each claim is tested against every premise chunk
    individually and against all premises joined (for claims that combine two chunks); a claim is
    supported if the best entailment probability >= threshold."""

    def __init__(self, model_name: str, device: str = "auto", threshold: float = 0.5,
                 batch_size: int = 16, max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.device = resolve_device(device)
        log.info("loading NLI model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        id2label = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        ent = [i for i, l in id2label.items() if l.startswith("entail")]
        # cross-encoder NLI heads are [contradiction, entailment, neutral]
        self.entail_idx = ent[0] if ent else 1
        self.threshold, self.batch_size, self.max_length = threshold, batch_size, max_length
        self._lock = threading.Lock()

    def _entail_probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        with self._lock, torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch], padding=True,
                                     truncation="only_first", max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
                probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, self.entail_idx]
                out.append(probs.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def check(self, claims: list[str], premises: list[str]) -> list[ClaimVerdict]:
        if not claims:
            return []
        if not premises:
            return [ClaimVerdict(c, False, 0.0) for c in claims]
        sets = list(premises) + ([" ".join(premises)] if len(premises) > 1 else [])
        pairs = [(p, c) for c in claims for p in sets]
        probs = self._entail_probs(pairs).reshape(len(claims), len(sets))
        best = probs.max(axis=1)
        return [ClaimVerdict(c, bool(b >= self.threshold), round(float(b), 4), "nli")
                for c, b in zip(claims, best)]


_STOP = set("a an the of in on at to for is are was were be by and or with it its this that from as per "
            "can will would has have had their his her they them you your".split())


def content_tokens(text: str) -> set[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return {t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in _STOP}


class LexicalNLI:
    """Offline stand-in: a claim is 'entailed' if >= threshold of its content tokens occur in one
    premise (or in all premises together). Not a real NLI model."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def check(self, claims: list[str], premises: list[str]) -> list[ClaimVerdict]:
        ptoks = [content_tokens(p) for p in premises]
        joint = set().union(*ptoks) if ptoks else set()
        out = []
        for c in claims:
            ct = content_tokens(c)
            if not ct or not ptoks:
                out.append(ClaimVerdict(c, False, 0.0))
                continue
            best = max([len(ct & pt) / len(ct) for pt in ptoks] + [len(ct & joint) / len(ct)])
            out.append(ClaimVerdict(c, best >= self.threshold, round(best, 4), "nli"))
        return out


_CACHE: dict[tuple[str, str], NLIChecker] = {}
_LOCK = threading.Lock()


def get_nli(cfg: Any) -> NLIChecker:
    c = cfg.nli
    key = (c.backend, c.get("model", ""))
    with _LOCK:
        if key not in _CACHE:
            if c.backend == "lexical":
                _CACHE[key] = LexicalNLI()
            elif c.backend == "transformers":
                _CACHE[key] = TransformersNLI(c.model, c.get("device", "auto"), float(c.get("threshold", 0.5)),
                                              int(c.get("batch_size", 16)))
            else:
                raise ValueError(f"Unknown nli.backend: {c.backend!r}")
        return _CACHE[key]
