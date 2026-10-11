"""Pure metric helpers (no model calls) so they are cheap to unit-test."""
from __future__ import annotations

import re
from typing import Any, Iterable

from rapidfuzz import fuzz

W_ACCURACY, W_NO_HALLUCINATION, W_QUALITY = 0.4, 0.4, 0.2

_ABSTAIN = re.compile(
    r"\b(i\s+(?:do\s*not|don't|dont)\s+know|cannot\s+find|can't\s+find|can\s*not\s+be\s+determined|"
    r"not\s+enough\s+information|insufficient\s+information|no\s+information|unable\s+to\s+(?:find|answer)|"
    r"(?:is|are)\s+not\s+(?:mentioned|provided|specified|available|stated)|"
    r"(?:does|do)\s*n[o']t\s+(?:say|specify|mention|contain|provide|state)|"
    r"(?:doesn't|don't)\s+(?:say|specify|mention|contain|provide|state)|"
    r"don't\s+have\s+(?:enough\s+)?information)\b", re.I)


def _norm_apos(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def is_abstention(answer: str) -> bool:
    """True if the answer says it doesn't know / can't find the information."""
    return bool(answer and _ABSTAIN.search(_norm_apos(answer)))


def split_claims(answer: str, min_words: int = 3) -> list[str]:
    """Split an answer into sentence-level claims, dropping questions, fragments and abstentions."""
    if not answer or not answer.strip():
        return []
    parts: list[str] = []
    for line in re.split(r"\n+", answer.strip()):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if line:
            parts.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[$])", line))
    claims = []
    for p in parts:
        p = p.strip()
        if not p or p.endswith("?") or len(p.split()) < min_words or is_abstention(p):
            continue
        claims.append(p)
    return claims


def normalize_text(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", _norm_apos(text).lower())
    return " ".join(re.findall(r"[a-z0-9.$%]+", text))


def fuzzy_score(answer: str, gold: str) -> float:
    """0..1 token-set fuzzy match (logged signal, not the accuracy verdict)."""
    if not answer or not gold:
        return 0.0
    return round(fuzz.token_set_ratio(normalize_text(answer), normalize_text(gold)) / 100.0, 4)


def quality_from_ratings(relevance: int, completeness: int, clarity: int) -> float:
    """Mean 1-5 rubric rating normalised to 0..1 (1 -> 0%, 5 -> 100%)."""
    mean = (relevance + completeness + clarity) / 3.0
    return round((mean - 1.0) / 4.0, 4)


def composite_score(accuracy: float, hallucination_rate: float, quality: float) -> float:
    """composite = 0.4*accuracy + 0.4*(1-hallucination) + 0.2*quality (all in 0..1)."""
    return round(W_ACCURACY * accuracy + W_NO_HALLUCINATION * (1.0 - hallucination_rate) + W_QUALITY * quality, 6)


def heuristic_correct(fuzzy: float, emb_sim: float) -> bool:
    """Fallback only when the LLM judge is unavailable."""
    return fuzzy >= 0.8 and emb_sim >= 0.7


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question rows into run-level metrics (all rates in 0..1)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "hallucination_rate": 0.0, "quality": 0.0, "composite": 0.0,
                "n_errors": 0, "error_rate": 0.0, "by_category": {}}
    acc = sum(1 for r in rows if r["correct"]) / n
    hall = sum(1 for r in rows if r["hallucinated"]) / n
    qvals = [r["quality"] for r in rows if r.get("quality") is not None]
    qual = _mean(qvals)
    errors = sum(1 for r in rows if r.get("error"))
    by_cat: dict[str, dict[str, float]] = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "accuracy": sum(1 for r in sub if r["correct"]) / len(sub),
            "hallucination_rate": sum(1 for r in sub if r["hallucinated"]) / len(sub),
            "quality": _mean(r["quality"] for r in sub if r.get("quality") is not None),
        }
    return {"n": n, "accuracy": acc, "hallucination_rate": hall, "quality": qual,
            "composite": composite_score(acc, hall, qual), "n_errors": errors,
            "error_rate": errors / n, "by_category": by_cat}
