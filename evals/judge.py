"""LLM-as-judge: correctness verdict (strict JSON schema) and 1-5 quality rubric."""
from __future__ import annotations

import logging
from typing import Any

from common.llm import LLM, chat_json
from evals.metrics import quality_from_ratings

log = logging.getLogger(__name__)

VERDICT_PROMPT = """You are a strict grader. Decide whether the AGENT ANSWER is correct for the QUESTION, using the GOLD ANSWER as the reference.
- correct: it states the same key facts and numbers as the gold answer. Wording and formatting may differ (for example "$1,200" and "1200 dollars"). Extra correct detail is fine.
- incorrect: key facts or numbers are missing, different or contradicted, or the answer says it does not know.

Reply with ONLY this JSON object:
{{"verdict": "correct" or "incorrect", "reason": "<one short sentence>"}}

QUESTION: {question}
GOLD ANSWER: {gold}
AGENT ANSWER: {answer}"""

RUBRIC_PROMPT = """You are scoring an AI assistant's answer against a rubric. Give each criterion an integer from 1 (poor) to 5 (excellent):
- relevance: does the answer address the question that was asked?
- completeness: does it contain all the information needed, compared with the gold answer?
- clarity: is it clear, well phrased and concise?

Reply with ONLY this JSON object:
{{"relevance": <1-5>, "completeness": <1-5>, "clarity": <1-5>}}

QUESTION: {question}
GOLD ANSWER: {gold}
AGENT ANSWER: {answer}"""

UNANSWERABLE_GOLD = ("This question cannot be answered from the knowledge base. The ideal reply is a brief, "
                     "honest statement that the answer is not known or not available.")


def validate_verdict(obj: dict[str, Any]) -> dict[str, Any]:
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in {"correct", "incorrect"}:
        raise ValueError("'verdict' must be exactly \"correct\" or \"incorrect\"")
    return {"verdict": v, "reason": str(obj.get("reason", "")).strip()[:300]}


def validate_rubric(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, int] = {}
    for key in ("relevance", "completeness", "clarity"):
        try:
            raw = obj[key]
            val = int(round(float(raw)))
            if isinstance(raw, bool):
                raise TypeError
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{key}' must be an integer 1-5") from None
        if not 1 <= val <= 5:
            raise ValueError(f"'{key}' must be between 1 and 5, got {val}")
        out[key] = val
    return out


class Judge:
    def __init__(self, llm: LLM, max_json_retries: int = 3) -> None:
        self.llm = llm
        self.retries = max_json_retries

    def verdict(self, question: str, gold: str, answer: str) -> dict[str, Any]:
        prompt = VERDICT_PROMPT.format(question=question, gold=gold, answer=answer)
        return chat_json(self.llm, [{"role": "user", "content": prompt}], validate_verdict,
                         temperature=0.0, max_tokens=200, retries=self.retries)

    def rate(self, question: str, gold: str, answer: str) -> dict[str, Any]:
        """Return {relevance, completeness, clarity, quality(0..1)}."""
        prompt = RUBRIC_PROMPT.format(question=question, gold=gold, answer=answer or "(no answer)")
        r = chat_json(self.llm, [{"role": "user", "content": prompt}], validate_rubric,
                      temperature=0.0, max_tokens=120, retries=self.retries)
        r["quality"] = quality_from_ratings(r["relevance"], r["completeness"], r["clarity"])
        return r
