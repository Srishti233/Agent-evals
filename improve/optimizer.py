"""Optimizer: asks the local LLM to analyse failures and propose a revised prompt (+ allowed knobs)."""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from agent.config import AgentConfig
from common.llm import LLM, chat_json

log = logging.getLogger(__name__)

OPTIMIZER_PROMPT = """You are an expert prompt engineer improving a tool-using question-answering agent that answers questions from a company knowledge base (handbook and product docs).

CURRENT SYSTEM PROMPT:
\"\"\"{prompt}\"\"\"

CURRENT SETTINGS: top_k={top_k}, temperature={temperature}
CURRENT METRICS: accuracy {acc:.0%}, hallucination rate {hall:.0%} (lower is better), quality {qual:.0%}

The agent's tools (search_docs, calculator) and its JSON action format are added automatically and cannot be changed. You may only rewrite the behavioural instructions in the system prompt.

FAILED CASES: {n_failed} of {n_total}
Failure patterns: {patterns}
Examples:
{examples}

ALLOWED SETTINGS: top_k in {grid_k}; temperature in {grid_t}

Your task:
1. Briefly analyse what went wrong across the failures.
2. Write an improved system prompt (at most 120 words). Make it general: do NOT copy specific questions, names or answers from the examples into it.
3. Optionally change top_k and/or temperature, choosing only from the allowed values.

Reply with ONLY this JSON object:
{{"analysis": "<2-3 sentences>", "system_prompt": "<the full new system prompt>", "top_k": <int>, "temperature": <float>, "rationale": "<1-2 sentences on why these changes should help>"}}"""


@dataclass
class Proposal:
    system_prompt: str
    top_k: int
    temperature: float
    analysis: str
    rationale: str


def _clip(text: str, n: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def summarize_failures(failed: list[dict[str, Any]], max_examples: int = 8) -> tuple[str, str]:
    """Return (failure-pattern counts, formatted examples), sampled round-robin across categories."""
    counts: Counter[str] = Counter()
    for r in failed:
        for reason in str(r.get("failure_reasons", "")).split("; "):
            if reason:
                counts[re.sub(r"\d+", "N", reason.split(":")[0].split("(")[0]).strip()] += 1
    patterns = "; ".join(f"{k} x{v}" for k, v in counts.most_common()) or "none"

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in failed:
        by_cat[r["category"]].append(r)
    picked: list[dict[str, Any]] = []
    while len(picked) < max_examples and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(picked) < max_examples:
                picked.append(by_cat[cat].pop(0))
    lines = [f"{i}. [{r['category']}] Q: {_clip(r['question'])} | Agent: {_clip(r['answer'] or '(none)')} | "
             f"Gold: {_clip(r['gold_answer'])} | Problem: {_clip(r['failure_reasons'])}"
             for i, r in enumerate(picked, 1)]
    return patterns, "\n".join(lines)


def _snap(value: Any, grid: list[float], default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(grid, key=lambda g: abs(g - v))


def propose(llm: LLM, current: AgentConfig, metrics: dict[str, Any], failed: list[dict[str, Any]],
            n_total: int, grid: dict[str, list[float]], max_examples: int = 8, retries: int = 3) -> Proposal:
    """Ask the LLM for a revised prompt. Raises LLMBadOutputError if it never produces a valid proposal."""
    patterns, examples = summarize_failures(failed, max_examples)
    prompt = OPTIMIZER_PROMPT.format(
        prompt=current.system_prompt, top_k=current.top_k, temperature=current.temperature,
        acc=metrics["accuracy"], hall=metrics["hallucination_rate"], qual=metrics["quality"],
        n_failed=len(failed), n_total=n_total, patterns=patterns, examples=examples or "(none)",
        grid_k=list(grid["top_k"]), grid_t=list(grid["temperature"]))
    norm = lambda s: " ".join(str(s).split())  # noqa: E731

    def validate(obj: dict[str, Any]) -> dict[str, Any]:
        sp = norm(obj.get("system_prompt", ""))
        if not 30 <= len(sp) <= 1200:
            raise ValueError("'system_prompt' must be a string of 30-1200 characters")
        top_k = int(_snap(obj.get("top_k", current.top_k), [float(k) for k in grid["top_k"]], current.top_k))
        temp = _snap(obj.get("temperature", current.temperature), [float(t) for t in grid["temperature"]],
                     current.temperature)
        if sp == norm(current.system_prompt) and top_k == current.top_k and temp == current.temperature:
            raise ValueError("proposal is identical to the current configuration; propose a real change")
        return {"system_prompt": sp, "top_k": top_k, "temperature": temp,
                "analysis": norm(obj.get("analysis", ""))[:600], "rationale": norm(obj.get("rationale", ""))[:600]}

    out = chat_json(llm, [{"role": "user", "content": prompt}], validate, temperature=0.4,
                    max_tokens=700, retries=retries)
    return Proposal(**out)
