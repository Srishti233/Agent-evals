"""Hallucination detection: split answer into claims, test each against retrieved context."""
from __future__ import annotations

import re
from typing import Any

from evals.metrics import split_claims
from evals.nli import ClaimVerdict, NLIChecker

_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Normalised numbers in text ('1,200' -> '1200', '2.50' -> '2.5')."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    out = set()
    for m in _NUM.findall(text):
        out.add(m.rstrip("0").rstrip(".") if "." in m else m.lstrip("0") or "0")
    return out


def build_premises(contexts: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return (retrieved chunk texts + calculator facts, calculator result strings)."""
    premises = [c["text"] for c in contexts]
    calc_results: list[str] = []
    for tc in tool_calls:
        if tc.get("tool") == "calculator" and not str(tc.get("output", "")).startswith("Error"):
            premises.append(f"Calculation: {tc['input']} = {tc['output']}")
            calc_results.append(str(tc["output"]))
    return premises, calc_results


def _tool_grounded(claim: str, premises: list[str], calc_results: list[str]) -> bool:
    """A numeric claim is supported if it contains a calculator result and every number in it
    appears in the retrieved text or tool results (arithmetic can't be 'entailed' by NLI models)."""
    if not calc_results:
        return False
    claim_nums = numbers_in(claim)
    result_nums = set().union(*(numbers_in(r) for r in calc_results))
    if not claim_nums or not (claim_nums & result_nums):
        return False
    known = set().union(*(numbers_in(p) for p in premises)) | result_nums
    return claim_nums <= known


def detect_hallucinations(answer: str, contexts: list[dict[str, Any]], tool_calls: list[dict[str, Any]],
                          checker: NLIChecker) -> dict[str, Any]:
    """Return {claims, n_claims, unsupported, n_unsupported, hallucinated, verdicts}."""
    claims = split_claims(answer)
    premises, calc_results = build_premises(contexts, tool_calls)
    verdicts: list[ClaimVerdict] = checker.check(claims, premises)
    for v in verdicts:
        if not v.entailed and _tool_grounded(v.claim, premises, calc_results):
            v.entailed, v.supported_by = True, "tool"
    unsupported = [v.claim for v in verdicts if not v.entailed]
    return {"claims": claims, "n_claims": len(claims), "unsupported": unsupported,
            "n_unsupported": len(unsupported), "hallucinated": bool(unsupported),
            "verdicts": [{"claim": v.claim, "entailed": v.entailed, "score": v.score,
                          "supported_by": v.supported_by} for v in verdicts]}
