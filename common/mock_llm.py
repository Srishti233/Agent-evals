"""Deterministic offline stand-in for the LLM.

PURPOSE: exercise the *plumbing* (agent loop, judge JSON, optimizer, accept/rollback, DB,
dashboard) on machines with no Ollama/GPU, e.g. CI. It is NOT a language model and the
metrics it produces are not meaningful.
"""
from __future__ import annotations

import json
import re

STOP = set("a an the of in on at to for is are was were be by and or with what which who whom how many "
           "much does do did it its this that from as per can will would should if has have had your their "
           "his her they them you i we our us not no than then there these those".split())

_RULES = [
    "Never guess: if the search results do not clearly contain the answer, reply exactly: I don't know.",
    "State exact numbers, names and units from the search results, and answer in a single short sentence.",
    "When a question needs several facts, search again with a more specific query before answering.",
]


def _toks(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in STOP]


class MockLLM:
    model = "mock"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
             json_mode: bool = False, max_tokens: int = 512) -> str:
        first = messages[0]["content"] if messages else ""
        last = messages[-1]["content"] if messages else ""
        if "ACTION PROTOCOL" in first:
            return self._agent(messages)
        if "strict grader" in last:
            return self._grade(last)
        if "scoring an AI assistant" in last:
            return self._rate(last)
        if "prompt engineer" in last:
            return self._optimize(last)
        return "{}"

    # -- agent ---------------------------------------------------------------------
    def _agent(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        question = next(m["content"] for m in messages if m["role"] == "user")
        n_assistant = sum(m["role"] == "assistant" for m in messages)
        if n_assistant == 0:
            return json.dumps({"thought": "look it up", "action": "search_docs", "input": question})
        obs = messages[-1]["content"]
        passages = [ln for ln in obs.splitlines() if ln.startswith("[")]
        qt = set(_toks(question))
        best, best_ov = "", 0.0
        for p in passages:
            body = re.sub(r"^\[[^\]]+\]\s*", "", p)
            for sent in re.split(r"(?<=[.!?])\s+", body):
                ov = len(qt & set(_toks(sent))) / max(1, len(qt))
                if ov > best_ov:
                    best, best_ov = sent, ov
        threshold = 0.5 if "Never guess" in system else 0.3
        answer = best if best_ov >= threshold else "I don't know."
        return json.dumps({"thought": "answer from passages", "action": "final_answer", "input": answer})

    # -- judge ---------------------------------------------------------------------
    @staticmethod
    def _field(prompt: str, name: str) -> str:
        m = re.search(rf"{name}:\s*(.*?)(?:\n[A-Z ]+:|\Z)", prompt, re.S)
        return m.group(1).strip() if m else ""

    def _grade(self, prompt: str) -> str:
        gold, ans = set(_toks(self._field(prompt, "GOLD ANSWER"))), set(_toks(self._field(prompt, "AGENT ANSWER")))
        recall = len(gold & ans) / max(1, len(gold))
        ok = recall >= 0.6
        return json.dumps({"verdict": "correct" if ok else "incorrect",
                           "reason": f"mock grader: {recall:.0%} of gold tokens present"})

    def _rate(self, prompt: str) -> str:
        q, ans = set(_toks(self._field(prompt, "QUESTION"))), set(_toks(self._field(prompt, "AGENT ANSWER")))
        gold = set(_toks(self._field(prompt, "GOLD ANSWER")))
        rel = 5 if q & ans else 3
        recall = len(gold & ans) / max(1, len(gold))
        comp = 5 if recall >= 0.6 else 3 if recall >= 0.3 else 2
        return json.dumps({"relevance": rel, "completeness": comp, "clarity": 4})

    # -- optimizer -----------------------------------------------------------------
    def _optimize(self, prompt: str) -> str:
        m = re.search(r'CURRENT SYSTEM PROMPT:\s*"""(.*?)"""', prompt, re.S)
        current = m.group(1).strip() if m else ""
        k = re.search(r"top_k=(\d+), temperature=([\d.]+)", prompt)
        top_k, temp = (int(k.group(1)), float(k.group(2))) if k else (3, 0.2)
        new = current
        for rule in _RULES:
            if rule not in current:
                new = f"{current} {rule}"
                break
        else:
            new = current + " Be concise."
        return json.dumps({
            "analysis": "Mock analysis: failures are dominated by wrong or unsupported answers.",
            "system_prompt": new,
            "top_k": 4 if top_k == 3 else 3,
            "temperature": temp,
            "rationale": "Mock optimizer appends the next hardening rule and nudges top_k.",
        })
