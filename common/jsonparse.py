"""Tolerant JSON extraction for small-LLM output (code fences, chatter, trailing commas)."""
from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    """Raised when no JSON object can be recovered from text."""


def _loads_dict(s: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _repair(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object found in `text`, or raise JSONParseError."""
    if text is None or not text.strip():
        raise JSONParseError("empty response")
    s = text.strip()

    candidates = [s]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for cand in candidates:
        for variant in (cand, _repair(cand)):
            obj = _loads_dict(variant)
            if obj is not None:
                return obj

    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        chunk = s[m.start():]
        for variant in (chunk, _repair(chunk)):
            try:
                obj, _ = decoder.raw_decode(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    # last resort: outermost braces, repaired
    a, b = s.find("{"), s.rfind("}")
    if 0 <= a < b:
        obj = _loads_dict(_repair(s[a:b + 1]))
        if obj is not None:
            return obj
    raise JSONParseError(f"no valid JSON object in response: {s[:120]!r}")
