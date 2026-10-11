"""ReAct-style tool-using QA agent (JSON actions, max N steps)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.config import AgentConfig
from agent.retriever import Retriever, get_retriever
from agent.tools import calculator, search_docs
from common.llm import LLM, LLMError, chat_json, get_llm

log = logging.getLogger(__name__)

ACTIONS = ("search_docs", "calculator", "final_answer")

# Fixed protocol appended to every version's behavioural system prompt. The optimizer can only
# rewrite the behavioural part, so it can never break the tool/JSON contract.
PROTOCOL = """

ACTION PROTOCOL
On every turn reply with exactly ONE JSON object and nothing else:
{{"thought": "<brief reasoning>", "action": "<search_docs|calculator|final_answer>", "input": "<string>"}}

Actions:
- search_docs: input is a search query. You get back passages from the knowledge base, each prefixed with [doc_id].
- calculator: input is an arithmetic expression using numbers and + - * / ** and parentheses only (no units, no words).
- final_answer: input is your final answer to the user.

You may take at most {max_steps} steps in total, and the last step must be final_answer."""


@dataclass
class AgentResult:
    question: str
    answer: str
    contexts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    steps: int = 0
    version: int = 0
    truncated: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_action(obj: dict[str, Any]) -> dict[str, Any]:
    raw = str(obj.get("action", "")).strip().lower()
    action = next((a for a in ACTIONS if a in raw), None)
    if action is None:
        raise ValueError(f"'action' must be one of {list(ACTIONS)}")
    value = obj.get("input", obj.get("answer"))
    if isinstance(value, (dict, list)) or value is None or not str(value).strip():
        raise ValueError("'input' must be a non-empty string")
    return {"thought": str(obj.get("thought", ""))[:300], "action": action, "input": str(value).strip()}


class Agent:
    def __init__(self, cfg: Any, agent_cfg: AgentConfig, llm: LLM | None = None,
                 retriever: Retriever | None = None) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        self.llm = llm or get_llm(cfg, "agent")
        self.retriever = retriever or get_retriever(cfg, agent_cfg.chunk_size)
        self.max_steps = int(cfg.agent.get("max_steps", 4))

    @property
    def system_prompt(self) -> str:
        return self.agent_cfg.system_prompt.strip() + PROTOCOL.format(max_steps=self.max_steps)

    def run(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        res = AgentResult(question=question, answer="", version=self.agent_cfg.version)
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt},
                                          {"role": "user", "content": question}]
        seen: set[tuple[str, int]] = set()
        try:
            for step in range(1, self.max_steps + 1):
                res.steps = step
                if step == self.max_steps and step > 1:
                    messages[-1] = {**messages[-1], "content": messages[-1]["content"] +
                                    "\n\nThis is your LAST step: reply now with action final_answer."}
                act = chat_json(self.llm, messages, _validate_action,
                                temperature=self.agent_cfg.temperature,
                                max_tokens=int(self.cfg.agent.get("max_tokens", 400)),
                                retries=int(self.cfg.llm.get("max_json_retries", 3)))
                if act["action"] == "final_answer":
                    res.answer = act["input"]
                    break
                if act["action"] == "search_docs":
                    obs, hits = search_docs(self.retriever, act["input"], self.agent_cfg.top_k)
                    for h in hits:
                        key = (h["doc_id"], h["chunk_id"])
                        if key not in seen:
                            seen.add(key)
                            res.contexts.append(h)
                else:
                    obs = calculator(act["input"])
                res.tool_calls.append({"step": step, "tool": act["action"], "input": act["input"],
                                       "output": obs[:1500]})
                messages.append({"role": "assistant", "content": json.dumps(act)})
                messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            else:
                res.truncated = True
                res.answer = "I don't know."
                log.debug("step budget exhausted for %r", question)
        except LLMError as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            log.warning("agent error on %r: %s", question[:60], res.error)
        res.latency_s = round(time.perf_counter() - t0, 3)
        return res
