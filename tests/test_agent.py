import json

import pytest

from agent.agent import Agent
from agent.config import AgentConfig
from agent.retriever import Retriever, chunk_text
from common.llm import LLMUnavailableError


class Scripted:
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def act(action, inp):
    return {"thought": "t", "action": action, "input": inp}


@pytest.fixture()
def agent_cfg():
    return AgentConfig(version=1, system_prompt="Be helpful. Say I don't know if unsure.", top_k=3, chunk_size=120)


def test_chunking():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, 10, 2)
    assert chunks[0].split()[0] == "w0" and chunks[1].split()[0] == "w8"
    assert chunks[-1].split()[-1] == "w24"
    assert chunk_text("short text", 10, 2) == ["short text"]
    with pytest.raises(ValueError):
        chunk_text("x", 0, 0)


def test_retriever_finds_right_doc(offline_cfg):
    r = Retriever(offline_cfg, 120)
    assert r.build() == 18
    assert r.search("What is the minimum password length?", 3)[0]["doc_id"] == "12_security_policy"
    # lexical stand-in embedder: "Atlas" appears in several docs, so only require the right doc in top-3
    assert "02_product_atlas" in [h["doc_id"] for h in r.search("Atlas payload capacity", 3)]
    # second build reuses the persisted index; a different chunk size gets its own index dir
    r2 = Retriever(offline_cfg, 120)
    assert r2.build() == 18 and r2.dir == r.dir
    assert Retriever(offline_cfg, 30).build() > 18


def test_agent_search_then_answer(offline_cfg, agent_cfg):
    llm = Scripted([act("search_docs", "password length"), act("final_answer", "14 characters.")])
    res = Agent(offline_cfg, agent_cfg, llm=llm).run("What is the minimum password length?")
    assert res.answer == "14 characters." and res.steps == 2 and res.error is None
    assert res.tool_calls[0]["tool"] == "search_docs"
    assert any(c["doc_id"] == "12_security_policy" for c in res.contexts)
    assert "ACTION PROTOCOL" in llm.calls[0][0]["content"] and agent_cfg.system_prompt in llm.calls[0][0]["content"]


def test_agent_calculator_and_bad_json_recovery(offline_cfg, agent_cfg):
    llm = Scripted(["garbage", act("calculator", "24000 * 10"), act("final_answer", "$240,000")])
    res = Agent(offline_cfg, agent_cfg, llm=llm).run("cost of 10 Atlas?")
    assert res.answer == "$240,000"
    assert res.tool_calls[0]["output"] == "240000"


def test_agent_step_budget_exhausted_abstains(offline_cfg, agent_cfg):
    llm = Scripted([act("search_docs", f"q{i}") for i in range(4)])
    res = Agent(offline_cfg, agent_cfg, llm=llm).run("anything")
    assert res.truncated and res.answer == "I don't know." and res.steps == 4 and len(res.tool_calls) == 4
    assert "LAST step" in llm.calls[-1][-1]["content"]


def test_agent_records_llm_failure_instead_of_crashing(offline_cfg, agent_cfg):
    llm = Scripted([LLMUnavailableError("down")])
    res = Agent(offline_cfg, agent_cfg, llm=llm).run("x")
    assert res.answer == "" and "LLMUnavailableError" in res.error


def test_agent_invalid_action_is_retried(offline_cfg, agent_cfg):
    llm = Scripted([{"action": "launch_missiles", "input": "x"}, act("final_answer", "ok")])
    assert Agent(offline_cfg, agent_cfg, llm=llm).run("x").answer == "ok"
