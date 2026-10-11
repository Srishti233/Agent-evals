import pytest

import improve.loop as loop
from agent.config import AgentConfig, VersionStore, parse_version
from common.llm import LLMBadOutputError
from evals.metrics import composite_score
from evals.runner import RunSummary
from improve.loop import composite_improved, describe_changes, run_improve_loop
from improve.optimizer import Proposal, propose, summarize_failures


# ------------------------------------------------------------------ accept / rollback rule
def test_accept_rollback_decision():
    assert composite_improved(0.71, 0.70) is True
    assert composite_improved(0.70, 0.70) is False          # ties are rolled back
    assert composite_improved(0.69, 0.70) is False
    assert composite_improved(0.705, 0.70, min_delta=0.01) is False
    assert composite_improved(0.72, 0.70, min_delta=0.01) is True


def test_composite_matches_spec():
    assert composite_score(0.8, 0.1, 0.7) == pytest.approx(0.4 * 0.8 + 0.4 * 0.9 + 0.2 * 0.7)


# ------------------------------------------------------------------ version store
def test_version_store_roundtrip(tmp_path):
    s = VersionStore(tmp_path)
    v1 = s.ensure_baseline({"system_prompt": "a  b", "top_k": 3, "temperature": 0.2, "chunk_size": 100})
    assert v1.version == 1 and v1.system_prompt == "a b" and s.current_version() == 1
    s.save(AgentConfig(version=s.next_version(), system_prompt="new", top_k=4, parent=1, status="candidate"))
    assert s.list_versions() == [1, 2] and s.current_version() == 1   # saving does not move CURRENT
    s.set_current("v2")
    assert s.current().top_k == 4
    assert parse_version("v7") == 7 and parse_version(3) == 3
    with pytest.raises(ValueError):
        parse_version("x")
    with pytest.raises(FileNotFoundError):
        s.set_current(9)


def test_describe_changes():
    a = AgentConfig(1, "One. Two.", top_k=3, temperature=0.2)
    b = AgentConfig(2, "One. Two. Three.", top_k=4, temperature=0.2)
    d = describe_changes(a, b)
    assert "top_k 3 -> 4" in d and "+Three." in d and "temperature" not in d


# ------------------------------------------------------------------ optimizer
FAILED = [{"category": "math", "question": "q1", "answer": "x", "gold_answer": "y",
           "failure_reasons": "wrong answer; hallucinated: 2 unsupported claim(s)"},
          {"category": "unanswerable", "question": "q2", "answer": "z", "gold_answer": "I don't know.",
           "failure_reasons": "answered although it should abstain (unanswerable)"}]


def test_summarize_failures():
    patterns, ex = summarize_failures(FAILED, 5)
    assert "wrong answer x1" in patterns and "hallucinated x1" in patterns
    assert ex.count("\n") == 1 and "[math]" in ex and "[unanswerable]" in ex


class OptLLM:
    model = "opt"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, **kw):
        import json
        return json.dumps(self.replies.pop(0))


def test_propose_snaps_to_grid_and_rejects_noops():
    cur = AgentConfig(1, "x" * 40, top_k=3, temperature=0.2)
    grid = {"top_k": [2, 3, 4, 5], "temperature": [0.0, 0.2, 0.5]}
    metrics = {"accuracy": .5, "hallucination_rate": .2, "quality": .6}
    llm = OptLLM([
        {"system_prompt": "x" * 40, "top_k": 3, "temperature": 0.2},                   # no-op -> retried
        {"system_prompt": "short"},                                                    # too short -> retried
        {"system_prompt": "A better prompt that is long enough to pass.", "top_k": 99, "temperature": 0.4,
         "analysis": "a", "rationale": "r"}])
    p = propose(llm, cur, metrics, FAILED, 40, grid)
    assert p.top_k == 5            # 99 is off-grid -> snapped to the nearest allowed value
    assert p.temperature == 0.5    # 0.4 is off-grid -> nearest allowed value
    assert p.analysis == "a" and p.rationale == "r" and llm.replies == []   # no-op and too-short were retried


def test_propose_gives_up():
    cur = AgentConfig(1, "x" * 40)
    llm = OptLLM([{"nope": 1}] * 3)
    with pytest.raises(LLMBadOutputError):
        propose(llm, cur, {"accuracy": 0, "hallucination_rate": 0, "quality": 0}, FAILED, 40,
                {"top_k": [3], "temperature": [0.2]})


# ------------------------------------------------------------------ the loop with scripted scores
def _summary(version, composite, failed=True):
    rows = [{"case_id": "c1", "category": "factual", "question": "q", "answer": "a", "gold_answer": "g",
             "failed": failed, "failure_reasons": "wrong answer"}]
    m = {"accuracy": composite, "hallucination_rate": 0.0, "quality": composite, "composite": composite, "n": 1,
         "n_errors": 0}
    return RunSummary(run_id=version, version=version, metrics=m, n_failed=1, rows=rows, valid=True)


def _install_fakes(monkeypatch, scores, proposals=None):
    """scores: composite per candidate eval, in order. First entry of `baseline` handled separately."""
    it = iter(scores)
    n = {"props": 0}

    def fake_run_eval(cfg, agent_cfg, **kw):
        return _summary(agent_cfg.version, next(it))

    def fake_propose(llm, current, metrics, failed, n_total, grid, max_examples=8, retries=3):
        n["props"] += 1
        return Proposal(f"{current.system_prompt} Rule{n['props']}.", current.top_k, current.temperature,
                        f"analysis{n['props']}", f"why{n['props']}")

    monkeypatch.setattr(loop, "run_eval", fake_run_eval)
    monkeypatch.setattr(loop, "propose", fake_propose)
    monkeypatch.setattr(loop, "get_llm", lambda cfg, role="agent": object())
    monkeypatch.setattr(loop, "_baseline", lambda cfg, cur, cb: (_summary(1, 0.60).metrics, _summary(1, 0.60).rows, 1))


def test_loop_accepts_improvement_and_rolls_back_regression(offline_cfg, monkeypatch):
    _install_fakes(monkeypatch, [0.70, 0.65, 0.80])   # v2 better, v3 worse, v4 better
    res = run_improve_loop(offline_cfg, iterations=3, patience=5)
    assert [i.accepted for i in res.iterations] == [True, False, True]
    store = VersionStore(offline_cfg.resolve("paths.versions_dir"))
    assert store.list_versions() == [1, 2, 3, 4]
    assert [store.load(v).status for v in (1, 2, 3, 4)] == ["baseline", "accepted", "rejected", "accepted"]
    assert store.load(3).parent == 2 and store.load(4).parent == 2   # v4 branched from v2, not the rejected v3
    assert store.current_version() == 4 and res.final_version == 4
    assert res.final_metrics["composite"] == 0.80
    log = offline_cfg.resolve("paths.changelog").read_text()
    assert "ACCEPTED" in log and "ROLLED BACK" in log and "analysis1" in log and "why2" in log
    from evals import db
    ch = db.list_changelog(offline_cfg.resolve("paths.db"))
    assert [c["accepted"] for c in ch] == [1, 0, 1]


def test_loop_rollback_leaves_current_untouched(offline_cfg, monkeypatch):
    _install_fakes(monkeypatch, [0.50, 0.55])
    res = run_improve_loop(offline_cfg, iterations=2, patience=5)
    assert not any(i.accepted for i in res.iterations) and res.final_version == 1
    assert VersionStore(offline_cfg.resolve("paths.versions_dir")).current_version() == 1


def test_loop_stops_on_plateau(offline_cfg, monkeypatch):
    _install_fakes(monkeypatch, [0.6, 0.6, 0.6, 0.6, 0.6])
    res = run_improve_loop(offline_cfg, iterations=5, patience=2)
    assert len(res.iterations) == 2 and "plateau" in res.stopped_because


def test_loop_stops_when_nothing_fails(offline_cfg, monkeypatch):
    _install_fakes(monkeypatch, [])
    monkeypatch.setattr(loop, "_baseline", lambda cfg, cur, cb: (_summary(1, 0.9).metrics, [], 1))
    res = run_improve_loop(offline_cfg, iterations=3)
    assert res.iterations == [] and res.stopped_because == "no failed cases left"
