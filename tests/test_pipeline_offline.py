"""End-to-end runs using the offline backends + failure-mode behaviour of the runner and CLI."""
import pytest

import main as cli
from agent.config import store_from_cfg
from common.config import load_config
from common.llm import LLMTimeoutError, LLMUnavailableError
from evals import db
from evals.runner import EvalRunInvalid, run_eval


class Garbage:
    model = "garbage"

    def chat(self, messages, **kw):
        return "I refuse to produce JSON."


class Timeouts:
    model = "timeouts"

    def chat(self, messages, **kw):
        raise LLMTimeoutError("simulated timeout")


def test_full_eval_run_persists_everything(offline_cfg):
    cur = store_from_cfg(offline_cfg).current()
    s = run_eval(offline_cfg, cur, show_progress=False)
    m = s.metrics
    assert m["n"] == 40 and s.valid and m["n_errors"] == 0
    assert all(0.0 <= m[k] <= 1.0 for k in ("accuracy", "hallucination_rate", "quality", "composite"))
    path = offline_cfg.resolve("paths.db")
    run = db.get_run(path, s.run_id)
    assert run["status"] == "complete" and run["agent_version"] == 1 and run["n_cases"] == 40
    rows = db.get_results(path, s.run_id)
    assert len(rows) == 40 and {r["category"] for r in rows} == {"factual", "multi-hop", "math", "unanswerable"}
    r0 = rows[0]
    for key in ("answer", "contexts", "tool_calls", "latency_s", "fuzzy", "emb_sim", "verdict", "quality", "agent_version"):
        assert key in r0
    # failed flag is consistent with its reasons
    assert all(bool(r["failed"]) == bool(r["failure_reasons"]) for r in rows)
    # unanswerable cases are judged by the abstention rule, not by the LLM judge
    assert all(r["verdict_source"] == "rule" for r in rows if r["category"] == "unanswerable")
    assert abs(run["accuracy"] - sum(r["correct"] for r in rows) / 40) < 1e-9


def test_limit_run_is_partial_and_not_used_as_baseline(offline_cfg):
    cur = store_from_cfg(offline_cfg).current()
    s = run_eval(offline_cfg, cur, limit=5, show_progress=False)
    path = offline_cfg.resolve("paths.db")
    assert s.metrics["n"] == 5 and "partial" in db.get_run(path, s.run_id)["notes"]
    assert db.latest_run(path, version=1) is None


def test_judge_returning_garbage_falls_back_to_heuristic(offline_cfg):
    cur = store_from_cfg(offline_cfg).current()
    s = run_eval(offline_cfg, cur, limit=16, judge_llm=Garbage(), show_progress=False)
    rows = db.get_results(offline_cfg.resolve("paths.db"), s.run_id)
    assert any(r["verdict_source"] == "fallback" for r in rows)
    assert "judge failure" in db.get_run(offline_cfg.resolve("paths.db"), s.run_id)["notes"]
    assert s.metrics["n"] == 16 and s.valid          # the run still completes


def test_agent_timeouts_mark_run_invalid(offline_cfg):
    cur = store_from_cfg(offline_cfg).current()
    s = run_eval(offline_cfg, cur, limit=8, llm=Timeouts(), show_progress=False)
    assert s.metrics["n_errors"] == 8 and not s.valid
    rows = db.get_results(offline_cfg.resolve("paths.db"), s.run_id)
    assert all(r["failed"] and "agent error" in r["failure_reasons"] for r in rows)
    with pytest.raises(EvalRunInvalid):
        run_eval(offline_cfg, cur, limit=4, llm=Timeouts(), show_progress=False, raise_if_invalid=True)


@pytest.fixture()
def down_cfg(tmp_path, offline_cfg):
    p = tmp_path / "down.yaml"
    p.write_text(f"extends: {offline_cfg['_path']}\nllm:\n  backend: ollama\n  host: http://127.0.0.1:1\n"
                 f"  max_retries: 1\n  backoff_s: 0\n")
    return load_config(p)


def test_eval_fails_fast_when_ollama_is_down(down_cfg):
    cur = store_from_cfg(down_cfg).current()
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        run_eval(down_cfg, cur, show_progress=False)


def test_cli_exit_codes_and_messages(down_cfg, offline_cfg, capsys):
    assert cli.main(["--config", offline_cfg["_path"], "setup"]) == 0
    assert "Setup complete" in capsys.readouterr().out
    assert cli.main(["--config", down_cfg["_path"], "setup"]) == 2
    assert "ollama serve" in capsys.readouterr().out.lower()
    assert cli.main(["--config", down_cfg["_path"], "eval", "--no-progress"]) == 2
    assert "ERROR" in capsys.readouterr().err
    assert cli.main(["--config", offline_cfg["_path"], "ask", "How many approvals does a pull request need?"]) == 0
    assert "Answer (v1)" in capsys.readouterr().out
