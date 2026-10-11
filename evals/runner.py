"""Runs the eval set against one agent version (parallel), scores it, stores results in SQLite."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from agent.agent import Agent
from agent.config import AgentConfig
from agent.retriever import get_retriever
from common.embeddings import Embedder, cosine, get_embedder
from common.llm import LLM, LLMError, LLMUnavailableError, get_llm
from evals import db
from evals.hallucination import detect_hallucinations
from evals.judge import UNANSWERABLE_GOLD, Judge
from evals.metrics import aggregate, fuzzy_score, heuristic_correct, is_abstention
from evals.nli import NLIChecker, get_nli

log = logging.getLogger(__name__)

ProgressCb = Callable[[int, int], None]


class EvalRunInvalid(RuntimeError):
    """Too many agent/infrastructure errors for the metrics to be trustworthy."""


@dataclass
class RunSummary:
    run_id: int
    version: int
    metrics: dict[str, Any]
    n_failed: int = 0
    duration_s: float = 0.0
    valid: bool = True
    rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def composite(self) -> float:
        return float(self.metrics["composite"])


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"id", "question", "gold_answer", "category", "supporting_doc_ids"}
    for i, c in enumerate(data):
        missing = required - set(c)
        if missing:
            raise ValueError(f"eval case #{i} missing fields: {sorted(missing)}")
    return data


def eval_set_hash(cases: list[dict[str, Any]]) -> str:
    return hashlib.sha1(json.dumps(cases, sort_keys=True).encode()).hexdigest()[:12]


def preflight(cfg: Any) -> None:
    """Fail fast with a helpful message if Ollama is down or a model is missing."""
    if cfg.llm.backend != "ollama":
        return
    from common.llm import OllamaLLM
    probe = get_llm(cfg, "agent")
    assert isinstance(probe, OllamaLLM)
    models = probe.list_models()  # raises LLMUnavailableError with guidance
    for needed in {cfg.llm.model, cfg.llm.judge_model}:
        if not OllamaLLM._has(models, needed):
            raise LLMError(f"Model '{needed}' is not pulled on {probe.host}. Run: python main.py setup")


def evaluate_case(case: dict[str, Any], agent: Agent, judge: Judge, nli: NLIChecker, embedder: Embedder,
                  quality_fail_threshold: float) -> dict[str, Any]:
    """Run the agent on one case and compute all signals. Never raises for model/judge failures."""
    cat, gold = case["category"], case["gold_answer"]
    res = agent.run(case["question"])
    answer = res.answer
    abstained = is_abstention(answer)

    fz = fuzzy_score(answer, gold)
    vecs = embedder.encode([answer or " ", gold])
    sim = round(max(0.0, cosine(vecs[0], vecs[1])), 4)

    # ---- accuracy ----
    judge_failed = False
    if res.error or not answer.strip():
        correct, source, verdict, reason = False, "rule", "incorrect", res.error or "empty answer"
    elif cat == "unanswerable":
        correct, source = abstained, "rule"
        verdict = "correct" if abstained else "incorrect"
        reason = "abstained as required" if abstained else "answered a question that has no answer"
    elif abstained:
        correct, source, verdict, reason = False, "rule", "incorrect", "abstained on an answerable question"
    else:
        try:
            v = judge.verdict(case["question"], gold, answer)
            correct, source, verdict, reason = v["verdict"] == "correct", "judge", v["verdict"], v["reason"]
        except LLMUnavailableError:
            raise
        except LLMError as exc:
            judge_failed = True
            correct, source, verdict = heuristic_correct(fz, sim), "fallback", "correct" if heuristic_correct(fz, sim) else "incorrect"
            reason = f"judge failed ({type(exc).__name__}); fell back to fuzzy+embedding heuristic"
            log.warning("%s: %s", case["id"], reason)

    # ---- hallucination ----
    hall = detect_hallucinations(answer, res.contexts, res.tool_calls, nli) if answer and not res.error else \
        {"n_claims": 0, "n_unsupported": 0, "unsupported": [], "hallucinated": False}

    # ---- quality ----
    q: dict[str, Any] = {"relevance": None, "completeness": None, "clarity": None, "quality": None}
    if not res.error:
        try:
            q = judge.rate(case["question"], gold if cat != "unanswerable" else UNANSWERABLE_GOLD, answer)
        except LLMUnavailableError:
            raise
        except LLMError as exc:
            judge_failed = True
            log.warning("%s: quality judge failed (%s)", case["id"], type(exc).__name__)

    # ---- retrieval recall (logged only) ----
    support = set(case.get("supporting_doc_ids") or [])
    recall = None
    if support:
        recall = round(len(support & {c["doc_id"] for c in res.contexts}) / len(support), 4)

    # ---- failure reasons ----
    reasons: list[str] = []
    if res.error:
        reasons.append(f"agent error: {res.error[:120]}")
    elif not correct:
        reasons.append("answered although it should abstain (unanswerable)" if cat == "unanswerable" else
                       "abstained on an answerable question" if abstained else "wrong answer")
    if hall["hallucinated"]:
        reasons.append(f"hallucinated: {hall['n_unsupported']} unsupported claim(s)")
    if q["quality"] is not None and q["quality"] < quality_fail_threshold:
        reasons.append(f"low quality ({q['quality']:.0%})")
    if res.truncated:
        reasons.append("step budget exhausted")

    return {
        "case_id": case["id"], "category": cat, "question": case["question"], "gold_answer": gold,
        "answer": answer, "abstained": abstained, "contexts": res.contexts, "tool_calls": res.tool_calls,
        "latency_s": res.latency_s, "steps": res.steps, "fuzzy": fz, "emb_sim": sim,
        "retrieval_recall": recall, "verdict": verdict, "verdict_source": source, "verdict_reason": reason,
        "correct": correct, "n_claims": hall["n_claims"], "n_unsupported": hall["n_unsupported"],
        "unsupported": hall["unsupported"], "hallucinated": hall["hallucinated"],
        "relevance": q["relevance"], "completeness": q["completeness"], "clarity": q["clarity"],
        "quality": q["quality"], "failed": bool(reasons), "failure_reasons": "; ".join(reasons),
        "error": res.error, "agent_version": agent.agent_cfg.version, "judge_failed": judge_failed,
    }


def run_eval(cfg: Any, agent_cfg: AgentConfig, *, limit: int | None = None, progress_cb: ProgressCb | None = None,
             show_progress: bool = True, llm: LLM | None = None, judge_llm: LLM | None = None,
             nli: NLIChecker | None = None, embedder: Embedder | None = None,
             raise_if_invalid: bool = False) -> RunSummary:
    """Evaluate one agent version on the eval set and persist everything to SQLite."""
    preflight(cfg)
    cases = load_eval_set(cfg.resolve("paths.eval_set"))
    full_hash = eval_set_hash(cases)
    if limit:
        cases = cases[:limit]
    db_path = cfg.resolve("paths.db")
    embedder = embedder or get_embedder(cfg)
    nli = nli or get_nli(cfg)
    llm = llm or get_llm(cfg, "agent")
    judge = Judge(judge_llm or get_llm(cfg, "judge"), int(cfg.llm.get("max_json_retries", 3)))
    agent = Agent(cfg, agent_cfg, llm=llm, retriever=get_retriever(cfg, agent_cfg.chunk_size))
    q_thresh = float(cfg.eval.get("quality_fail_threshold", 0.5))

    run_id = db.create_run(db_path, agent_version=agent_cfg.version,
                           eval_hash=full_hash, model=getattr(llm, "model", "?"),
                           judge_model=getattr(judge.llm, "model", "?"), agent_config=agent_cfg.to_dict())
    log.info("run %d: evaluating %s on %d cases", run_id, agent_cfg.name, len(cases))
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    workers = max(1, int(cfg.eval.get("workers", 4)))
    try:
        with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=workers) as pool, \
                tqdm(total=len(cases), desc=f"eval {agent_cfg.name}", disable=not show_progress, unit="q") as bar:
            futures = {pool.submit(evaluate_case, c, agent, judge, nli, embedder, q_thresh): c for c in cases}
            for fut in as_completed(futures):
                case = futures[fut]
                try:
                    row = fut.result()
                except LLMUnavailableError:
                    for f in futures:
                        f.cancel()
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
                    log.exception("case %s crashed", case["id"])
                    row = _crash_row(case, agent_cfg.version, exc)
                db.save_result(db_path, run_id, row)
                rows.append(row)
                bar.update(1)
                if progress_cb:
                    progress_cb(len(rows), len(cases))
    except BaseException:
        db.finish_run(db_path, run_id, aggregate(rows), status="failed", notes="aborted")
        raise

    metrics = aggregate(rows)
    n_judge_failed = sum(1 for r in rows if r.get("judge_failed"))
    partial = "partial (limit)" if limit and limit < len(load_eval_set(cfg.resolve("paths.eval_set"))) else ""
    notes = "; ".join(x for x in [partial, f"{n_judge_failed} judge failure(s)" if n_judge_failed else ""] if x)
    db.finish_run(db_path, run_id, metrics, notes=notes)
    valid = metrics["error_rate"] <= float(cfg.eval.get("max_error_rate", 0.2))
    if not valid:
        msg = (f"{metrics['n_errors']}/{metrics['n']} cases errored; metrics are not trustworthy "
               f"(is Ollama overloaded or down?)")
        log.error(msg)
        if raise_if_invalid:
            raise EvalRunInvalid(msg)
    return RunSummary(run_id, agent_cfg.version, metrics, sum(1 for r in rows if r["failed"]),
                      round(time.perf_counter() - t0, 1), valid, rows)


def _crash_row(case: dict[str, Any], version: int, exc: Exception) -> dict[str, Any]:
    return {"case_id": case["id"], "category": case["category"], "question": case["question"],
            "gold_answer": case["gold_answer"], "answer": "", "abstained": False, "contexts": [],
            "tool_calls": [], "latency_s": 0.0, "steps": 0, "fuzzy": 0.0, "emb_sim": 0.0,
            "retrieval_recall": None, "verdict": "incorrect", "verdict_source": "rule",
            "verdict_reason": "evaluation crashed", "correct": False, "n_claims": 0, "n_unsupported": 0,
            "unsupported": [], "hallucinated": False, "relevance": None, "completeness": None,
            "clarity": None, "quality": None, "failed": True,
            "failure_reasons": f"agent error: {type(exc).__name__}: {exc}"[:200],
            "error": f"{type(exc).__name__}: {exc}"[:300], "agent_version": version, "judge_failed": False}
