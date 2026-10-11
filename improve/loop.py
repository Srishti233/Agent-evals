"""Improve loop: evaluate -> analyse failures -> propose -> re-evaluate -> accept or roll back."""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.config import AgentConfig, VersionStore, store_from_cfg
from common.llm import LLMBadOutputError, LLMUnavailableError, get_llm
from evals import db
from evals.metrics import composite_score
from evals.runner import RunSummary, eval_set_hash, load_eval_set, run_eval
from improve.optimizer import Proposal, propose

log = logging.getLogger(__name__)


def composite_improved(candidate: float, best: float, min_delta: float = 0.0) -> bool:
    """Accept only if the candidate composite beats the best by MORE than min_delta."""
    return candidate > best + min_delta + 1e-9


def describe_changes(old: AgentConfig, new: AgentConfig) -> str:
    parts = []
    if old.top_k != new.top_k:
        parts.append(f"top_k {old.top_k} -> {new.top_k}")
    if old.temperature != new.temperature:
        parts.append(f"temperature {old.temperature} -> {new.temperature}")
    if " ".join(old.system_prompt.split()) != " ".join(new.system_prompt.split()):
        a = [s for s in old.system_prompt.replace(". ", ".\n").splitlines()]
        b = [s for s in new.system_prompt.replace(". ", ".\n").splitlines()]
        diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
        parts.append("system prompt changed:\n" + "\n".join(f"    {d}" for d in diff))
    return "\n".join(parts) or "no change"


@dataclass
class IterationResult:
    iteration: int
    base_version: int
    candidate_version: int | None
    accepted: bool
    base_composite: float
    candidate_composite: float | None
    changes: str
    reason: str

    @property
    def delta(self) -> float:
        return (self.candidate_composite - self.base_composite) if self.candidate_composite is not None else 0.0


@dataclass
class LoopResult:
    baseline: RunSummary | dict[str, Any]
    final_version: int
    final_metrics: dict[str, Any]
    iterations: list[IterationResult]
    stopped_because: str


def _baseline(cfg: Any, current: AgentConfig, progress_cb: Callable[[int, int], None] | None) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Return (metrics, result rows, run_id) of the current version, reusing a stored run if allowed."""
    db_path = cfg.resolve("paths.db")
    h = eval_set_hash(load_eval_set(cfg.resolve("paths.eval_set")))
    if cfg.improve.get("reuse_baseline", True):
        run = db.latest_run(db_path, version=current.version, eval_hash=h)
        if run:
            log.info("reusing stored baseline run %d for %s", run["run_id"], current.name)
            m = {k: run[k] for k in ("accuracy", "hallucination_rate", "quality", "composite")}
            m.update(n=run["n_cases"], n_errors=run["n_errors"] or 0)
            return m, db.get_results(db_path, run["run_id"]), run["run_id"]
    s = run_eval(cfg, current, progress_cb=progress_cb, raise_if_invalid=True)
    return s.metrics, s.rows, s.run_id


def _write_changelog(path: Path, it: IterationResult, cand: AgentConfig | None, analysis: str, rationale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Agent improvement changelog\n\nEach entry: what changed, why, and whether it was accepted.\n",
                        encoding="utf-8")
    verdict = "ACCEPTED" if it.accepted else "ROLLED BACK"
    cand_s = f"v{it.candidate_version}" if it.candidate_version else "(none)"
    lines = [
        f"\n## Iteration {it.iteration}: v{it.base_version} -> {cand_s} [{verdict}]",
        f"*{datetime.now(timezone.utc).isoformat(timespec='seconds')}*\n",
        f"- Composite: {it.base_composite:.4f} -> "
        + (f"{it.candidate_composite:.4f} (delta {it.delta:+.4f})" if it.candidate_composite is not None else "n/a"),
        f"- Decision: {it.reason}",
        f"- Changes:\n{it.changes}" if it.changes else "- Changes: none",
    ]
    if analysis:
        lines.append(f"- Failure analysis (LLM): {analysis}")
    if rationale:
        lines.append(f"- Why (LLM): {rationale}")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_improve_loop(cfg: Any, iterations: int | None = None, patience: int | None = None,
                     progress_cb: Callable[[int, int], None] | None = None,
                     store: VersionStore | None = None) -> LoopResult:
    icfg = cfg.improve
    iterations = iterations if iterations is not None else int(icfg.get("iterations", 3))
    patience = patience if patience is not None else int(icfg.get("patience", 2))
    eps, min_delta = float(icfg.get("plateau_eps", 0.005)), float(icfg.get("min_delta", 0.0))
    grid = {"top_k": list(icfg.grid["top_k"]), "temperature": list(icfg.grid["temperature"])}
    db_path = cfg.resolve("paths.db")
    store = store or store_from_cfg(cfg)
    optimizer_llm = get_llm(cfg, "judge")

    current = store.current()
    base_metrics, rows, _ = _baseline(cfg, current, progress_cb)
    start_metrics = dict(base_metrics)
    best = float(base_metrics["composite"])
    n_total = len(load_eval_set(cfg.resolve("paths.eval_set")))
    log.info("baseline %s: composite=%.4f acc=%.0f%% hall=%.0f%% quality=%.0f%%", current.name, best,
             base_metrics["accuracy"] * 100, base_metrics["hallucination_rate"] * 100, base_metrics["quality"] * 100)

    results: list[IterationResult] = []
    stalled, stopped = 0, f"completed {iterations} iteration(s)"
    for i in range(1, iterations + 1):
        failed = [r for r in rows if r.get("failed")]
        if not failed:
            stopped = "no failed cases left"
            break
        log.info("iteration %d/%d: analysing %d failed case(s) of %s", i, iterations, len(failed), current.name)
        try:
            prop: Proposal = propose(optimizer_llm, current, base_metrics, failed, n_total, grid,
                                     int(icfg.get("max_failure_examples", 8)))
        except LLMUnavailableError:
            raise
        except LLMBadOutputError as exc:
            it = IterationResult(i, current.version, None, False, best, None, "no proposal",
                                 f"optimizer produced no valid proposal: {exc}")
            results.append(it)
            _write_changelog(cfg.resolve("paths.changelog"), it, None, "", "")
            db.add_changelog(db_path, {"iteration": i, "base_version": current.version, "candidate_version": 0,
                                       "accepted": False, "base_composite": best, "candidate_composite": best,
                                       "delta": 0.0, "changes": it.changes, "reason": it.reason})
            stalled += 1
            if stalled >= patience:
                stopped = f"no progress for {patience} iteration(s)"
                break
            continue

        cand = AgentConfig(version=store.next_version(), system_prompt=prop.system_prompt, top_k=prop.top_k,
                           temperature=prop.temperature, chunk_size=current.chunk_size, parent=current.version,
                           status="candidate", notes=prop.rationale)
        store.save(cand)
        changes = describe_changes(current, cand)
        summary = run_eval(cfg, cand, progress_cb=progress_cb)
        cand_score = summary.composite

        if not summary.valid:
            accepted, reason = False, "run invalid (too many agent/infrastructure errors); rolled back"
        elif composite_improved(cand_score, best, min_delta):
            accepted, reason = True, f"composite improved by {cand_score - best:+.4f}; accepted"
        else:
            accepted, reason = False, f"composite {cand_score:.4f} did not beat {best:.4f}; rolled back"

        cand.status = "accepted" if accepted else "rejected"
        store.save(cand)
        it = IterationResult(i, current.version, cand.version, accepted, best, cand_score, changes, reason)
        results.append(it)
        _write_changelog(cfg.resolve("paths.changelog"), it, cand, prop.analysis, prop.rationale)
        db.add_changelog(db_path, {"iteration": i, "base_version": current.version,
                                   "candidate_version": cand.version, "accepted": accepted,
                                   "base_composite": best, "candidate_composite": cand_score,
                                   "delta": cand_score - best, "changes": changes,
                                   "reason": f"{reason}. Analysis: {prop.analysis} Why: {prop.rationale}"})
        log.info("iteration %d: %s -> v%d %s (%.4f -> %.4f)", i, current.name, cand.version,
                 "ACCEPTED" if accepted else "rolled back", best, cand_score)

        if accepted:
            gain = cand_score - best
            store.set_current(cand.version)
            current, base_metrics, rows, best = cand, summary.metrics, summary.rows, cand_score
            stalled = 0 if gain >= eps else stalled + 1
        else:
            stalled += 1  # rollback: `current`, `rows`, `best` stay as they were
        if stalled >= patience:
            stopped = f"plateau: no meaningful improvement for {patience} iteration(s)"
            break

    return LoopResult(start_metrics, current.version, base_metrics, results, stopped)


def metrics_line(m: dict[str, Any]) -> str:
    return (f"accuracy {m['accuracy']:.1%} | hallucination {m['hallucination_rate']:.1%} | "
            f"quality {m['quality']:.1%} | composite {m['composite']:.4f}")


__all__ = ["run_improve_loop", "composite_improved", "describe_changes", "composite_score", "metrics_line"]
