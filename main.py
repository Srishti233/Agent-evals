#!/usr/bin/env python3
"""Agent Evals CLI:  setup | eval | improve | ask | dashboard"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.config import store_from_cfg  # noqa: E402
from common.config import Cfg, load_config  # noqa: E402
from common.llm import LLMError, LLMUnavailableError, OllamaLLM, get_llm  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("main")


def _fmt_metrics(m: dict) -> str:
    return (f"  Accuracy       {m['accuracy']:6.1%}\n  Hallucination  {m['hallucination_rate']:6.1%}   (lower is better)\n"
            f"  Quality        {m['quality']:6.1%}\n  Composite      {m['composite']:6.4f}   "
            f"(0.4*acc + 0.4*(1-hall) + 0.2*quality)")


# ------------------------------------------------------------------------------------ setup
def cmd_setup(cfg: Cfg, args: argparse.Namespace) -> int:
    from agent.retriever import Retriever, load_docs
    from common.embeddings import get_embedder
    from evals.nli import get_nli
    from evals.runner import load_eval_set

    print("== 1/4 LLM backend ==")
    if cfg.llm.backend == "mock":
        print("  backend=mock (offline test double): nothing to check or pull")
    else:
        probe = get_llm(cfg, "agent")
        assert isinstance(probe, OllamaLLM)
        try:
            models = probe.list_models()
        except LLMUnavailableError as exc:
            print(f"  ERROR: {exc}\n  Install Ollama (https://ollama.com/download), run `ollama serve`, "
                  f"or point llm.host / $OLLAMA_HOST at the machine that runs it.")
            return 2
        print(f"  Ollama reachable at {probe.host} ({len(models)} model(s) installed)")
        for name in dict.fromkeys([cfg.llm.model, cfg.llm.judge_model]):
            if OllamaLLM._has(models, name):
                print(f"  model {name}: already present")
                continue
            if args.skip_pull:
                print(f"  model {name}: MISSING (--skip-pull given)")
                return 2
            print(f"  pulling {name} ...")
            last = {"s": ""}

            def cb(status: str, done: int | None, total: int | None) -> None:
                label = f"{status} {done * 100 // total}%" if done and total else status
                if label != last["s"] and (not done or (done * 100 // max(total or 1, 1)) % 10 == 0):
                    print(f"    {label}")
                    last["s"] = label
            probe.pull(name, cb)
            print(f"  model {name}: pulled")

    print("== 2/4 Embedding model ==")
    emb = get_embedder(cfg)
    print(f"  embedder ready: {emb.name} (dim={emb.dim})")

    print("== 3/4 NLI model ==")
    get_nli(cfg)
    print(f"  NLI backend ready: {cfg.nli.backend} ({cfg.nli.get('model', '-') if cfg.nli.backend != 'lexical' else 'lexical stand-in'})")

    print("== 4/4 Vector index + data ==")
    store = store_from_cfg(cfg)
    current = store.current()
    n_docs = len(load_docs(cfg.resolve("paths.docs_dir")))
    n_chunks = Retriever(cfg, current.chunk_size, emb).build(force=True)
    cases = load_eval_set(cfg.resolve("paths.eval_set"))
    print(f"  indexed {n_docs} docs -> {n_chunks} chunks (chunk_size={current.chunk_size})")
    print(f"  eval set: {len(cases)} cases; current agent version: {current.name}")
    print("Setup complete. Next: python main.py eval")
    return 0


# -------------------------------------------------------------------------------------- eval
def cmd_eval(cfg: Cfg, args: argparse.Namespace) -> int:
    from evals.runner import run_eval
    store = store_from_cfg(cfg)
    agent_cfg = store.load(args.version) if args.version else store.current()
    summary = run_eval(cfg, agent_cfg, limit=args.limit, show_progress=not args.no_progress)
    m = summary.metrics
    print(f"\nEvaluation of {agent_cfg.name} (run #{summary.run_id}, {m['n']} cases, {summary.duration_s}s)")
    print(_fmt_metrics(m))
    print("\n  By category:")
    for cat, c in m["by_category"].items():
        print(f"    {cat:<13} n={c['n']:<3} acc={c['accuracy']:6.1%}  hall={c['hallucination_rate']:6.1%}  quality={c['quality']:6.1%}")
    print(f"\n  Failed cases: {summary.n_failed}   (results saved to {cfg.resolve('paths.db')})")
    if not summary.valid:
        print(f"  WARNING: {m['n_errors']} case(s) errored - metrics may be unreliable. See logs.")
        return 1
    return 0


# ---------------------------------------------------------------------------------- improve
def cmd_improve(cfg: Cfg, args: argparse.Namespace) -> int:
    from improve.loop import metrics_line, run_improve_loop
    res = run_improve_loop(cfg, iterations=args.iters, patience=args.patience)
    print("\n================ IMPROVE LOOP SUMMARY ================")
    for it in res.iterations:
        tag = "ACCEPTED   " if it.accepted else "rolled back"
        cand = f"v{it.candidate_version}" if it.candidate_version else "--"
        score = f"{it.candidate_composite:.4f}" if it.candidate_composite is not None else "  n/a "
        print(f" iter {it.iteration}: v{it.base_version} -> {cand:<4} {tag}  composite {it.base_composite:.4f} -> {score}")
    print(f"\nStopped: {res.stopped_because}")
    print("\nBEFORE (baseline):\n" + _fmt_metrics(res.baseline))
    print(f"\nAFTER  (current = v{res.final_version}):\n" + _fmt_metrics(res.final_metrics))
    d = res.final_metrics["composite"] - res.baseline["composite"]
    print(f"\nComposite change: {d:+.4f}")
    print(f"Changelog: {cfg.resolve('paths.changelog')}")
    return 0


# ------------------------------------------------------------------------------------- ask
def cmd_ask(cfg: Cfg, args: argparse.Namespace) -> int:
    from agent.agent import Agent
    store = store_from_cfg(cfg)
    agent_cfg = store.load(args.version) if args.version else store.current()
    res = Agent(cfg, agent_cfg).run(args.question)
    print(f"\nAnswer ({agent_cfg.name}): {res.answer or '(no answer)'}")
    if res.error:
        print(f"Error: {res.error}")
    print(f"Latency: {res.latency_s}s over {res.steps} step(s)")
    for tc in res.tool_calls:
        print(f"  [step {tc['step']}] {tc['tool']}({tc['input']!r}) -> {tc['output'][:160].replace(chr(10), ' ')}")
    if res.contexts:
        print("  Retrieved: " + ", ".join(f"{c['doc_id']} ({c['score']:.2f})" for c in res.contexts))
    return 1 if res.error else 0


# -------------------------------------------------------------------------------- dashboard
def cmd_dashboard(cfg: Cfg, args: argparse.Namespace) -> int:
    host = args.host or cfg.dashboard.host
    port = args.port or int(cfg.dashboard.port)
    env = dict(os.environ, AGENT_EVALS_CONFIG=cfg["_path"])
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
           "--server.address", host, "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    print(f"Dashboard -> http://{shown}:{port}   (no auth: on a remote box use an SSH tunnel: "
          f"ssh -L {port}:localhost:{port} user@server)")
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Agent Evals: Track -> Measure -> Improve")
    p.add_argument("--config", help="path to config YAML (default: config.yaml or $AGENT_EVALS_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="check Ollama, pull models, load NLI/embeddings, build the vector index")
    s.add_argument("--skip-pull", action="store_true", help="fail instead of pulling missing models")
    s.set_defaults(fn=cmd_setup)

    e = sub.add_parser("eval", help="run one evaluation of the current (or given) version")
    e.add_argument("--version", help="agent version to evaluate, e.g. v2 (default: current)")
    e.add_argument("--limit", type=int, help="only run the first N cases (smoke test; stored as partial)")
    e.add_argument("--no-progress", action="store_true")
    e.set_defaults(fn=cmd_eval)

    i = sub.add_parser("improve", help="run the optimise/re-evaluate/accept-or-rollback loop")
    i.add_argument("--iters", type=int, help="max iterations (default from config: 3)")
    i.add_argument("--patience", type=int, help="stop after N non-improving iterations")
    i.set_defaults(fn=cmd_improve)

    a = sub.add_parser("ask", help="ask the agent a question")
    a.add_argument("question")
    a.add_argument("--version")
    a.set_defaults(fn=cmd_ask)

    d = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    d.add_argument("--host")
    d.add_argument("--port", type=int)
    d.set_defaults(fn=cmd_dashboard)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging("DEBUG" if args.verbose else cfg.logging.get("level", "INFO"), cfg.resolve("paths.log_dir"))
    try:
        return int(args.fn(cfg, args))
    except LLMUnavailableError as exc:
        print(f"\nERROR: {exc}\nStart Ollama (`ollama serve`) or set OLLAMA_HOST, then retry.", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
