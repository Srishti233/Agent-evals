"""Streamlit dashboard: metric cards, trend chart, failed cases, prompt diff, changelog, run-now."""
from __future__ import annotations

import difflib
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from agent.config import store_from_cfg  # noqa: E402
from common.config import load_config  # noqa: E402
from common.llm import LLMError  # noqa: E402
from evals import db  # noqa: E402
from evals.metrics import aggregate  # noqa: E402

st.set_page_config(page_title="Agent Evals", page_icon="📈", layout="wide")

# Streamlit >= 1.50 replaced **STRETCH with width="stretch".
_ST_MINOR = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _ST_MINOR >= (1, 50) else {"use_container_width": True}


@st.cache_resource
def get_cfg():
    return load_config()  # honours $AGENT_EVALS_CONFIG (set by `main.py dashboard`)


cfg = get_cfg()
db_path = cfg.resolve("paths.db")
store = store_from_cfg(cfg)


def pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(str(text).split())) if s]


def diff_html(a: str, b: str, left: str, right: str) -> str:
    """Side-by-side sentence diff as an HTML table; translucent colours work in light and dark themes."""
    la, lb = sentences(a), sentences(b)
    colors = {"delete": "rgba(255,80,80,.25)", "insert": "rgba(60,200,90,.25)", "replace": "rgba(255,200,0,.25)"}
    cell = "padding:6px 10px;vertical-align:top;border:1px solid rgba(128,128,128,.3);width:50%;"
    rows = [f"<tr><th style='{cell}'>{html.escape(left)}</th><th style='{cell}'>{html.escape(right)}</th></tr>"]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb).get_opcodes():
        bg_l = colors["delete" if tag in ("delete", "replace") else "insert"] if tag != "equal" else ""
        bg_r = colors["insert" if tag in ("insert", "replace") else "delete"] if tag != "equal" else ""
        if tag == "replace":
            bg_l, bg_r = colors["replace"], colors["replace"]
        lt = "<br>".join(html.escape(x) for x in la[i1:i2])
        rt = "<br>".join(html.escape(x) for x in lb[j1:j2])
        rows.append(f"<tr><td style='{cell}background:{bg_l if lt else ''}'>{lt}</td>"
                    f"<td style='{cell}background:{bg_r if rt else ''}'>{rt}</td></tr>")
    return "<table style='width:100%;table-layout:fixed;border-collapse:collapse;font-size:14px'>" + "".join(rows) + "</table>"


# ---------------------------------------------------------------------------------- sidebar
st.sidebar.title("Agent Evals")
st.sidebar.caption("Track → Measure → Improve")
current = store.current()
st.sidebar.write(f"**Current agent:** `{current.name}` · top_k={current.top_k} · temp={current.temperature}")
st.sidebar.write(f"**Model:** `{cfg.llm.model}` (judge `{cfg.llm.judge_model}`) · backend `{cfg.llm.backend}`")
if cfg.llm.backend == "mock":
    st.sidebar.warning("Offline mock backend: metrics are not meaningful.")

if st.sidebar.button("▶ Run eval now", type="primary", **STRETCH):
    from evals.runner import run_eval
    bar = st.progress(0.0, text=f"Evaluating {current.name} …")
    try:
        s = run_eval(cfg, current, show_progress=False,
                     progress_cb=lambda done, total: bar.progress(done / total, text=f"{done}/{total} cases"))
        bar.empty()
        st.session_state["flash"] = f"Run #{s.run_id} finished: composite {s.composite:.3f}" + \
            ("" if s.valid else " (WARNING: many cases errored)")
        st.rerun()
    except LLMError as exc:
        bar.empty()
        st.sidebar.error(f"Eval failed: {exc}")
if "flash" in st.session_state:
    st.sidebar.success(st.session_state.pop("flash"))
if st.sidebar.button("↻ Refresh", **STRETCH):
    st.rerun()

# ------------------------------------------------------------------------------------ data
runs = pd.DataFrame(db.list_runs(db_path))
st.title("Agent Evals Dashboard")
if runs.empty:
    st.info("No completed eval runs yet. Click **Run eval now** or run `python main.py eval`.")
    st.stop()

full = runs[~runs["notes"].fillna("").str.contains("partial")]
runs = full if not full.empty else runs
latest = runs.iloc[-1]
prev = None  # compare against the most recent run of a different, non-rejected version
for _, r in runs[runs["agent_version"] != latest["agent_version"]].iloc[::-1].iterrows():
    if store.load(int(r["agent_version"])).status in ("accepted", "baseline"):
        prev = r
        break

st.subheader(f"Latest: agent v{int(latest['agent_version'])} · run #{int(latest['run_id'])} · {latest['finished_at']}")
c1, c2, c3, c4 = st.columns(4)


def delta(col: str, invert: bool = False):
    if prev is None:
        return None
    d = (latest[col] - prev[col]) * 100
    return f"{d:+.1f} pts vs v{int(prev['agent_version'])}"


c1.metric("Accuracy", pct(latest["accuracy"]), delta("accuracy"))
c2.metric("Hallucination", pct(latest["hallucination_rate"]), delta("hallucination_rate"), delta_color="inverse")
c3.metric("Quality", pct(latest["quality"]), delta("quality"))
c4.metric("Composite", f"{latest['composite']:.3f}", None if prev is None else f"{latest['composite'] - prev['composite']:+.3f}")

tab_trend, tab_fail, tab_diff, tab_log, tab_runs = st.tabs(
    ["📈 Metrics by version", "❌ Failed cases", "🔀 Prompt diff", "📝 Changelog", "🗂 All runs"])

# ------------------------------------------------------------------------------------ trend
with tab_trend:
    per_version = runs.groupby("agent_version", as_index=False).last().sort_values("agent_version")
    status = {v: store.load(int(v)).status for v in per_version["agent_version"]}
    xs = [f"v{int(v)}" + ("" if status[v] in ("accepted", "baseline") else f" ({status[v]})")
          for v in per_version["agent_version"]]
    fig = go.Figure()
    for col, name in [("accuracy", "Accuracy"), ("hallucination_rate", "Hallucination"), ("quality", "Quality")]:
        fig.add_trace(go.Scatter(x=xs, y=per_version[col] * 100, mode="lines+markers", name=name))
    fig.add_trace(go.Scatter(x=xs, y=per_version["composite"] * 100, mode="lines+markers", name="Composite ×100",
                             line=dict(dash="dot", color="gray")))
    fig.update_layout(yaxis_title="%", yaxis_range=[0, 100], height=420, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, **STRETCH)
    st.caption("Hallucination: lower is better. Versions marked (rejected) were evaluated but rolled back.")

    results = pd.DataFrame(db.get_results(db_path, int(latest["run_id"])))
    if not results.empty:
        st.markdown("**Breakdown by category (latest run)**")
        cat = results.groupby("category").agg(n=("case_id", "count"), accuracy=("correct", "mean"),
                                              hallucination=("hallucinated", "mean"), quality=("quality", "mean"),
                                              avg_latency_s=("latency_s", "mean")).reset_index()
        for c in ("accuracy", "hallucination", "quality"):
            cat[c] = (cat[c] * 100).round(1)
        cat["avg_latency_s"] = cat["avg_latency_s"].round(2)
        st.dataframe(cat, **STRETCH, hide_index=True)

# ------------------------------------------------------------------------------- failed cases
with tab_fail:
    options = {f"run #{int(r.run_id)} · v{int(r.agent_version)} · {r.finished_at}": int(r.run_id)
               for r in runs.sort_values("run_id", ascending=False).itertuples()}
    chosen = st.selectbox("Run", list(options), index=0)
    res = pd.DataFrame(db.get_results(db_path, options[chosen]))
    failed = res[res["failed"] == 1] if not res.empty else res
    st.write(f"**{len(failed)}** failed of {len(res)} cases")
    if not failed.empty:
        cats = st.multiselect("Category", sorted(failed["category"].unique()), default=sorted(failed["category"].unique()))
        view = failed[failed["category"].isin(cats)][["case_id", "category", "question", "answer", "gold_answer", "failure_reasons"]]
        view = view.rename(columns={"answer": "agent_answer", "failure_reasons": "reason"})
        st.dataframe(view, **STRETCH, hide_index=True)
        pick = st.selectbox("Inspect a case", list(failed["case_id"]))
        row = failed[failed["case_id"] == pick].iloc[0]
        st.markdown(f"**Q:** {row['question']}  \n**Agent:** {row['answer'] or '(none)'}  \n**Gold:** {row['gold_answer']}")
        if row["unsupported"]:
            st.markdown("**Unsupported claims:** " + " | ".join(row["unsupported"]))
        with st.expander("Tool calls & retrieved contexts"):
            st.json({"tool_calls": row["tool_calls"], "contexts": [{k: c[k] for k in ("doc_id", "score")} for c in row["contexts"]]})

# ----------------------------------------------------------------------------------- diff
with tab_diff:
    versions = store.list_versions()
    if len(versions) < 2:
        st.info("Only one agent version exists so far. Run `python main.py improve` to create more.")
    left_col, right_col = st.columns(2)
    a = left_col.selectbox("From", versions, index=max(0, len(versions) - 2), format_func=lambda v: f"v{v}")
    b = right_col.selectbox("To", versions, index=len(versions) - 1, format_func=lambda v: f"v{v}")
    va, vb = store.load(a), store.load(b)
    st.write(f"v{a}: top_k={va.top_k}, temperature={va.temperature}, status={va.status}  →  "
             f"v{b}: top_k={vb.top_k}, temperature={vb.temperature}, status={vb.status}")
    st.markdown(diff_html(va.system_prompt, vb.system_prompt, f"v{a}", f"v{b}"), unsafe_allow_html=True)
    st.caption("Green = added, red = removed, yellow = changed sentence.")

# -------------------------------------------------------------------------------- changelog
with tab_log:
    log_rows = db.list_changelog(db_path)
    if not log_rows:
        st.info("No improvement iterations recorded yet.")
    for r in reversed(log_rows):
        icon = "✅" if r["accepted"] else "↩️"
        title = (f"{icon} Iteration {r['iteration']}: v{r['base_version']} → "
                 f"{'v' + str(r['candidate_version']) if r['candidate_version'] else '—'} · "
                 f"{r['base_composite']:.3f} → {r['candidate_composite']:.3f} ({r['delta']:+.3f})")
        with st.expander(title):
            st.markdown(f"**Decision:** {r['reason']}")
            st.code(r["changes"] or "no change", language="diff")

# ---------------------------------------------------------------------------------- all runs
with tab_runs:
    show = runs[["run_id", "agent_version", "finished_at", "n_cases", "accuracy", "hallucination_rate", "quality",
                 "composite", "n_errors", "model", "judge_model", "notes"]].sort_values("run_id", ascending=False)
    st.dataframe(show, **STRETCH, hide_index=True)
