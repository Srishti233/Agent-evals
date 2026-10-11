# Agent Evals: Track → Measure → Improve

A fully local, free, open-source loop for evaluating and improving a tool-using QA agent.
**No paid APIs, no API keys.** LLM = Ollama (default `llama3.2:3b`), embeddings = `all-MiniLM-L6-v2`,
hallucination check = `cross-encoder/nli-deberta-v3-small`, vector store = FAISS, history = SQLite,
dashboard = Streamlit + Plotly.

> **Verification status (please read).** The build environment had no access to the Ollama registry or
> Hugging Face and no GPU, so the *real* models were **never executed** by me. What *was* run and passed:
> 105 pytest tests, the **real Ollama HTTP client against a fake Ollama server** (down / timeout / 5xx retry /
> 404 / malformed JSON / pull stream), and the complete `setup → eval → improve → dashboard` pipeline using the
> built-in **offline profile** (deterministic stand-ins for the LLM, embeddings and NLI). The two real-model code
> paths that are exercised only on your machine are `SentenceTransformerEmbedder` (`common/embeddings.py`) and
> `TransformersNLI` (`evals/nli.py`). Run `python main.py setup` first; it loads both and fails loudly if anything is off.
> Metrics from the offline profile are **not meaningful**; they only prove the plumbing.

## Architecture

```
                    ┌───────────────────────── data/docs/*.md (18 docs) ──────────────┐
                    ▼                                                                  │
 question ──► AGENT (agent/)                                  ┌───────────────┐        │
              ReAct loop, ≤ 4 steps, JSON actions ───────────►│ search_docs   │◄── FAISS index (MiniLM)
              versioned config: prompt/top_k/temp/chunk       │ calculator    │    (agent/retriever.py)
              (agent/versions/vN.json + CURRENT)              └───────────────┘    safe `ast` math
                    │ answer, contexts, tool_calls, latency
                    ▼
 EVALUATOR (evals/)  runs data/eval_set.json (40 cases) in parallel ─► SQLite (data/runs.db)
   ├─ Accuracy      LLM judge verdict (strict JSON, retries) + fuzzy + embedding sim logged;
   │                unanswerable ⇒ correct only if the agent abstains
   ├─ Hallucination answer → claims → NLI(entailment) vs retrieved chunks (+ calculator results)
   └─ Quality       LLM rubric 1-5: relevance/completeness/clarity → 0-100 %
                    │ composite = 0.4·acc + 0.4·(1−hall) + 0.2·quality
                    ▼
 IMPROVE LOOP (improve/)   failed cases ─► optimizer LLM proposes new prompt (+ top_k/temp from a grid)
   new version vN+1 ─► re-run evals ─► composite better? ── yes ─► ACCEPT (CURRENT → vN+1)
                                                          └─ no ──► ROLL BACK (CURRENT unchanged)
   repeat N iterations or until plateau; every decision is written to improve/CHANGELOG.md + SQLite
                    │
                    ▼
 DASHBOARD (dashboard/app.py)  metric cards · trend by version · failed cases · prompt diff · changelog · Run eval now
```

## Where to run it (not on a laptop)

Any Linux server works; a GPU makes evals much faster. Three options:

### 0. GitHub Codespaces (no Docker, no local install)
Click **Code → Codespaces → Create codespace on main**. The included `.devcontainer/` installs Ollama, CPU PyTorch and the
requirements and starts `ollama serve` (the first build takes several minutes). Then, in the terminal:
```bash
python main.py setup
python main.py eval --limit 8      # quick check first: CPU-only is slow
python main.py eval && python main.py improve --iters 3
python main.py dashboard           # open forwarded port 8501
```
Codespaces storage is wiped when the codespace is deleted: download `data/runs.db` and `agent/versions/` first.
*(The devcontainer config is untested; if the build fails, run the commands from section B manually.)*

### A. Docker Compose on one server
```bash
git clone https://github.com/<your-username>/agent-evals && cd agent-evals
docker compose up -d ollama                       # GPU: uncomment the `deploy` block in docker-compose.yml first
docker compose run --rm app python main.py setup  # pulls llama3.2:3b, downloads MiniLM + NLI, builds the index
docker compose run --rm app python main.py eval
docker compose run --rm app python main.py improve --iters 3
docker compose up -d dashboard                    # bound to 127.0.0.1:8501 → tunnel: ssh -L 8501:localhost:8501 user@server
```
For GPU PyTorch: `docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121`.

### B. Bare Linux VM
```bash
curl -fsSL https://ollama.com/install.sh | sh     # installs + starts the Ollama service
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on GPU boxes; use the default wheel
pip install -r requirements.txt
python main.py setup && python main.py eval && python main.py improve --iters 3
python main.py dashboard                          # then SSH-tunnel port 8501
```
Requires Python 3.10+. `main.py dashboard` binds `0.0.0.0:8501` by default (`dashboard.host` in config.yaml) and
**has no authentication**: use an SSH tunnel or a reverse proxy with auth; do not expose it publicly.

### C. Ollama on a GPU box, app elsewhere
Run Ollama on the GPU machine (`OLLAMA_HOST=0.0.0.0 ollama serve`, firewall it!), then on the app machine:
`export OLLAMA_HOST=http://gpu-box:11434` (or set `llm.host`). `python main.py setup` pulls the models remotely.

**Performance knobs:** `eval.workers` (default 4) should be ≤ `OLLAMA_NUM_PARALLEL` on the Ollama server.
A run is roughly 40 questions × (≤ 4 agent calls + 2 judge calls); I could not time this on real models, so expect
minutes on a GPU and considerably longer on CPU. A bigger judge (e.g. `judge_model: qwen2.5:7b`) usually grades more
reliably than a 3B model grading itself; set it in `config.yaml`.

## CLI

| Command | What it does |
|---|---|
| `python main.py setup` | Checks Ollama, pulls missing models, loads embedding + NLI models, builds the vector index |
| `python main.py eval [--version v2] [--limit N]` | One evaluation (progress bar, results saved to SQLite). `--limit` runs are stored as *partial* and ignored as baselines |
| `python main.py improve --iters 3 [--patience 2]` | Full optimise → re-eval → accept/rollback loop |
| `python main.py ask "question" [--version v1]` | Talk to the agent; prints tool calls and retrieved docs |
| `python main.py dashboard [--host H --port P]` | Streamlit dashboard |

Global flags: `--config path.yaml` (or `$AGENT_EVALS_CONFIG`), `-v` for debug logs. Exit codes: 0 ok, 1 invalid/unreliable
run, 2 Ollama unreachable / model missing.

## How the metrics work

* **Accuracy.** The judge LLM sees question, gold and agent answer and must return `{"verdict": "correct"|"incorrect", "reason": ...}`
  (strict schema; on malformed output the model is told what was wrong and retried, 3×). Fuzzy match (rapidfuzz) and
  embedding cosine are logged next to it. *Unanswerable* cases are decided by rule: correct iff the answer abstains
  (“I don't know”, “cannot find”, …). An abstention on an answerable question is wrong. If the judge keeps failing, the case falls
  back to `fuzzy ≥ 0.8 and cosine ≥ 0.7`; this is recorded (`verdict_source = fallback`) and noted on the run.
* **Hallucination rate.** Answer → sentence claims → NLI against each retrieved chunk *and* all chunks joined. A claim is
  supported if entailment probability ≥ `nli.threshold` (0.5). Calculator results count as evidence, and a numeric claim whose
  numbers all come from the context/calculator is accepted (NLI models can't verify arithmetic). Rate = % of answers with ≥ 1
  unsupported claim. Abstentions contain no claims.
* **Quality.** Judge rubric 1-5 on relevance, completeness, clarity → `(mean−1)/4`.
* **Failed case** = wrong, hallucinated, quality < `eval.quality_fail_threshold` (0.5), agent error, or step budget exhausted.
* **Composite** = `0.4·accuracy + 0.4·(1−hallucination) + 0.2·quality`.

## The improve loop

1. Take the failed cases of the current version's latest full run (reused from SQLite if the eval set is unchanged).
2. The optimizer LLM sees the prompt, metrics, failure patterns and up to 8 sampled failures and returns JSON: analysis, a new
   behavioural system prompt, and optionally `top_k ∈ {2,3,4,5}` / `temperature ∈ {0.0,0.2,0.5}` (snapped to the grid;
   no-op or malformed proposals are retried). The tool/JSON protocol is appended by code, so the optimizer can't break it.
3. A new `agent/versions/vN.json` is written and evaluated. It is **accepted only if composite improves** by more than
   `improve.min_delta`; otherwise **rolled back** (`CURRENT` untouched; the rejected version stays on disk with `status: rejected`).
   A run with > 20 % agent/infra errors is invalid and also rolled back.
4. Stops after N iterations, when no failures remain, or on plateau (`patience` consecutive iterations with no gain ≥ `plateau_eps`).
   Every decision is appended to `improve/CHANGELOG.md` and the `changelog` table.

**Caveats you should know:** with only 40 cases, one question moves accuracy by 2.5 pts, and sampling noise can make a change look
better than it is, so raise `improve.min_delta` (e.g. 0.02) for more conservative accepts. The optimizer sees eval failures, so
repeated loops can overfit the eval set; keep a separate hold-out set before trusting gains. Prompt text is told not to copy specific
questions, but that is a request, not an enforcement.

## Configuration
Everything is in `config.yaml` (models, host, timeouts/retries, embedding/NLI backends and thresholds, paths, workers,
improve grid, dashboard host/port). Relative paths resolve against the config file.

## Tests
```bash
pytest -q          # 105 tests, ~8 s, no Ollama or model downloads needed
```
Covers calculator safety (injection, DoS exponents), claim splitting / abstention / metric math, JSON recovery, judge schema
validation, accept/rollback + plateau logic with scripted scores, version store, the Ollama client against a fake server,
the agent loop (retry, step budget, LLM failure), dataset integrity, and full offline eval runs incl. failure modes.

## Offline smoke profile (no Ollama / no downloads)
`python main.py --config config.offline.yaml setup|eval|improve|dashboard` uses a mock LLM, hashed bag-of-words embeddings and a
lexical NLI stand-in, with state in `.offline/` (delete to reset). Use it for CI or to check wiring. **Metrics are meaningless.**

### Sample output (offline profile, mock backends, so the numbers are plumbing-only)
```
$ python main.py --config config.offline.yaml ask "How many weeks of paid parental leave do primary caregivers receive?"
Answer (v1): # Parental Leave Primary caregivers receive 26 weeks of paid parental leave.
Latency: 0.002s over 2 step(s)
  [step 1] search_docs('How many weeks of paid parental leave do primary caregivers receive?') -> [06_parental_leave] # Parental Leave Primary caregivers receive 26 weeks of paid parental leave. Secondary caregivers receive 8 weeks of paid parental leave. Th
  Retrieved: 06_parental_leave (0.73), 05_leave_policy (0.24), 11_engineering_practices (0.11)

$ python main.py --config config.offline.yaml eval
Evaluation of v1 (run #1, 40 cases, 0.2s)
  Accuracy        35.0%
  Hallucination    0.0%   (lower is better)
  Quality         73.5%
  Composite      0.6871   (0.4*acc + 0.4*(1-hall) + 0.2*quality)

  By category:
    factual       n=16  acc= 75.0%  hall=  0.0%  quality= 84.9%
    math          n=8   acc=  0.0%  hall=  0.0%  quality= 63.5%
    multi-hop     n=8   acc= 12.5%  hall=  0.0%  quality= 69.8%
    unanswerable  n=8   acc= 12.5%  hall=  0.0%  quality= 64.6%

  Failed cases: 26   (results saved to data/runs.db)

$ python main.py --config config.offline.yaml improve --iters 3
================ IMPROVE LOOP SUMMARY ================
 iter 1: v1 -> v2   rolled back  composite 0.6871 -> 0.6863
 iter 2: v1 -> v3   rolled back  composite 0.6871 -> 0.6863

Stopped: plateau: no meaningful improvement for 2 iteration(s)

BEFORE (baseline):
  Accuracy        35.0%
  Hallucination    0.0%   (lower is better)
  Quality         73.5%
  Composite      0.6871   (0.4*acc + 0.4*(1-hall) + 0.2*quality)

AFTER  (current = v1):
  Accuracy        35.0%
  Hallucination    0.0%   (lower is better)
  Quality         73.5%
  Composite      0.6871   (0.4*acc + 0.4*(1-hall) + 0.2*quality)

Composite change: +0.0000
Changelog: improve/CHANGELOG.md
```
The sample shows two candidates rolled back because they did not beat the baseline composite. The *accept* path (and branching from the last accepted version after a rollback) is covered by `tests/test_improve_logic.py` with scripted scores.

## Project layout
```
main.py              CLI                         config.yaml / config.offline.yaml
agent/               agent.py (ReAct) tools.py (safe calc, search) retriever.py (chunk+FAISS) config.py (versions) versions/
common/              llm.py (Ollama client, chat_json) mock_llm.py embeddings.py jsonparse.py config.py logging_setup.py
evals/               runner.py metrics.py judge.py nli.py hallucination.py db.py
improve/             optimizer.py loop.py CHANGELOG.md (generated)
dashboard/app.py     Streamlit UI
data/                docs/ (18 fictional "Lumora Labs" docs), eval_set.json (40 cases; regenerate with scripts/make_eval_set.py)
tests/               pytest suite
Dockerfile docker-compose.yml requirements.txt
```

## Troubleshooting
* `Cannot reach Ollama` → `ollama serve`, or fix `OLLAMA_HOST` / `llm.host`. `Model ... not found` → `python main.py setup`.
* Many `agent error` / run marked invalid → Ollama overloaded or timing out; lower `eval.workers`, raise `llm.timeout_s`.
* First `setup` is slow: it downloads the LLM (~2 GB), MiniLM and the NLI model once (cached in `~/.cache/huggingface`).
* Judge failures are listed in the run notes; consider a larger `judge_model`.
