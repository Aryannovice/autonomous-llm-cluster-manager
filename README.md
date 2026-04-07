# Llama SRE Orchestrator (OpenEnv)

An OpenEnv environment where an agent acts as an autonomous SRE for a simulated **3-node GPU inference cluster**. The agent must keep p95 latency and error rate under task-specific SLOs while deterministic incidents unfold (VRAM leak, RTT spikes, throughput throttling).

- Core env package: `llama_sre_orchestrator/`
- Repo-root validator wrapper: `server/` + `openenv.yaml`
- Baseline runner (submission requirement): `inference.py`

## What this satisfies (submission constraints)

This repo is structured specifically to pass common OpenEnv hackathon validators:

- Repo-root `inference.py` exists and runs all tasks end-to-end.
- **3 tasks** selectable via `reset(task_id=...)`.
- **Fixed 60-step episodes** (deterministic discrete-time simulation).
- **Deterministic grading**: terminal episode score returned in `[0,1]` as `reward` when `done=True`.
- `openenv validate` works from **repo root**.
- Root `Dockerfile` builds/runs from repo root.
- Web UI can be enabled at `/web` (OpenEnv Gradio UI) via a supported flag.

## Tasks (difficulty progression)

Select a task per episode via `reset(task_id=...)`:

- `vram_recovery_easy`: single-fault VRAM leak (node 1)
- `network_spike_medium`: RTT spike (node 2) + throughput throttle (node 0)
- `mixed_incidents_hard`: **dual-fault** + stricter SLO (p95 250ms, error 0.5%)

Trajectory rationale:

- **Easy** is local and recoverable with a single intervention.
- **Medium** requires coordinating routing decisions (drain/rebalance) with capacity tuning.
- **Hard** is a multi-incident scenario where the agent must prioritize which failure mode to address first under a tighter SLO.

V2 realism: drain/resume is not instantaneous (traffic moves over ~2 steps), so actions have delayed effects.

## Actions (structured JSON)

Action kinds:

- `noop`
- `set_node_params` (requires `node`, optionally `batch_size`, `max_concurrency`)
- `drain_node` (requires `node`)
- `resume_node` (requires `node`)
- `restart_node` (requires `node`)
- `rebalance` (uses `strategy`: `even | least_rtt | least_vram | min_oom`)

## Observations (what the agent sees)

Each step returns cluster + node metrics, including:

- Cluster: `tps`, `p95_ms`, `error_rate`, `sla_pass_step`
- Node: `traffic_share`, `vram_used_pct`, `rtt_ms`, `oom_rate`, `queue_depth`, `draining`, `is_healthy`

At episode end (`done=True`), the observation includes episode summaries:

- `final_score`, `uptime`, `avg_p95_ms`, `avg_error_rate`, `restart_count`
- `score_breakdown` (V2 weighted components)

## V2 grading (weighted score breakdown)

At `done=True`, the environment emits a deterministic `score_breakdown` and sets the final `reward` to `score_breakdown.weighted_total`.

Components (all mapped into `[0,1]`):

- **Availability (40%)**: mean availability proxy `1 - avg_error_rate`
- **Latency (30%)**: linear scaling from 1.0 at/below threshold to 0.0 at/above `2x` threshold
- **Efficiency (30%)**: served fraction + restart minimization (penalizes “restart spam”)

This provides partial credit signals while still producing a single deterministic episode score.

## Determinism model

- The **environment** and **grader** are deterministic.
- The baseline agent in `inference.py` can optionally use an LLM. If LLM output is unavailable/invalid, it falls back to a deterministic heuristic policy.

## Run locally

### 1) Start the server

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 2) Run the baseline

```bash
python inference.py --base-url http://127.0.0.1:8000
```

### 3) Validate like the submission script

On Windows venvs, this is typically:

```bash
.venv/Scripts/openenv.exe validate
```

## Web interface (/web)

OpenEnv’s built-in Gradio UI can be mounted at `/web` by setting:

- PowerShell:

```powershell
$env:ENABLE_WEB_INTERFACE="true"
```

- bash/zsh:

```bash
export ENABLE_WEB_INTERFACE=true
```

Then open:

```text
http://127.0.0.1:8000/web
```

### Demo inputs for the web UI

Reset payloads (use the Reset tab/payload box if present):

- Easy: `{"task_id":"vram_recovery_easy"}`
- Medium: `{"task_id":"network_spike_medium"}`
- Hard: `{"task_id":"mixed_incidents_hard"}`

Valid action inputs:

- `kind`: `noop | set_node_params | drain_node | resume_node | restart_node | rebalance`
- `node`: `0 | 1 | 2`
- `batch_size`: `1 | 2 | 4 | 8 | 16`
- `max_concurrency`: `1 | 2 | 4 | 8 | 16 | 32`
- `strategy` (rebalance only): `even | least_rtt | least_vram | min_oom`

Copy/paste sequences:

**Medium task quick demo (drain + rebalance)**

1) Reset to medium.
2) Pre-boost healthy nodes:
	- `{"kind":"set_node_params","node":0,"batch_size":16,"max_concurrency":32}`
	- `{"kind":"set_node_params","node":1,"batch_size":16,"max_concurrency":32}`
3) When RTT spike hits node 2:
	- `{"kind":"drain_node","node":2}`
	- `{"kind":"rebalance","strategy":"least_rtt"}`
4) When spike ends:
	- `{"kind":"resume_node","node":2}`
	- `{"kind":"rebalance","strategy":"even"}`

**Hard task quick demo (dual-fault stabilization)**

1) Reset to hard.
2) Boost capacity on nodes 0/1:
	- `{"kind":"set_node_params","node":0,"batch_size":16,"max_concurrency":32}`
	- `{"kind":"set_node_params","node":1,"batch_size":16,"max_concurrency":32}`
3) Drain RTT-spiking node 2 when needed:
	- `{"kind":"drain_node","node":2}`
4) If node 1 VRAM/oom becomes risky:
	- `{"kind":"set_node_params","node":1,"batch_size":4,"max_concurrency":8}`
	- `{"kind":"restart_node","node":1}` (only if OOMing badly)
5) Endgame:
	- `{"kind":"resume_node","node":2}`
	- `{"kind":"rebalance","strategy":"even"}`

Tip: keep stepping until `done=True` and inspect `score_breakdown` in the raw JSON response.

## Docker (lean build)

The root Dockerfile is intended to be Space-friendly (8GB RAM environments) by keeping dependencies minimal.

```bash
docker build -t metaxhf:lean .
docker run --rm -p 8000:8000 metaxhf:lean
```

Then check:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/web`

Build context is kept small and secrets are protected via `.dockerignore`.

## Secrets (HF token safety)

- Do not hardcode tokens in code.
- For local dev, you may use a `.env` file (ignored by git) and `python-dotenv` will load it.
- For Hugging Face Spaces, add the token as a **Secret** (recommended name: `HUGGING_FACE_HUB_TOKEN`).

Example `.env` (local only):

```env
HUGGING_FACE_HUB_TOKEN=hf_...
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
```
