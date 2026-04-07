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
- `set_node_params` (requires `node`, optionally `batch_size`, `max_concurrency`, `precision`)
- `drain_node` (requires `node`)
- `resume_node` (requires `node`)
- `restart_node` (requires `node`)
- `rebalance` (uses `strategy`: `even | least_rtt | least_vram | min_oom`)

### What each action does (beginner-friendly)

- `noop`
	- What it does: take no action this step.
	- When to use: everything looks stable and within SLO.

- `set_node_params`
	- What it does: changes capacity/latency/VRAM tradeoffs for a specific node.
	- Fields:
		- `node`: which node (0/1/2)
		- `batch_size`: larger can increase throughput but can worsen tail latency and VRAM pressure
		- `max_concurrency`: larger can serve more requests but can increase queueing and VRAM pressure
		- `precision`: `fp16|bf16|int8|int4` — lower precision reduces *model weights* VRAM cost in the simulator
	- When to use:
		- Tail latency (`p95_ms`) is rising and/or `queue_depth` is high → reduce `batch_size` and/or `max_concurrency`
		- Throughput (`tps`) is low vs `incoming_rps` and nodes are healthy → increase `batch_size`/`max_concurrency`
		- VRAM pressure is rising (`vram_velocity > 0`) → try `precision:"int4"` or smaller batch

- `drain_node`
	- What it does: gradually shifts traffic off a node over ~2 steps (not instant).
	- When to use: a node has an RTT spike or instability and you want to stop routing traffic to it.

- `resume_node`
	- What it does: gradually reintroduces traffic to a drained node over ~2 steps.
	- When to use: the incident is over and the node metrics are back to normal.

- `restart_node`
	- What it does: simulates a reboot (brief cooldown) and clears VRAM leak state on that node.
	- When to use: the node is already OOMing / failing badly. This is reactive, so it’s usually worse than proactive tuning.

- `rebalance`
	- What it does: redistributes traffic shares across serving nodes.
	- Strategies:
		- `even`: equal traffic across all serving nodes
		- `least_rtt`: prefer nodes with lower RTT
		- `least_vram`: prefer nodes with lower VRAM usage
		- `min_oom`: prefer nodes with lower OOM rate
	- When to use: SLA is failing and you need to quickly route away from a “bad” node.

## Observations (what the agent sees)

Each step returns cluster + node metrics, including:

- Cluster: `tps`, `p95_ms`, `p95_trend`, `error_rate`, `sla_pass_step`
- Node: `traffic_share`, `precision`, `vram_used_pct`, `vram_velocity`, `rtt_ms`, `oom_rate`, `queue_depth`, `draining`, `is_healthy`

At episode end (`done=True`), the observation includes episode summaries:

- `final_score`, `uptime`, `avg_p95_ms`, `avg_error_rate`, `restart_count`
- `score_breakdown` (V2 weighted components)

## Rewards (what the `reward` means)

This environment uses a simple reward convention:

- **Non-terminal steps:** `reward=0.0`
- **Terminal step (`done=True` at step 60):** `reward` is set to the deterministic final score in `[0,1]`
	- This is exactly `score_breakdown.weighted_total`.

Rule of thumb: evaluate runs using the final `reward` and `score_breakdown` at `done=True`.

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

Important: if you want the web UI, set `ENABLE_WEB_INTERFACE=true` **before** starting the server.

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

### Why you must click Reset first

The web UI lets you send actions, but an episode must be initialized first.

- **Reset** selects the `task_id`, creates the 3 nodes, and starts a fresh 60-step episode.
- If you press **Step** before **Reset**, the environment has no active episode yet, so node-scoped actions (like `set_node_params` with `node=0`) historically could cause "index out of range" style errors.

Rule of thumb: always **Reset → Step → Step → …**.

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
- `precision` (set_node_params only): `fp16 | bf16 | int8 | int4`
- `strategy` (rebalance only): `even | least_rtt | least_vram | min_oom`

### Quick "test values" (what to try and why)

These are small, deterministic probes to confirm the environment features are behaving as intended.

**A) Quantization reduces VRAM pressure**

Goal: see `nodes[i].vram_used_pct` drop (and `nodes[i].vram_velocity` go negative) after switching precision.

1) Reset to hard: `{"task_id":"mixed_incidents_hard"}`
2) Apply int4 on node 1: `{"kind":"set_node_params","node":1,"precision":"int4"}`
3) Watch observation fields:
	- `nodes[1].precision` becomes `int4`
	- `nodes[1].vram_velocity` becomes negative on the next step if VRAM usage drops

**B) Drain + rebalance responds to RTT spikes**

Goal: reduce tail latency impact from the RTT-spiking node.

1) Reset to medium: `{"task_id":"network_spike_medium"}`
2) When node 2 RTT spikes:
	- `{"kind":"drain_node","node":2}`
	- `{"kind":"rebalance","strategy":"least_rtt"}`
3) Watch:
	- `cluster.p95_trend` should move toward 0 or negative after the change

**C) Capacity boost (throughput) without restart spam**

Goal: increase served throughput while keeping error rate low.

- `{"kind":"set_node_params","node":0,"batch_size":16,"max_concurrency":32}`
- `{"kind":"set_node_params","node":1,"batch_size":16,"max_concurrency":32}`

Watch:
- `cluster.tps` rises
- `score_breakdown.efficiency` improves at `done=True`

**D) Queue pressure test (proactive throttling trigger)**

Goal: understand why we sometimes lower `batch_size` before OOM.

- If a node becomes extremely backed up (queue depth far exceeding `max_concurrency`), lowering `batch_size` can reduce queueing/tail latency.
- Observe `nodes[i].queue_depth`, `cluster.p95_ms`, and `cluster.p95_trend` while experimenting with:
  - `{"kind":"set_node_params","node":0,"batch_size":2,"max_concurrency":32}` (lower compute batch; may reduce queue latency)

Tip: the deterministic signals to watch while testing are:
- `cluster.p95_trend` (is p95 getting better or worse step-to-step?)
- `nodes[i].vram_velocity` (is VRAM pressure rising or falling?)

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
