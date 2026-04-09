---
title: Autonomous LLM Cluster Manager
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

## Phase-2 Validator Compatibility Checklist

To match organizer guidance and common Phase-2 deep validation expectations, this repo includes:

- **openenv.yaml task declarations** (root + `llama_sre_orchestrator/openenv.yaml`)
  - `port: 7860`
  - `tasks:` contains:
    - `vram_recovery_easy` (max_steps: 60)
    - `network_spike_medium` (max_steps: 60)
    - `mixed_incidents_hard` (max_steps: 60)
  - Each task includes an explicit YAML grader:
    - `grader.type: llm`
    - `grader.prompt_template: ...` returning JSON `{"score": <float>}`

- **Runtime graders (OpenEnv Rubric)** implemented in code and invoked on each step.
- **Strict score bounds**: all emitted task scores are squeezed to the interior range `0.01..0.99` (never `0.0` or `1.0`).
- **Baseline runner output** (`inference.py`) prints structured `[START]`, `[STEP]`, `[END]` JSON and includes redundant task/grader schema variants in `[END]` for parser compatibility.

Despite these changes, the Phase-2 validator still reports: “Not enough tasks with graders” and “One or more task scores are out of range”.

Reference configs: `openenv.yaml` and `llama_sre_orchestrator/openenv.yaml`.
# Llama SRE Orchestrator (OpenEnv)

# OpenEnv Autonomous SRE Environment

An OpenEnv environment where an agent acts as an **autonomous Site Reliability Engineer (SRE)** for a simulated **3-node GPU inference cluster**.  

The agent must maintain:
- **p95 latency** under task-specific SLOs  
- **Error rate** within acceptable thresholds  

while deterministic incidents unfold:
- VRAM leak  
- RTT spikes  
- Throughput throttling  

---

## 🔗 Live Environment
- **Web UI:** [aryannovice-autonomous-llm-cluster-manager.hf.space/web](https://aryannovice-autonomous-llm-cluster-manager.hf.space/web)  
- **Hugging Face Spaces:** [aryannovice-autonomous-llm-cluster-manager.hf.space](https://aryannovice-autonomous-llm-cluster-manager.hf.space/)
- **Just the space** [https://huggingface.co/spaces/Aryannovice/autonomous-llm-cluster-manager](https://huggingface.co/spaces/Aryannovice/autonomous-llm-cluster-manager)
- Core env package: `llama_sre_orchestrator/`
- Repo-root validator wrapper: `server/` + `openenv.yaml`
- Baseline runner (submission requirement): `inference.py`

## Quickstart (beginner-friendly)

You can interact with this environment in two ways:

- **Web UI**: click buttons, paste JSON actions, and watch metrics.
- **Baseline script**: run `inference.py` to play all tasks end-to-end and print final scores.

### A) Install (once)

From the repo root:

```bash
python -m venv .venv
```

- Windows PowerShell:

```powershell
& .\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### B) Run the server

If you want the web UI, set `ENABLE_WEB_INTERFACE=true` **before** starting the server.

- PowerShell:

```powershell
$env:ENABLE_WEB_INTERFACE="true"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/web`

### C) Run the baseline (optional)

In a second terminal (same venv):

```bash
python inference.py --base-url http://127.0.0.1:8000
```

### D) Validate like the submission script

```bash
.venv/Scripts/openenv.exe validate
```

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

### What the “trend/velocity” fields are for

- `cluster.p95_trend` is `p95_ms[t] - p95_ms[t-1]`.
	- Negative means tail latency is improving.
	- Positive means tail latency is getting worse.
- `nodes[i].vram_velocity` is `vram_used_pct[t] - vram_used_pct[t-1]`.
	- Positive means VRAM pressure is rising (leaks/unsafe settings).
	- Negative means VRAM pressure is falling (recovery actions worked).

## Rewards (what the `reward` means)

This environment uses a simple reward convention:

- **Non-terminal steps:** `reward=0.0`
- **Terminal step (`done=True` at step 60):** `reward` is set to the deterministic final score in `[0,1]`
	- This is exactly `score_breakdown.weighted_total`.

Rule of thumb: evaluate runs using the final `reward` and `score_breakdown` at `done=True`.

In the web UI this often looks like:

- While stepping: `Reward: 0.0` and `Done: False`
- At the end: `Done: True` and `Reward: <final_score>`

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

(See **Quickstart** above for the simplest run commands.)

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

### Web UI navigation (what to click)

Most OpenEnv UIs expose three main operations:

- **Reset**: starts a new episode for a task (you choose `task_id`).
- **Step**: applies one action and advances time by one step.
- **Get state**: shows the current observation without taking an action.

Typical loop:

1) Click **Reset** with a payload (example below).
2) Click **Step** repeatedly (or Step → Get state → Step) to watch the metrics.
3) Stop when `done=True` (the final score + `score_breakdown` will appear).

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

## Understanding baseline output (`inference.py`)

When you run `python inference.py --base-url http://127.0.0.1:8000`, it prints a JSON summary.

Example (your numbers will vary slightly by policy):

```json
{
	"scores": {
		"vram_recovery_easy": 1.0,
		"network_spike_medium": 0.64,
		"mixed_incidents_hard": 0.56
	},
	"mean": 0.73,
	"details": {
		"vram_recovery_easy": {
			"score": 1.0,
			"score_breakdown": {
				"availability": 1.0,
				"latency": 1.0,
				"efficiency": 1.0,
				"weighted_total": 1.0,
				"avg_p95_ms": 140.3,
				"avg_error_rate": 0.0,
				"avg_tps": 900.0,
				"restart_count": 0.0,
				"p95_threshold_ms": 350.0
			}
		}
	}
}
```

How to read it:

- `scores[task_id]` is the **final episode score** for that task.
- `mean` is the simple average of task scores.
- `details[task_id].score_breakdown` provides “partial credit” components:
	- `availability`: mostly controlled by error rate
	- `latency`: how far p95 is from the task’s threshold
	- `efficiency`: served throughput + avoiding restart spam
	- `weighted_total`: the final deterministic score (same as terminal `reward`)

## Example observation (single step)

When you click **Get state** (or after each **Step**) in `/web`, you’ll see an observation JSON. Below is an **illustrative** example showing the most important fields.

```json
{
	"task_id": "mixed_incidents_hard",
	"step": 12,
	"incoming_rps": 900.0,
	"incident": "RTT spike on node 2.",
	"cluster": {
		"tps": 860.2,
		"p95_ms": 410.5,
		"p95_trend": 35.1,
		"error_rate": 0.012,
		"sla_pass_step": false
	},
	"nodes": [
		{
			"id": 0,
			"traffic_share": 0.33,
			"batch_size": 16,
			"max_concurrency": 32,
			"precision": "fp16",
			"vram_used_pct": 0.78,
			"vram_velocity": 0.01,
			"rtt_ms": 12.0,
			"oom_rate": 0.0,
			"queue_depth": 18.4,
			"is_healthy": true,
			"draining": false
		}
	]
}
```

What you’re trying to do while stepping:

- Keep `cluster.p95_ms` under the task SLO, and watch `cluster.p95_trend`:
	- Positive trend = getting worse → intervene (drain/rebalance/tune params)
	- Negative trend = getting better → your last actions helped
- Keep `error_rate` low (driving `availability` in `score_breakdown`).
- Watch `nodes[i].queue_depth` for early warning of latency blow-ups.
- Watch `nodes[i].vram_used_pct` and `nodes[i].vram_velocity` for VRAM risk and leaks.

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

## Deploy to Hugging Face Spaces (Docker)

You can deploy this repo as a **Docker Space** and submit the Space URL as your demo.

1) Create the Space
- Hugging Face → **New Space**
- **SDK**: Docker
- Pick a name (this becomes your demo URL)

2) Connect your GitHub repo
- In the Space settings, connect to your GitHub repository (or push the repo to HF directly).
- Spaces will build using the root `Dockerfile`.

3) Add (optional) secrets / variables

If you want LLM-assisted actions from `inference.py` (optional), add these in Space settings:
- **Secret**: `HUGGING_FACE_HUB_TOKEN` (recommended)
- **Variable**: `MODEL_NAME` (example: `meta-llama/Meta-Llama-3.1-8B-Instruct`)

The environment itself does not require secrets to run.

4) Confirm the demo URL
- The Space root URL should open the web UI (redirects to `/web` when enabled).
- Useful endpoints:
	- `/web` (interactive UI)
	- `/health` (health check)

Note: Spaces commonly sets `PORT=7860`. The Docker image honors `PORT` automatically.

## Troubleshooting (common beginner gotchas)

- "port already in use" on 8000
	- Something else is running on port 8000. Stop that process or use another port:
		- `python -m uvicorn server.app:app --host 127.0.0.1 --port 8001`

- Web UI not showing up
	- Make sure `ENABLE_WEB_INTERFACE=true` is set **before** you start uvicorn.
	- Then open `/web`.

- Root URL shows JSON or redirects
	- That’s expected. The useful endpoints are `/health` and `/web`.

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
