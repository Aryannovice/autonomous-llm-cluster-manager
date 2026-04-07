# Llama SRE Orchestrator (OpenEnv)

An OpenEnv environment where an agent acts as an autonomous SRE for a simulated **3-node GPU inference cluster**, keeping latency/error SLOs while handling deterministic incidents.

- Environment package: `llama_sre_orchestrator/`
- Hugging Face Space (target): `Aryannovice/llama-sre-orchestrator`

## Submission/validator expectations (what this repo is built for)

- Repo-root `inference.py` that can run episodes for all tasks.
- 3+ tasks selectable via `reset(task_id=...)`.
- Deterministic scoring with final episode score returned in `[0, 1]` (the environment returns it as `reward` when `done=True`).
- `openenv validate` runnable from **repo root**.
- Docker build/run support from **repo root** (a root `Dockerfile` is present).

## Tasks

Select a task per episode via `reset(task_id=...)`:
- `vram_recovery_easy`: VRAM leak on node 1
- `network_spike_medium`: RTT spike on node 2 + throughput throttle on node 0
- `mixed_incidents_hard`: combined incidents with stricter SLO (p95 250ms, error 0.5%)

## Run locally

### 1) Start the environment server

From repo root:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 2) Run the baseline agent

```bash
python inference.py --base-url http://127.0.0.1:8000
```

It prints a JSON with per-task scores and the mean.

### 3) Validate like the submission script

On Windows venvs, the CLI is typically:

```bash
.venv/Scripts/openenv.exe validate
```

You should see an `[OK] ... Ready for multi-mode deployment` message.

## LLM configuration (optional)

If you want the baseline to call a model (it falls back to heuristics otherwise):

- `MODEL_NAME` (example: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `API_BASE_URL` (defaults to `https://router.huggingface.co/v1`)
- Token (any one of):
	- `HUGGING_FACE_HUB_TOKEN` (recommended)
	- `HF_TOKEN`
	- `HUGGINGFACEHUB_API_TOKEN`
	- `HF_API_TOKEN`
	- `API_KEY`
	- `OPENAI_API_KEY`

### Using a `.env` file locally

`inference.py` will automatically load a `.env` file **if present** (via `python-dotenv`).

Example `.env` (do not commit this file):

```env
HUGGING_FACE_HUB_TOKEN=hf_...
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
```

Notes:
- Use **exact** variable names (`MODEL_NAME`, not `model_name`).
- Avoid spaces around `=` in `.env` to prevent parsing surprises.
