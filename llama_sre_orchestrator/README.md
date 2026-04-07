---
title: Llama Sre Orchestrator Environment Server
emoji: 🎼
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Llama Sre Orchestrator Environment

An OpenEnv environment where an agent acts as an autonomous SRE for a deterministic
**3-node GPU inference cluster simulator**. The goal is to keep SLOs while handling
incidents like VRAM leaks and network RTT spikes.

This Space exposes:
- Web UI at `/web`
- OpenAPI docs at `/docs`
- WebSocket endpoint at `/ws`

## Quick Start

The simplest way to use the environment is via `LlamaSreOrchestratorEnv`:

```python
from llama_sre_orchestrator import LlamaSreOrchestratorAction, LlamaSreOrchestratorEnv

try:
    env = LlamaSreOrchestratorEnv.from_docker_image("llama_sre_orchestrator-env:latest")

    result = env.reset(task_id="vram_recovery_easy")
    print("Task:", result.observation.task_id)

    while True:
        # Minimal baseline: do nothing
        result = env.step(LlamaSreOrchestratorAction(kind="noop"))
        if result.done:
            print("Final score:", result.reward)
            break

finally:
    # Always clean up
    env.close()
```

That's it! The `LlamaSreOrchestratorEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t llama_sre_orchestrator-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Tasks

Pick one per episode with `reset(task_id=...)`:
- `vram_recovery_easy`: VRAM leak on node 1
- `network_spike_medium`: RTT spike on node 2 + throughput throttle on node 0
- `mixed_incidents_hard`: combined incidents with stricter SLO (p95 250ms, error 0.5%)

### Action
`LlamaSreOrchestratorAction` is structured JSON with a bounded action space:

- `kind`: `noop | set_node_params | drain_node | resume_node | restart_node | rebalance`
- `node`: `0|1|2` for node-scoped actions
- `batch_size`: `1|2|4|8|16` (for `set_node_params`)
- `max_concurrency`: `1|2|4|8|16|32` (for `set_node_params`)
- `strategy`: `even|least_rtt|least_vram|min_oom` (for `rebalance`)

### Observation
`LlamaSreOrchestratorObservation` provides cluster + per-node metrics plus episode-level
scoring stats when `done=True`.

See the repo-root `inference.py` for a baseline agent that runs all tasks.
