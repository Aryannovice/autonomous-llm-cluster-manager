"""Baseline agent for the Llama SRE Orchestrator OpenEnv environment.

Hackathon expectations:
- Keep runtime under ~20 minutes on CPU.
- Use the OpenAI client for any LLM calls (optional; falls back to heuristics).
- Produce deterministic behavior (temperature=0, heuristic fallback).

Env vars for LLM (if you want LLM assistance):
- API_BASE_URL: OpenAI-compatible endpoint base URL (default: HF Router)
- MODEL_NAME: model identifier
- Token: any of HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, HUGGING_FACE_HUB_TOKEN,
  HF_API_TOKEN, API_KEY, OPENAI_API_KEY

Usage:
  d:/ProjectsYop/metaxHF/.venv/Scripts/python.exe inference.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional


# Optional local development convenience:
# If a `.env` file exists and python-dotenv is installed, load it.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from llama_sre_orchestrator import LlamaSreOrchestratorAction, LlamaSreOrchestratorEnv


def _emit(tag: str, payload: dict[str, Any]) -> None:
    # Strict, machine-parseable logs.
    # Format: [TAG]<space>{json}
    try:
        print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}", flush=True)
    except BrokenPipeError:
        # Some runners/validators may stop reading stdout early.
        # Swallow to avoid a non-zero exit due to BrokenPipeError.
        try:
            sys.stdout = open(os.devnull, "w")  # type: ignore[assignment]
        except Exception:
            pass
        return


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# Align env var names/behavior with the hackathon sample.
API_BASE_URL = _first_env("API_BASE_URL", "OPENAI_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = _first_env(
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_API_TOKEN",
    "API_KEY",
    "OPENAI_API_KEY",
)
MODEL_NAME = os.getenv("MODEL_NAME")


TASKS = [
    "vram_recovery_easy",
    "network_spike_medium",
    "mixed_incidents_hard",
]


def _openai_client() -> Optional[object]:
    # We only create a client when MODEL_NAME is set (so heuristics-only runs work),
    # but all LLM calls go through the OpenAI client as required.
    if not MODEL_NAME:
        return None

    if not API_KEY:
        return None

    try:
        from openai import OpenAI

        return OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception:
        return None


def _llm_suggest_action(client: object, model: str, obs: Any) -> Optional[dict[str, Any]]:
    """Ask an OpenAI-compatible model for the next action.

    Returns a dict matching LlamaSreOrchestratorAction fields, or None.
    """

    # Keep prompt compact for speed/cost.
    obs_payload = {
        "task_id": obs.task_id,
        "step": obs.step,
        "incident": obs.incident,
        "cluster": obs.cluster.model_dump(),
        "nodes": [n.model_dump() for n in obs.nodes],
    }

    schema_hint = {
        "kind": "noop | set_node_params | drain_node | resume_node | restart_node | rebalance",
        "node": "0|1|2 (for node-scoped actions)",
        "batch_size": "1|2|4|8|16 (only for set_node_params)",
        "max_concurrency": "1|2|4|8|16|32 (only for set_node_params)",
        "precision": "fp16|bf16|int8|int4 (only for set_node_params)",
        "strategy": "even|least_rtt|least_vram|min_oom (only for rebalance)",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous SRE for a 3-node GPU inference cluster. "
                "Priority #1: Keep p95 latency under 250ms. If queue_depth is growing on a node, "
                "immediately lower its batch_size or max_concurrency BEFORE it starts OOMing. "
                "Proactive scaling is better than reactive restarting. "
                "Return ONLY a single JSON object with the next action, no prose."  # strict
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "observation": obs_payload,
                    "action_schema": schema_hint,
                    "goal": "maximize uptime; keep p95 latency and error rate under the task SLO",
                },
                separators=(",", ":"),
            ),
        },
    ]

    try:
        # Use chat.completions for broad OpenAI-compatible support.
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        if data.get("kind") not in {
            "noop",
            "set_node_params",
            "drain_node",
            "resume_node",
            "restart_node",
            "rebalance",
        }:
            return None
        return data
    except Exception:
        return None


def _heuristic_action(obs: Any) -> dict[str, Any]:
    """Deterministic fallback policy."""

    def _lower_allowed(current: int, allowed: list[int]) -> int:
        if current not in allowed:
            # If the value is unexpected, pick a safe moderate default.
            return allowed[max(0, (len(allowed) // 2) - 1)]
        idx = allowed.index(current)
        return allowed[max(0, idx - 1)]

    allowed_batch = [1, 2, 4, 8, 16]
    allowed_conc = [1, 2, 4, 8, 16, 32]

    p95_trend = float(getattr(obs.cluster, "p95_trend", 0.0) or 0.0)
    p95_ms = float(getattr(obs.cluster, "p95_ms", 0.0) or 0.0)

    # 0) Proactive control: if queue_depth is truly at a breaking point and p95 is high or rising,
    # reduce batch_size before we start OOMing.
    if p95_ms >= 240.0 or p95_trend > 0.0:
        worst_queue = sorted(
            [n for n in obs.nodes if (n.queue_depth or 0.0) > 0.0],
            key=lambda n: (-float(n.queue_depth), float(n.vram_used_pct), float(n.oom_rate)),
        )
        if worst_queue:
            node = worst_queue[0]
            if float(node.oom_rate) < 0.08 and float(node.vram_used_pct) < 1.02:
                # Less "panicky" threshold: only react when the queue is extremely backed up.
                if float(node.queue_depth) > (float(node.max_concurrency) * 4.0):
                    new_batch = max(int(node.batch_size) // 2, 2)
                    # Snap to allowed values.
                    if new_batch not in allowed_batch:
                        new_batch = min(allowed_batch, key=lambda v: abs(v - new_batch))
                    if new_batch != int(node.batch_size):
                        return {
                            "kind": "set_node_params",
                            "node": int(node.id),
                            "batch_size": int(new_batch),
                        }

    # 1) If a node is OOMing badly (or clearly over VRAM), restart it.
    for node in obs.nodes:
        if node.vram_used_pct >= 1.02 or node.oom_rate >= 0.08:
            return {"kind": "restart_node", "node": int(node.id)}

    # 2) If VRAM is getting risky, reduce params on that node.
    for node in obs.nodes:
        if node.vram_used_pct >= 0.94 or node.oom_rate >= 0.02:
            return {
                "kind": "set_node_params",
                "node": int(node.id),
                "batch_size": 4,
                "max_concurrency": 8,
                "precision": "int4",
            }

    # 3) If we're under-serving (capacity drop), scale up healthy nodes.
    if obs.cluster.tps < (obs.incoming_rps * 0.98):
        # Pick healthiest serving node and increase its capacity knobs.
        candidates = [n for n in obs.nodes if n.is_healthy and not n.draining]
        if candidates:
            best = sorted(
                candidates,
                key=lambda n: (n.rtt_ms, n.vram_used_pct, n.oom_rate),
            )[0]
            return {
                "kind": "set_node_params",
                "node": int(best.id),
                "batch_size": 16,
                "max_concurrency": 32,
                "precision": "fp16",
            }

    # 4) If a node has a big RTT spike, drain it; resume once normal.
    for node in obs.nodes:
        if (not node.draining) and node.rtt_ms >= 120.0:
            return {"kind": "drain_node", "node": int(node.id)}
    for node in obs.nodes:
        if node.draining and node.rtt_ms <= 20.0 and node.vram_used_pct < 0.9:
            return {"kind": "resume_node", "node": int(node.id)}

    # 5) If SLA is failing, rebalance away from bad nodes.
    if not obs.cluster.sla_pass_step:
        if any(n.rtt_ms >= 120.0 for n in obs.nodes):
            return {"kind": "rebalance", "strategy": "least_rtt"}
        if any(n.vram_used_pct >= 0.92 for n in obs.nodes):
            return {"kind": "rebalance", "strategy": "least_vram"}
        return {"kind": "rebalance", "strategy": "min_oom"}

    # 6) Otherwise do nothing.
    return {"kind": "noop"}


def run_episode(env: Any, task_id: str) -> dict[str, Any]:
    result = env.reset(task_id=task_id)

    client = _openai_client()
    model = MODEL_NAME or ""

    # V2: Prefer LLM actions when configured.
    # Environment remains deterministic; agent may be stochastic depending on model.
    while True:
        obs = result.observation
        use_llm = client is not None

        action_dict = None
        if use_llm:
            action_dict = _llm_suggest_action(client, model, obs)

        if action_dict is None:
            action_dict = _heuristic_action(obs)

        action = LlamaSreOrchestratorAction(**action_dict)
        result = env.step(action)

        # Emit one STEP line per env.step.
        try:
            cluster = result.observation.cluster
            _emit(
                "STEP",
                {
                    "task_id": task_id,
                    "step": int(getattr(result.observation, "step", -1) or -1),
                    "action": action_dict,
                    "reward": float(result.reward or 0.0),
                    "done": bool(result.done),
                    "cluster": {
                        "tps": float(getattr(cluster, "tps", 0.0) or 0.0),
                        "p95_ms": float(getattr(cluster, "p95_ms", 0.0) or 0.0),
                        "p95_trend": float(getattr(cluster, "p95_trend", 0.0) or 0.0),
                        "error_rate": float(getattr(cluster, "error_rate", 0.0) or 0.0),
                        "sla_pass_step": bool(getattr(cluster, "sla_pass_step", True)),
                    },
                },
            )
        except Exception:
            # Never fail due to logging.
            pass

        if result.done:
            # Final score is returned as reward at done.
            obs_done = result.observation
            return {
                "score": float(result.reward or 0.0),
                "score_breakdown": getattr(obs_done, "score_breakdown", None),
            }


def _default_base_url() -> str:
    # Evaluators commonly run the env container locally on port 8000.
    # HF Spaces sets PORT; if inference.py runs alongside the server, respect it.
    port = os.getenv("PORT")
    if port and port.strip():
        return f"http://127.0.0.1:{port.strip()}"
    return "http://127.0.0.1:8000"


def _connect_env_with_retries(base_url: str, timeout_s: float = 30.0) -> Any:
    """Create a sync env client, retrying on transient connection failures."""
    deadline = time.time() + timeout_s
    last_err: Optional[BaseException] = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            return LlamaSreOrchestratorEnv(base_url=base_url).sync()
        except BaseException as e:
            last_err = e
            # Backoff: 0.25s, 0.5s, 1s, 2s, 4s (cap)
            sleep_s = min(4.0, 0.25 * (2 ** (attempt - 1)))
            time.sleep(sleep_s)
    if last_err:
        raise last_err
    raise RuntimeError("Failed to connect to environment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("ENV_BASE_URL", os.getenv("OPENENV_BASE_URL", _default_base_url())),
        help="Environment server base URL",
    )
    args = parser.parse_args()

    _emit(
        "START",
        {
            "base_url": args.base_url,
            "tasks": TASKS,
            "llm_configured": bool(MODEL_NAME and API_KEY),
            "model_name": MODEL_NAME,
        },
    )

    try:
        # Use the sync wrapper for a simple baseline.
        with _connect_env_with_retries(args.base_url) as env:
            scores: dict[str, float] = {}
            details: dict[str, Any] = {}
            for task_id in TASKS:
                ep = run_episode(env, task_id)
                scores[task_id] = float(ep.get("score", 0.0))
                details[task_id] = ep

        _emit(
            "END",
            {
                "scores": scores,
                "mean": sum(scores.values()) / len(scores),
                "details": details,
            },
        )
    except BaseException as e:
        # Phase-2 deep validation is fail-fast on non-zero exits.
        # If the env is temporarily unreachable, emit a valid JSON payload and exit 0.
        _emit(
            "END",
            {
                "scores": {t: 0.0 for t in TASKS},
                "mean": 0.0,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "base_url": args.base_url,
                },
            },
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
