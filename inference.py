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
from typing import Any, Optional


# Optional local development convenience:
# If a `.env` file exists and python-dotenv is installed, load it.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from llama_sre_orchestrator import LlamaSreOrchestratorAction, LlamaSreOrchestratorEnv


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
        "strategy": "even|least_rtt|least_vram|min_oom (only for rebalance)",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous SRE for a 3-node GPU inference cluster. "
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


def run_episode(env: Any, task_id: str) -> float:
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

        if result.done:
            # Final score is returned as reward at done.
            return float(result.reward or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("ENV_BASE_URL", "http://localhost:8000"),
        help="Environment server base URL",
    )
    args = parser.parse_args()

    # Use the sync wrapper for a simple baseline.
    with LlamaSreOrchestratorEnv(base_url=args.base_url).sync() as env:
        scores: dict[str, float] = {}
        for task_id in TASKS:
            scores[task_id] = run_episode(env, task_id)

    print(json.dumps({"scores": scores, "mean": sum(scores.values()) / len(scores)}, indent=2))


if __name__ == "__main__":
    main()
