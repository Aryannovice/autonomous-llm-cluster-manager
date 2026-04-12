"""Baseline agent for the Llama SRE Orchestrator OpenEnv environment.

Submission contract (stdout):
- One [START], multiple [STEP], and one [END] per task episode.
- [START] uses a single task id, not a comma-separated task list.
- Printed reward/score values stay strictly in [0.01, 0.99].

Environment variables:
- API_BASE_URL: LLM endpoint (default: Hugging Face router OpenAI-compatible URL)
- MODEL_NAME: model id (default: gpt-4o-mini)
- HF_TOKEN: mandatory; used as OpenAI client api_key

Usage:
  python inference.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Optional local development convenience:
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from llama_sre_orchestrator import LlamaSreOrchestratorAction, LlamaSreOrchestratorEnv


# --- Required env (submission spec) -------------------------------------------------
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None or not str(HF_TOKEN).strip():
    raise ValueError("HF_TOKEN environment variable is required")

SCORE_EPS = 1e-2

# region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug-f39562.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "f39562",
            "runId": "submission",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass


# endregion


def _clamp01_strict(x: float, eps: float = SCORE_EPS) -> float:
    """Interior (0,1) for internal episode scores when needed."""
    try:
        x = float(x)
    except Exception:
        x = 0.0
    if x != x:
        x = 0.0
    raw = max(0.0, min(1.0, x))
    safe = (raw * 0.98) + 0.01
    return float(round(safe, 4))


def _validator_reward_display(x: Any) -> float:
    """Every printed reward/score must be strictly inside (0,1): use [0.01, 0.99]."""
    try:
        v = float(x if x is not None else 0.5)
    except Exception:
        v = 0.5
    if v != v:
        v = 0.5
    return float(max(0.01, min(0.99, v)))


def _safe_print(line: str) -> None:
    try:
        print(line, flush=True)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)


def _fmt_error_field(msg: Optional[str]) -> str:
    if msg is None or msg == "":
        return "null"
    return json.dumps(msg, ensure_ascii=False)


def emit_start(*, task: str, env_name: str, model: str) -> None:
    _safe_print(f"[START] task={task} env={env_name} model={model}")


def emit_step(
    *,
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    last_action_error: Optional[str],
) -> None:
    err = _fmt_error_field(last_action_error)
    done_s = "true" if done else "false"
    r = _validator_reward_display(reward)
    _safe_print(
        f"[STEP] step={step} action={action_str} reward={r:.2f} done={done_s} error={err}"
    )


def emit_end(*, task: str, success: bool, steps: int, score: float) -> None:
    """[END] success=… steps=… score=… rewards=<t1>,<t2>,<t3> — three task-level scores."""
    succ = "true" if success else "false"
    sc = _validator_reward_display(score)
    _safe_print(f"[END] task={task} success={succ} steps={steps} score={sc:.3f}")


def _action_to_str(action_dict: dict[str, Any]) -> str:
    """Single-line action description; compact JSON has no spaces when possible."""
    return json.dumps(action_dict, separators=(",", ":"), ensure_ascii=False)


def _last_action_error_from_obs(obs: Any) -> Optional[str]:
    """Spec: raw last_action_error; env exposes incident / optional attribute."""
    err = getattr(obs, "last_action_error", None)
    if err is not None and str(err).strip():
        return str(err)
    return None


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw is not None else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw is not None else int(default)
    except Exception:
        return int(default)


LATENCY_ENTER_MS = _env_float("LATENCY_ENTER_MS", 292.0)
LATENCY_RECOVERY_MS = _env_float("LATENCY_RECOVERY_MS", 255.0)
TREND_ENTER_MS = _env_float("TREND_ENTER_MS", 0.10)
TREND_RECOVERY_MS = _env_float("TREND_RECOVERY_MS", 0.02)
QUEUE_LIMIT_HARD_MULT = _env_float("QUEUE_LIMIT_HARD_MULT", 3.0)
QUEUE_LIMIT_SOFT_MULT = _env_float("QUEUE_LIMIT_SOFT_MULT", 3.6)
STABLE_CADENCE_STEPS = max(1, _env_int("STABLE_CADENCE_STEPS", 4))
UNSTABLE_CADENCE_STEPS = max(1, _env_int("UNSTABLE_CADENCE_STEPS", 1))
RESTART_PERSIST_STEPS = max(1, _env_int("RESTART_PERSIST_STEPS", 4))
LLM_RETRY_MAX = max(1, _env_int("LLM_RETRY_MAX", 3))
LLM_BACKOFF_BASE_S = max(0.0, _env_float("LLM_BACKOFF_BASE_S", 0.10))
OPENAI_TIMEOUT_S = max(1.0, _env_float("OPENAI_TIMEOUT_S", 15.0))

TASKS = [
    "vram_recovery_easy",
    "network_spike_medium",
    "mixed_incidents_hard",
]
BENCHMARK_ENV_NAME = "llama_sre_orchestrator"


def _openai_client() -> Any:
    from openai import OpenAI

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        timeout=OPENAI_TIMEOUT_S,
    )


def _llm_suggest_action(client: object, model: str, obs: Any) -> Optional[dict[str, Any]]:
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
                "Return ONLY a single JSON object with the next action, no prose."
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
    def _lower_allowed(current: int, allowed: list[int]) -> int:
        if current not in allowed:
            return allowed[max(0, (len(allowed) // 2) - 1)]
        idx = allowed.index(current)
        return allowed[max(0, idx - 1)]

    allowed_batch = [1, 2, 4, 8, 16]
    allowed_conc = [1, 2, 4, 8, 16, 32]

    p95_trend = float(getattr(obs.cluster, "p95_trend", 0.0) or 0.0)
    p95_ms = float(getattr(obs.cluster, "p95_ms", 0.0) or 0.0)
    incoming_rps = float(getattr(obs, "incoming_rps", 0.0) or 0.0)
    tps = float(getattr(obs.cluster, "tps", 0.0) or 0.0)

    enter_latency_control = (p95_ms >= LATENCY_ENTER_MS) or (p95_trend > TREND_ENTER_MS)
    in_recovery_band = (p95_ms >= LATENCY_RECOVERY_MS) and (p95_trend > TREND_RECOVERY_MS)
    if enter_latency_control or in_recovery_band:
        worst_queue = sorted(
            [n for n in obs.nodes if (n.queue_depth or 0.0) > 0.0],
            key=lambda n: (-float(n.queue_depth), float(n.vram_used_pct), float(n.oom_rate)),
        )
        if worst_queue:
            node = worst_queue[0]
            if float(node.oom_rate) < 0.10 and float(node.vram_used_pct) < 1.03:
                queue_limit = float(node.max_concurrency) * (
                    QUEUE_LIMIT_HARD_MULT if p95_ms >= (LATENCY_ENTER_MS + 20.0) else QUEUE_LIMIT_SOFT_MULT
                )
                if float(node.queue_depth) > queue_limit:
                    new_conc = _lower_allowed(int(node.max_concurrency), allowed_conc)
                    if new_conc != int(node.max_concurrency):
                        return {
                            "kind": "set_node_params",
                            "node": int(node.id),
                            "max_concurrency": int(new_conc),
                        }
                    new_batch = max(int(node.batch_size) // 2, 2)
                    if new_batch not in allowed_batch:
                        new_batch = min(allowed_batch, key=lambda v: abs(v - new_batch))
                    if new_batch != int(node.batch_size):
                        return {
                            "kind": "set_node_params",
                            "node": int(node.id),
                            "batch_size": int(new_batch),
                        }

    for node in obs.nodes:
        if node.vram_used_pct >= 1.08 or node.oom_rate >= 0.15:
            return {"kind": "restart_node", "node": int(node.id)}

    for node in obs.nodes:
        if node.vram_used_pct >= 0.94 or node.oom_rate >= 0.02:
            return {
                "kind": "set_node_params",
                "node": int(node.id),
                "batch_size": 4,
                "max_concurrency": 8,
                "precision": "int4",
            }

    if tps < (incoming_rps * 0.97):
        for node in obs.nodes:
            if node.draining and node.rtt_ms <= 35.0 and node.vram_used_pct < 0.95:
                return {"kind": "resume_node", "node": int(node.id)}

        candidates = [n for n in obs.nodes if n.is_healthy and not n.draining]
        if candidates:
            best = sorted(
                candidates,
                key=lambda n: (n.rtt_ms, n.vram_used_pct, n.oom_rate),
            )[0]
            spike = incoming_rps > max(1.0, tps) * 1.2
            target_batch = 16 if spike else 8
            target_conc = 32 if spike else 16
            target_precision = "fp16" if (spike and best.vram_used_pct < 0.88) else "int8"
            return {
                "kind": "set_node_params",
                "node": int(best.id),
                "batch_size": target_batch,
                "max_concurrency": target_conc,
                "precision": target_precision,
            }

    for node in obs.nodes:
        if (not node.draining) and node.rtt_ms >= 120.0:
            return {"kind": "drain_node", "node": int(node.id)}
    for node in obs.nodes:
        if node.draining and node.rtt_ms <= 20.0 and node.vram_used_pct < 0.9:
            return {"kind": "resume_node", "node": int(node.id)}

    if not obs.cluster.sla_pass_step:
        if any(n.rtt_ms >= 120.0 for n in obs.nodes):
            return {"kind": "rebalance", "strategy": "least_rtt"}
        if any(n.vram_used_pct >= 0.92 for n in obs.nodes):
            return {"kind": "rebalance", "strategy": "least_vram"}
        return {"kind": "rebalance", "strategy": "min_oom"}

    return {"kind": "noop"}


def run_episode(
    env: Any,
    task_id: str,
    client: object,
    model: str,
) -> dict[str, Any]:
    """Run one task; emit one [STEP] line per env.step()."""
    result = env.reset(task_id=task_id)
    restart_pressure_count: dict[int, int] = {0: 0, 1: 0, 2: 0}

    while True:
        obs = result.observation

        action_dict = None
        p95_ms = float(getattr(obs.cluster, "p95_ms", 0.0) or 0.0)
        p95_trend = float(getattr(obs.cluster, "p95_trend", 0.0) or 0.0)
        error_rate = float(getattr(obs.cluster, "error_rate", 0.0) or 0.0)
        queue_peak = max([float(getattr(n, "queue_depth", 0.0) or 0.0) for n in obs.nodes] or [0.0])
        unstable = (
            (p95_ms >= LATENCY_ENTER_MS)
            or (p95_trend > TREND_ENTER_MS)
            or (error_rate > 0.010)
            or (queue_peak > 70.0)
            or (not bool(getattr(obs.cluster, "sla_pass_step", True)))
        )
        cadence = UNSTABLE_CADENCE_STEPS if unstable else STABLE_CADENCE_STEPS
        should_query_llm = (int(getattr(obs, "step", 0) or 0) % cadence) == 0

        if should_query_llm:
            backoffs = [LLM_BACKOFF_BASE_S * (2**i) for i in range(LLM_RETRY_MAX)]
            for attempt in range(LLM_RETRY_MAX):
                action_dict = _llm_suggest_action(client, model, obs)
                if action_dict is not None:
                    break
                if attempt < LLM_RETRY_MAX - 1:
                    time.sleep(backoffs[attempt])

        if action_dict is None:
            action_dict = _heuristic_action(obs)

        if action_dict.get("kind") == "restart_node":
            node_id = int(action_dict.get("node", -1))
            target_node = next((n for n in obs.nodes if int(getattr(n, "id", -1)) == node_id), None)
            severe = False
            if target_node is not None:
                cluster_error = float(getattr(obs.cluster, "error_rate", 0.0) or 0.0)
                severe = (
                    (
                        float(getattr(target_node, "vram_used_pct", 0.0) or 0.0) >= 1.08
                        or float(getattr(target_node, "oom_rate", 0.0) or 0.0) >= 0.15
                    )
                    and cluster_error >= 0.10
                )
            if node_id not in restart_pressure_count:
                restart_pressure_count[node_id] = 0
            restart_pressure_count[node_id] = (restart_pressure_count[node_id] + 1) if severe else 0
            if restart_pressure_count[node_id] < RESTART_PERSIST_STEPS:
                if target_node is not None:
                    action_dict = {
                        "kind": "set_node_params",
                        "node": node_id,
                        "batch_size": 4,
                        "max_concurrency": 8,
                        "precision": "int4",
                    }
                else:
                    action_dict = {"kind": "rebalance", "strategy": "min_oom"}

        action = LlamaSreOrchestratorAction(**action_dict)
        result = env.step(action)

        obs_after = result.observation
        cur_step = int(getattr(obs_after, "step", 0) or 0)
        err = _last_action_error_from_obs(obs_after)

        emit_step(
            step=cur_step,
            action_str=_action_to_str(action_dict),
            reward=float(result.reward if result.reward is not None else 0.0),
            done=bool(result.done),
            last_action_error=err,
        )

        if result.done:
            obs_done = result.observation
            score_breakdown = getattr(obs_done, "score_breakdown", None)
            if isinstance(score_breakdown, dict):
                wb = score_breakdown.get("weighted_total")
                if wb is not None:
                    score_breakdown = dict(score_breakdown)
                    score_breakdown["weighted_total"] = _clamp01_strict(float(wb))
            return {
                "score": _clamp01_strict(float(result.reward or 0.0)),
                "score_breakdown": score_breakdown,
                "steps": int(getattr(obs_done, "step", cur_step) or cur_step),
            }


def _default_base_url() -> str:
    port = os.getenv("PORT")
    if port and port.strip():
        return f"http://127.0.0.1:{port.strip()}"
    return "http://127.0.0.1:8000"


def _connect_env_with_retries(base_url: str, timeout_s: float = 30.0) -> Any:
    deadline = time.time() + timeout_s
    last_err: Optional[BaseException] = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            return LlamaSreOrchestratorEnv(base_url=base_url).sync()
        except BaseException as e:
            last_err = e
            sleep_s = min(4.0, 0.25 * (2 ** (attempt - 1)))
            time.sleep(sleep_s)
    if last_err:
        raise last_err
    raise RuntimeError("Failed to connect to environment")


def main() -> None:
    _debug_log(
        "H6",
        "inference.py:main",
        "inference main entered",
        {
            "cwd": os.getcwd(),
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "has_hf_token": bool(HF_TOKEN),
        },
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("ENV_BASE_URL", os.getenv("OPENENV_BASE_URL", _default_base_url())),
        help="Environment server base URL",
    )
    args = parser.parse_args()

    client = _openai_client()
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": '{"ping":"proxy"}'},
            ],
            temperature=0,
            max_tokens=16,
        )
    except Exception:
        try:
            client.models.list()
        except Exception:
            pass

    emitted_tasks: set[str] = set()
    try:
        with _connect_env_with_retries(args.base_url) as env:
            for task_id in TASKS:
                emit_start(task=task_id, env_name=BENCHMARK_ENV_NAME, model=MODEL_NAME)
                emitted_tasks.add(task_id)
                try:
                    ep = run_episode(
                        env,
                        task_id,
                        client,
                        MODEL_NAME,
                    )
                    emit_end(
                        task=task_id,
                        success=True,
                        steps=int(ep.get("steps", 60)),
                        score=float(ep.get("score", 0.5)),
                    )
                except BaseException:
                    emit_step(
                        step=0,
                        action_str='{"kind":"noop"}',
                        reward=0.01,
                        done=True,
                        last_action_error="task_execution_failed",
                    )
                    emit_end(task=task_id, success=False, steps=0, score=0.01)
                    raise
    except BaseException:
        for task_id in TASKS:
            if task_id in emitted_tasks:
                continue
            emit_start(task=task_id, env_name=BENCHMARK_ENV_NAME, model=MODEL_NAME)
            emit_step(
                step=0,
                action_str='{"kind":"noop"}',
                reward=0.01,
                done=True,
                last_action_error="env_connection_failed",
            )
            emit_end(task=task_id, success=False, steps=0, score=0.01)

    sys.exit(0)


if __name__ == "__main__":
    main()
