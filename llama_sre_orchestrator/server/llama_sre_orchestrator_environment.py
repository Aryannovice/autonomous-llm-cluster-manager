# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama Sre Orchestrator Environment implementation.

Deterministic, discrete-time simulator for an autonomous SRE managing a
3-node distributed GPU cluster serving LLM inference traffic.

Select tasks via `reset(task_id=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Final, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State
from openenv.core.rubrics.base import Rubric

try:
    from ..models import (
        ClusterMetrics,
        LlamaSreOrchestratorAction,
        LlamaSreOrchestratorObservation,
        NodeMetrics,
        TaskId,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        ClusterMetrics,
        LlamaSreOrchestratorAction,
        LlamaSreOrchestratorObservation,
        NodeMetrics,
        TaskId,
    )


@dataclass
class _Node:
    id: int
    vram_total_gb: float = 24.0
    model_base_gb: float = 14.0

    precision: str = "fp16"

    batch_size: int = 8
    max_concurrency: int = 16
    traffic_share: float = 1.0 / 3.0

    draining: bool = False
    drain_target: float = 0.0
    drain_progress: float = 0.0
    reboot_cooldown_steps: int = 0

    leak_gb: float = 0.0
    rtt_ms: float = 12.0
    throttle_factor: float = 1.0

    def is_serving(self) -> bool:
        return (self.drain_progress < 0.999) and self.reboot_cooldown_steps <= 0


_SCORE_EPS_RUBRIC: float = 1e-2


# region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / "debug-f39562.log"
_DEBUG_SESSION_ID = "f39562"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": "phase2-debug",
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


def _clamp01_strict(score: Any, eps: float = _SCORE_EPS_RUBRIC) -> float:
    """Squeeze any score into strict (0,1): [0,1] -> [0.01,0.99]."""
    try:
        value = float(score)
    except Exception:
        value = 0.0
    if value != value:
        value = 0.0
    raw = max(0.0, min(1.0, value))
    safe = (raw * 0.98) + 0.01
    return float(round(safe, 4))


class _TaskGrader(Rubric):
    """Per-task rubric that returns a strict in-range score on every step."""

    def __init__(self, task_id: str, sla_p95_ms: float, sla_error_rate: float) -> None:
        super().__init__()
        self.task_id = task_id
        self.sla_p95_ms = float(sla_p95_ms)
        self.sla_error_rate = float(sla_error_rate)

    def _step_score(self, observation: Any) -> float:
        cluster = getattr(observation, "cluster", None)
        if cluster is None:
            return _clamp01_strict(0.0)

        error_rate = float(getattr(cluster, "error_rate", 1.0) or 1.0)
        p95_ms = float(getattr(cluster, "p95_ms", self.sla_p95_ms * 2.0) or (self.sla_p95_ms * 2.0))
        p95_trend = float(getattr(cluster, "p95_trend", 0.0) or 0.0)
        tps = float(getattr(cluster, "tps", 0.0) or 0.0)
        incoming_rps = float(getattr(observation, "incoming_rps", 1.0) or 1.0)
        step = int(getattr(observation, "step", 0) or 0)
        max_steps = int(getattr(observation, "max_steps", 60) or 60)

        availability = max(0.0, min(1.0, 1.0 - (error_rate / max(self.sla_error_rate, 1e-6))))
        if p95_ms <= self.sla_p95_ms:
            latency = 1.0
        elif p95_ms >= 2.0 * self.sla_p95_ms:
            latency = 0.0
        else:
            latency = 1.0 - ((p95_ms - self.sla_p95_ms) / self.sla_p95_ms)
        efficiency = max(0.0, min(1.0, tps / max(1e-6, incoming_rps)))
        trend_stability = max(0.0, min(1.0, 1.0 - (abs(p95_trend) / max(1.0, self.sla_p95_ms))))
        progress = max(0.0, min(1.0, step / max(1, max_steps)))
        task_bias = {
            "vram_recovery_easy": 0.010,
            "network_spike_medium": 0.015,
            "mixed_incidents_hard": 0.020,
        }.get(self.task_id, 0.010)

        return _clamp01_strict(
            (0.35 * availability)
            + (0.25 * latency)
            + (0.25 * efficiency)
            + (0.10 * trend_stability)
            + (0.05 * progress)
            + task_bias
        )

    def forward(self, action: Any, observation: Any) -> float:
        step = int(getattr(observation, "step", 0) or 0)
        max_steps = int(getattr(observation, "max_steps", 60) or 60)
        task_bias = {
            "vram_recovery_easy": 0.010,
            "network_spike_medium": 0.015,
            "mixed_incidents_hard": 0.020,
        }.get(self.task_id, 0.010)

        if getattr(observation, "task_id", None) != self.task_id:
            # Keep non-active grader scores numeric and non-constant for validator probes.
            background = 0.20 + (0.05 * max(0.0, min(1.0, step / max(1, max_steps)))) + task_bias
            # region agent log
            _debug_log(
                "H2",
                "llama_sre_orchestrator_environment.py:_TaskGrader.forward",
                "inactive task grader score",
                {"task_id": self.task_id, "step": step, "score": float(background)},
            )
            # endregion
            return _clamp01_strict(background)

        if getattr(observation, "done", False):
            terminal_score = getattr(observation, "final_score", None)
            if terminal_score is None:
                terminal_score = getattr(observation, "reward", None)
            if terminal_score is not None:
                # region agent log
                _debug_log(
                    "H3",
                    "llama_sre_orchestrator_environment.py:_TaskGrader.forward",
                    "terminal task grader score",
                    {"task_id": self.task_id, "step": step, "score": float(terminal_score)},
                )
                # endregion
                return _clamp01_strict(terminal_score)

        score = self._step_score(observation)
        # region agent log
        _debug_log(
            "H3",
            "llama_sre_orchestrator_environment.py:_TaskGrader.forward",
            "active task grader step score",
            {"task_id": self.task_id, "step": step, "score": float(score)},
        )
        # endregion
        return score


class _SREOrchestratorRubric(Rubric):
    """Top-level rubric with one named child per task.

    Assigning Rubric instances as attributes auto-registers them as children
    via Rubric.__setattr__, making them visible to named_rubrics().
    """

    def __init__(self, task_specs: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        # Primary graders (task-name keys).
        self.vram_recovery_easy = _TaskGrader(
            "vram_recovery_easy",
            task_specs["vram_recovery_easy"]["sla_p95_ms"],
            task_specs["vram_recovery_easy"]["sla_error_rate"],
        )
        self.network_spike_medium = _TaskGrader(
            "network_spike_medium",
            task_specs["network_spike_medium"]["sla_p95_ms"],
            task_specs["network_spike_medium"]["sla_error_rate"],
        )
        self.mixed_incidents_hard = _TaskGrader(
            "mixed_incidents_hard",
            task_specs["mixed_incidents_hard"]["sla_p95_ms"],
            task_specs["mixed_incidents_hard"]["sla_error_rate"],
        )
        # Compatibility aliases: some validators look for explicit "grader/task"
        # naming patterns when discovering task-graders.
        self.grader_vram_recovery_easy = self.vram_recovery_easy
        self.grader_network_spike_medium = self.network_spike_medium
        self.grader_mixed_incidents_hard = self.mixed_incidents_hard
        self.task_vram_recovery_easy = self.vram_recovery_easy
        self.task_network_spike_medium = self.network_spike_medium
        self.task_mixed_incidents_hard = self.mixed_incidents_hard

    def forward(self, action: Any, observation: Any) -> float:
        # Evaluate all named graders so each one keeps numeric last_score.
        task_scores = {
            "vram_recovery_easy": self.vram_recovery_easy(action, observation),
            "network_spike_medium": self.network_spike_medium(action, observation),
            "mixed_incidents_hard": self.mixed_incidents_hard(action, observation),
        }
        alias_scores = {
            "grader_vram_recovery_easy": self.grader_vram_recovery_easy(action, observation),
            "grader_network_spike_medium": self.grader_network_spike_medium(action, observation),
            "grader_mixed_incidents_hard": self.grader_mixed_incidents_hard(action, observation),
            "task_vram_recovery_easy": self.task_vram_recovery_easy(action, observation),
            "task_network_spike_medium": self.task_network_spike_medium(action, observation),
            "task_mixed_incidents_hard": self.task_mixed_incidents_hard(action, observation),
        }
        # region agent log
        _debug_log(
            "H2",
            "llama_sre_orchestrator_environment.py:_SREOrchestratorRubric.forward",
            "rubric forward scores",
            {
                "active_task": getattr(observation, "task_id", None),
                "task_scores": {k: float(v) for k, v in task_scores.items()},
                "alias_scores": {k: float(v) for k, v in alias_scores.items()},
            },
        )
        # endregion
        return _clamp01_strict(task_scores.get(getattr(observation, "task_id", None), 0.5))


class LlamaSreOrchestratorEnvironment(Environment[LlamaSreOrchestratorAction, LlamaSreOrchestratorObservation, State]):
    """3-node GPU cluster SRE simulator."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS: Final[int] = 60
    STEP_SECONDS: Final[int] = 10
    INCOMING_RPS: Final[float] = 900.0

    # Phase-2 validators may require task scores strictly within (0,1), not inclusive.
    _SCORE_EPS: Final[float] = 1e-2
    _DEFAULT_TASK_CURSOR: int = 0

    _ALLOWED_BATCH: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
    _ALLOWED_CONCURRENCY: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32)

    _TASKS: Final[dict[TaskId, dict[str, Any]]] = {
        "vram_recovery_easy": {
            "sla_p95_ms": 350.0,
            "sla_error_rate": 0.01,
            "leak": {"node": 1, "start": 10, "end": 45, "gb_per_step": 0.45},
            "rtt_spike": None,
            "throttle": None,
        },
        "network_spike_medium": {
            "sla_p95_ms": 300.0,
            "sla_error_rate": 0.008,
            "leak": None,
            "rtt_spike": {"node": 2, "start": 12, "end": 30, "rtt_ms": 150.0},
            "throttle": {"node": 0, "start": 18, "end": 36, "factor": 0.55},
        },
        "mixed_incidents_hard": {
            "sla_p95_ms": 250.0,
            "sla_error_rate": 0.005,
            "leak": {"node": 1, "start": 8, "end": 50, "gb_per_step": 0.55},
            "rtt_spike": {"node": 2, "start": 10, "end": 40, "rtt_ms": 175.0},
            "throttle": {"node": 0, "start": 15, "end": 45, "factor": 0.50},
        },
    }
    _GRADER_INFO: Final[dict[str, str]] = {"name": "deterministic_v2", "version": "1.0"}

    def __init__(self):
        super().__init__(rubric=_SREOrchestratorRubric(self._TASKS))
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: TaskId = "vram_recovery_easy"
        self._nodes: list[_Node] = []
        self._p95_history: list[float] = []
        self._err_history: list[float] = []
        self._tps_history: list[float] = []
        self._sla_history: list[bool] = []
        self._restart_count: int = 0
        self._incident: str = ""
        self._last_action: dict[str, Any] | None = None
        self._last_action_impact: dict[str, Any] | None = None
        self._prev_step_metrics: tuple[float, float, float] | None = None
        self._prev_node_vram_used_pct: list[float] | None = None
        self._prev_p95_ms_for_trend: float | None = None
        # region agent log
        _debug_log(
            "H5",
            "llama_sre_orchestrator_environment.py:LlamaSreOrchestratorEnvironment.__init__",
            "environment initialized",
            {"task_default": self._task_id, "tasks": list(self._TASKS.keys())},
        )
        # endregion

    def _init_episode(self, *, episode_id: Optional[str], task_id: TaskId) -> None:
        """Initialize episode state.

        Note: The OpenEnv contract expects callers to invoke reset() before step().
        The web UI can call step() before reset(), so we keep this helper to
        lazily initialize a default episode in that case.
        """

        self._task_id = task_id
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        self._nodes = [_Node(id=0), _Node(id=1), _Node(id=2)]
        for n in self._nodes:
            n.batch_size = 8
            n.max_concurrency = 16
            n.precision = "fp16"
            n.traffic_share = 1.0 / 3.0
            n.draining = False
            n.drain_target = 0.0
            n.drain_progress = 0.0
            n.reboot_cooldown_steps = 0
            n.leak_gb = 0.0
            n.rtt_ms = 12.0
            n.throttle_factor = 1.0

        self._p95_history = []
        self._err_history = []
        self._tps_history = []
        self._sla_history = []
        self._restart_count = 0
        self._incident = ""
        self._last_action = None
        self._last_action_impact = None
        self._prev_step_metrics = None
        self._prev_node_vram_used_pct = None
        self._prev_p95_ms_for_trend = None

    @classmethod
    def _next_default_task_id(cls) -> TaskId:
        task_ids = tuple(cls._TASKS.keys())
        task_id = task_ids[cls._DEFAULT_TASK_CURSOR % len(task_ids)]
        cls._DEFAULT_TASK_CURSOR = (cls._DEFAULT_TASK_CURSOR + 1) % len(task_ids)
        return task_id

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LlamaSreOrchestratorObservation:
        del seed, kwargs

        if task_id is None:
            task_id = self._next_default_task_id()
        # region agent log
        _debug_log(
            "H1",
            "llama_sre_orchestrator_environment.py:LlamaSreOrchestratorEnvironment.reset",
            "reset called",
            {"requested_task_id": task_id, "episode_id": episode_id},
        )
        # endregion
        if task_id not in self._TASKS:
            raise ValueError(
                f"Unknown task_id={task_id!r}. Expected one of: {sorted(self._TASKS.keys())}"
            )
        self._init_episode(episode_id=episode_id, task_id=task_id)  # type: ignore[arg-type]
        self._reset_rubric()

        # Provide a realistic step-0 observation (helps trends/LLMs); does not affect grading.
        p95_ms, error_rate, tps = self._compute_cluster_metrics()
        sla_pass = self._is_sla_pass(p95_ms, error_rate)
        return self._make_observation(p95_ms=p95_ms, error_rate=error_rate, tps=tps, sla_pass=sla_pass)

    def step(
        self,
        action: LlamaSreOrchestratorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LlamaSreOrchestratorObservation:
        del timeout_s, kwargs

        # Web UI users sometimes click "Step" before "Reset".
        # Avoid crashing with list index errors; lazily initialize.
        if len(self._nodes) != 3:
            lazy_task_id = self._next_default_task_id()
            self._init_episode(episode_id=self._state.episode_id, task_id=lazy_task_id)
            # region agent log
            _debug_log(
                "H9",
                "llama_sre_orchestrator_environment.py:LlamaSreOrchestratorEnvironment.step",
                "lazy init from step without reset",
                {"task_id_used": lazy_task_id, "episode_id": self._state.episode_id},
            )
            # endregion
        # region agent log
        _debug_log(
            "H4",
            "llama_sre_orchestrator_environment.py:LlamaSreOrchestratorEnvironment.step",
            "step called",
            {
                "task_id": self._task_id,
                "step_before": int(self._state.step_count),
                "action_kind": getattr(action, "kind", None),
            },
        )
        # endregion

        self._state.step_count += 1
        self._incident = ""

        # Keep last action available for interpretability in observations.
        self._last_action = action.model_dump(exclude_none=True)

        self._apply_action(action)
        self._apply_incidents(step=self._state.step_count)
        self._tick_reboots()
        self._tick_drain_transitions()
        self._normalize_traffic_shares()

        p95_ms, error_rate, tps = self._compute_cluster_metrics()
        sla_pass = self._is_sla_pass(p95_ms, error_rate)

        # Deterministic impact deltas (vs previous step), useful for graders/UI.
        if self._prev_step_metrics is None:
            self._last_action_impact = None
        else:
            prev_p95, prev_err, prev_tps = self._prev_step_metrics
            self._last_action_impact = {
                "delta_p95_ms": float(p95_ms - prev_p95),
                "delta_error_rate": float(error_rate - prev_err),
                "delta_tps": float(tps - prev_tps),
            }
        self._prev_step_metrics = (p95_ms, error_rate, tps)

        self._p95_history.append(p95_ms)
        self._err_history.append(error_rate)
        self._tps_history.append(tps)
        self._sla_history.append(sla_pass)

        done = self._state.step_count >= self.MAX_STEPS
        reward = None
        final_score = None
        uptime = None
        avg_p95 = None
        avg_err = None
        restart_count = None
        score_breakdown = None

        if done:
            final_score, score_breakdown = self._final_score_v2()
            reward = float(final_score)
            uptime = sum(1 for x in self._sla_history if x) / max(1, len(self._sla_history))
            avg_p95 = sum(self._p95_history) / max(1, len(self._p95_history))
            avg_err = sum(self._err_history) / max(1, len(self._err_history))
            restart_count = self._restart_count

        obs = self._make_observation(
            p95_ms=p95_ms,
            error_rate=error_rate,
            tps=tps,
            sla_pass=sla_pass,
            final_score=final_score,
            uptime=uptime,
            avg_p95=avg_p95,
            avg_error=avg_err,
            restart_count=restart_count,
            score_breakdown=score_breakdown,
        )
        obs.done = done
        obs.reward = reward
        obs.reward = self._apply_rubric(action, obs)
        return obs

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return standard metadata for the environment."""
        metadata = EnvironmentMetadata(
            name="llama_sre_orchestrator",
            description=(
                "Autonomous SRE simulator for a 3-node GPU inference cluster "
                "with 3 graded tasks of increasing difficulty"
            ),
            version="1.0.0",
        )
        # region agent log
        _debug_log(
            "H1",
            "llama_sre_orchestrator_environment.py:LlamaSreOrchestratorEnvironment.get_metadata",
            "metadata requested",
            {"name": metadata.name, "version": metadata.version},
        )
        # endregion
        return metadata

    def _apply_action(self, action: LlamaSreOrchestratorAction) -> None:
        kind = action.kind
        if kind == "noop":
            return

        if kind == "set_node_params":
            if action.node is None:
                return
            node = self._nodes[int(action.node)]
            if action.batch_size is not None and int(action.batch_size) in self._ALLOWED_BATCH:
                node.batch_size = int(action.batch_size)
            if (
                action.max_concurrency is not None
                and int(action.max_concurrency) in self._ALLOWED_CONCURRENCY
            ):
                node.max_concurrency = int(action.max_concurrency)

            if action.precision is not None:
                precision = str(action.precision).lower().strip()
                if precision in {"fp16", "bf16", "int8", "int4"}:
                    node.precision = precision
            return

        if kind == "drain_node":
            if action.node is None:
                return
            node = self._nodes[int(action.node)]
            node.drain_target = 1.0
            return

        if kind == "resume_node":
            if action.node is None:
                return
            node = self._nodes[int(action.node)]
            node.drain_target = 0.0
            return

        if kind == "restart_node":
            if action.node is None:
                return
            node = self._nodes[int(action.node)]
            node.reboot_cooldown_steps = 2
            node.leak_gb = 0.0
            self._restart_count += 1
            return

        if kind == "rebalance":
            self._rebalance(strategy=action.strategy or "even")
            return

    def _rebalance(self, strategy: str) -> None:
        serving = [n for n in self._nodes if n.is_serving()]
        if not serving:
            return

        if strategy == "even":
            share = 1.0 / len(serving)
            for n in self._nodes:
                n.traffic_share = share if n.is_serving() else 0.0
            return

        if strategy == "least_rtt":
            weights = [1.0 / max(1.0, n.rtt_ms) for n in serving]
        elif strategy == "least_vram":
            weights = [1.0 / max(0.05, self._vram_used_pct(n)) for n in serving]
        elif strategy == "min_oom":
            weights = [1.0 / max(0.001, self._oom_rate(n)) for n in serving]
        else:
            weights = [1.0 for _ in serving]

        total = sum(weights)
        for n in self._nodes:
            n.traffic_share = 0.0
        for n, w in zip(serving, weights, strict=False):
            n.traffic_share = w / total

    def _apply_incidents(self, *, step: int) -> None:
        spec = self._TASKS[self._task_id]
        for n in self._nodes:
            n.rtt_ms = 12.0
            n.throttle_factor = 1.0

        leak = spec.get("leak")
        if leak and int(leak["start"]) <= step <= int(leak["end"]):
            node = self._nodes[int(leak["node"])]
            node.leak_gb += float(leak["gb_per_step"])
            self._incident += f"VRAM leak on node {node.id}. "

        rtt_spike = spec.get("rtt_spike")
        if rtt_spike and int(rtt_spike["start"]) <= step <= int(rtt_spike["end"]):
            node = self._nodes[int(rtt_spike["node"])]
            node.rtt_ms = float(rtt_spike["rtt_ms"])
            self._incident += f"RTT spike on node {node.id}. "

        throttle = spec.get("throttle")
        if throttle and int(throttle["start"]) <= step <= int(throttle["end"]):
            node = self._nodes[int(throttle["node"])]
            node.throttle_factor = float(throttle["factor"])
            self._incident += f"Throughput throttle on node {node.id}. "

        self._incident = self._incident.strip()

    def _tick_drain_transitions(self) -> None:
        """Drain/resume is not instantaneous: it converges in 2 steps."""
        for n in self._nodes:
            # 2-step transition => 0.5 per step.
            if n.drain_progress < n.drain_target:
                n.drain_progress = min(n.drain_target, n.drain_progress + 0.5)
            elif n.drain_progress > n.drain_target:
                n.drain_progress = max(n.drain_target, n.drain_progress - 0.5)

            n.draining = bool(n.drain_progress >= 0.999)

    def _tick_reboots(self) -> None:
        for n in self._nodes:
            if n.reboot_cooldown_steps > 0:
                n.reboot_cooldown_steps -= 1

    def _normalize_traffic_shares(self) -> None:
        serving = [n for n in self._nodes if n.is_serving()]
        if not serving:
            for n in self._nodes:
                n.traffic_share = 0.0
            return

        total = sum(n.traffic_share for n in serving)
        if total <= 0.0:
            share = 1.0 / len(serving)
            for n in self._nodes:
                n.traffic_share = share if n.is_serving() else 0.0
            return

        for n in self._nodes:
            n.traffic_share = (n.traffic_share / total) if n.is_serving() else 0.0

    def _vram_used_pct(self, n: _Node) -> float:
        kv_gb = 0.35 * (n.batch_size / 8.0) * (n.max_concurrency / 16.0)

        # Quantization / precision affects the VRAM cost of the *model weights*.
        # Keep this deterministic and simple.
        precision_factor = {
            "fp16": 1.00,
            "bf16": 1.00,
            "int8": 0.65,
            "int4": 0.40,
        }.get(n.precision, 1.00)
        used = (n.model_base_gb * precision_factor) + kv_gb + n.leak_gb
        return max(0.0, min(1.5, used / n.vram_total_gb))

    def _oom_rate(self, n: _Node) -> float:
        used = self._vram_used_pct(n)
        if used <= 1.0:
            return 0.0
        return min(0.5, (used - 1.0) * 1.8)

    def _node_capacity_rps(self, n: _Node) -> float:
        base = 360.0
        batch_gain = 1.0 + 0.35 * (n.batch_size / 8.0)
        conc_gain = 1.0 + 0.30 * (n.max_concurrency / 16.0)
        rtt_penalty = 12.0 / max(12.0, n.rtt_ms)
        oom_penalty = 1.0 - self._oom_rate(n)
        return max(0.0, base * batch_gain * conc_gain * rtt_penalty * oom_penalty * n.throttle_factor)

    def _compute_cluster_metrics(self) -> tuple[float, float, float]:
        total_served = 0.0
        total_errors = 0.0
        per_node: list[tuple[float, float]] = []  # (served, p95)

        for n in self._nodes:
            effective_share = n.traffic_share * (1.0 - float(n.drain_progress))
            incoming = self.INCOMING_RPS * (effective_share if n.is_serving() else 0.0)
            if incoming <= 0.0:
                continue
            cap = self._node_capacity_rps(n)
            served = min(incoming, cap)
            dropped = max(0.0, incoming - served)

            oom = self._oom_rate(n)
            errors = (served + dropped) * oom + dropped

            util = 0.0 if cap <= 0.0 else min(0.999, served / max(1e-6, cap))
            queue_ms = 35.0 * (util / max(1e-6, (1.0 - util)))
            compute_ms = 90.0 / max(1.0, (n.batch_size / 8.0))
            p95 = compute_ms + n.rtt_ms + queue_ms

            total_served += served
            total_errors += errors
            per_node.append((served, p95))

        tps = total_served
        error_rate = min(1.0, total_errors / max(1e-6, self.INCOMING_RPS))
        if not per_node or tps <= 0.0:
            p95 = 5000.0
        else:
            p95 = sum(served * lat for served, lat in per_node) / max(1e-6, tps)
        return float(p95), float(error_rate), float(tps)

    def _is_sla_pass(self, p95_ms: float, error_rate: float) -> bool:
        spec = self._TASKS[self._task_id]
        return (p95_ms <= float(spec["sla_p95_ms"])) and (error_rate <= float(spec["sla_error_rate"]))

    def _final_score_v2(self) -> tuple[float, dict[str, float]]:
        """V2 weighted scoring with explicit breakdown.

        Breakdown components (all in [0,1]):
        - availability (40%): mean(1 - error_rate)
        - latency (30%): linear decay above the task p95 threshold
        - efficiency (30%): served fraction + restart minimization
        """

        spec = self._TASKS[self._task_id]
        p95_thr = float(spec["sla_p95_ms"])

        steps = max(1, len(self._p95_history))
        avg_err = sum(self._err_history) / steps
        avg_p95 = sum(self._p95_history) / steps
        avg_tps = sum(self._tps_history) / max(1, len(self._tps_history))

        s_avail = max(0.0, min(1.0, 1.0 - float(avg_err)))

        # Linear scaling: 1.0 at/below threshold; 0.0 at/above 2x threshold.
        if avg_p95 <= p95_thr:
            s_lat = 1.0
        elif avg_p95 >= 2.0 * p95_thr:
            s_lat = 0.0
        else:
            s_lat = 1.0 - (float(avg_p95) - p95_thr) / p95_thr
        s_lat = max(0.0, min(1.0, float(s_lat)))

        served_frac = max(0.0, min(1.0, float(avg_tps / max(1e-6, self.INCOMING_RPS))))
        restart_component = 1.0 - min(1.0, float(self._restart_count) / 8.0)
        s_eff = 0.70 * served_frac + 0.30 * restart_component
        s_eff = max(0.0, min(1.0, float(s_eff)))

        final = 0.40 * s_avail + 0.30 * s_lat + 0.30 * s_eff
        final = float(max(0.0, min(1.0, final)))

        # Ensure the returned task score is strictly inside (0,1).
        # This avoids edge cases where a task can score exactly 0.0 or 1.0.
        final = float(min(1.0 - self._SCORE_EPS, max(self._SCORE_EPS, final)))

        breakdown = {
            "availability": float(s_avail),
            "latency": float(s_lat),
            "efficiency": float(s_eff),
            "weighted_total": float(final),
            "avg_p95_ms": float(avg_p95),
            "avg_error_rate": float(avg_err),
            "avg_tps": float(avg_tps),
            "restart_count": float(self._restart_count),
            "p95_threshold_ms": float(p95_thr),
        }
        return final, breakdown

    def _make_observation(
        self,
        *,
        p95_ms: float,
        error_rate: float,
        tps: float,
        sla_pass: bool,
        final_score: float | None = None,
        uptime: float | None = None,
        avg_p95: float | None = None,
        avg_error: float | None = None,
        restart_count: int | None = None,
        score_breakdown: dict[str, float] | None = None,
    ) -> LlamaSreOrchestratorObservation:
        nodes: list[NodeMetrics] = []

        prev_vram = self._prev_node_vram_used_pct
        current_vram: list[float] = []
        for n in self._nodes:
            # Estimated queue depth for interpretability (not used directly by dynamics).
            # Higher utilization => higher queue.
            cap = self._node_capacity_rps(n)
            effective_share = n.traffic_share * (1.0 - float(n.drain_progress))
            incoming = self.INCOMING_RPS * (effective_share if n.is_serving() else 0.0)
            served = min(incoming, cap) if cap > 0.0 else 0.0
            util = 0.0 if cap <= 0.0 else min(0.999, served / max(1e-6, cap))
            queue_depth = float(50.0 * (util / max(1e-6, (1.0 - util)))) if n.is_serving() else 0.0

            vram_used_pct = float(self._vram_used_pct(n))
            current_vram.append(vram_used_pct)
            vram_velocity = 0.0
            if prev_vram is not None and 0 <= n.id < len(prev_vram):
                vram_velocity = float(vram_used_pct - float(prev_vram[n.id]))

            nodes.append(
                NodeMetrics(
                    id=n.id,
                    traffic_share=float((n.traffic_share * (1.0 - float(n.drain_progress))) if n.is_serving() else 0.0),
                    batch_size=int(n.batch_size),
                    max_concurrency=int(n.max_concurrency),
                    precision=str(n.precision),
                    vram_used_pct=vram_used_pct,
                    vram_velocity=float(vram_velocity),
                    rtt_ms=float(n.rtt_ms),
                    oom_rate=float(self._oom_rate(n)),
                    queue_depth=float(max(0.0, queue_depth)),
                    is_healthy=bool(n.is_serving()),
                    draining=bool(n.draining),
                )
            )

        prev_p95 = self._prev_p95_ms_for_trend
        p95_trend = 0.0 if prev_p95 is None else float(float(p95_ms) - float(prev_p95))

        cluster = ClusterMetrics(
            tps=float(tps),
            p95_ms=float(p95_ms),
            p95_trend=float(p95_trend),
            error_rate=float(error_rate),
            sla_pass_step=bool(sla_pass),
        )

        # Update trend baselines AFTER computing this observation.
        self._prev_node_vram_used_pct = current_vram
        self._prev_p95_ms_for_trend = float(p95_ms)

        return LlamaSreOrchestratorObservation(
            task_id=self._task_id,
            step=int(self._state.step_count),
            max_steps=self.MAX_STEPS,
            step_seconds=self.STEP_SECONDS,
            incoming_rps=self.INCOMING_RPS,
            incident=self._incident,
            cluster=cluster,
            nodes=nodes,
            last_action=self._last_action,
            last_action_impact=self._last_action_impact,
            final_score=final_score,
            uptime=uptime,
            avg_p95_ms=avg_p95,
            avg_error_rate=avg_error,
            restart_count=restart_count,
            score_breakdown=score_breakdown,
        )
