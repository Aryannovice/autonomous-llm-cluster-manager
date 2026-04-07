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
from typing import Any, Final, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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

    batch_size: int = 8
    max_concurrency: int = 16
    traffic_share: float = 1.0 / 3.0

    draining: bool = False
    reboot_cooldown_steps: int = 0

    leak_gb: float = 0.0
    rtt_ms: float = 12.0
    throttle_factor: float = 1.0

    def is_serving(self) -> bool:
        return (not self.draining) and self.reboot_cooldown_steps <= 0


class LlamaSreOrchestratorEnvironment(Environment[LlamaSreOrchestratorAction, LlamaSreOrchestratorObservation, State]):
    """3-node GPU cluster SRE simulator."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS: Final[int] = 60
    STEP_SECONDS: Final[int] = 10
    INCOMING_RPS: Final[float] = 900.0

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

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: TaskId = "vram_recovery_easy"
        self._nodes: list[_Node] = []
        self._p95_history: list[float] = []
        self._err_history: list[float] = []
        self._sla_history: list[bool] = []
        self._restart_count: int = 0
        self._incident: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LlamaSreOrchestratorObservation:
        del seed, kwargs

        if task_id is None:
            task_id = "vram_recovery_easy"
        if task_id not in self._TASKS:
            raise ValueError(
                f"Unknown task_id={task_id!r}. Expected one of: {sorted(self._TASKS.keys())}"
            )
        self._task_id = task_id  # type: ignore[assignment]

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._nodes = [_Node(id=0), _Node(id=1), _Node(id=2)]
        for n in self._nodes:
            n.batch_size = 8
            n.max_concurrency = 16
            n.traffic_share = 1.0 / 3.0
            n.draining = False
            n.reboot_cooldown_steps = 0
            n.leak_gb = 0.0
            n.rtt_ms = 12.0
            n.throttle_factor = 1.0

        self._p95_history = []
        self._err_history = []
        self._sla_history = []
        self._restart_count = 0
        self._incident = ""

        return self._make_observation(p95_ms=0.0, error_rate=0.0, tps=0.0, sla_pass=True)

    def step(
        self,
        action: LlamaSreOrchestratorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LlamaSreOrchestratorObservation:
        del timeout_s, kwargs

        self._state.step_count += 1
        self._incident = ""

        self._apply_action(action)
        self._apply_incidents(step=self._state.step_count)
        self._tick_reboots()
        self._normalize_traffic_shares()

        p95_ms, error_rate, tps = self._compute_cluster_metrics()
        sla_pass = self._is_sla_pass(p95_ms, error_rate)

        self._p95_history.append(p95_ms)
        self._err_history.append(error_rate)
        self._sla_history.append(sla_pass)

        done = self._state.step_count >= self.MAX_STEPS
        reward = 0.02 if sla_pass else 0.0
        final_score = None
        uptime = None
        avg_p95 = None
        avg_err = None
        restart_count = None

        if done:
            final_score = self._final_score()
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
        )
        obs.done = done
        obs.reward = reward
        return obs

    @property
    def state(self) -> State:
        return self._state

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
            return

        if kind == "drain_node":
            if action.node is None:
                return
            self._nodes[int(action.node)].draining = True
            return

        if kind == "resume_node":
            if action.node is None:
                return
            self._nodes[int(action.node)].draining = False
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
        used = n.model_base_gb + kv_gb + n.leak_gb
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
            incoming = self.INCOMING_RPS * (n.traffic_share if n.is_serving() else 0.0)
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

    def _final_score(self) -> float:
        spec = self._TASKS[self._task_id]
        p95_thr = float(spec["sla_p95_ms"])
        err_thr = float(spec["sla_error_rate"])

        uptime = sum(1 for x in self._sla_history if x) / max(1, len(self._sla_history))
        avg_p95 = sum(self._p95_history) / max(1, len(self._p95_history))
        avg_err = sum(self._err_history) / max(1, len(self._err_history))

        lat_pen = max(0.0, (avg_p95 - p95_thr) / p95_thr)
        err_pen = max(0.0, (avg_err - err_thr) / max(1e-9, err_thr))
        restart_pen = min(0.25, 0.04 * self._restart_count)

        score = 1.35 * uptime - 0.30 * lat_pen - 0.35 * err_pen - restart_pen
        return float(max(0.0, min(1.0, score)))

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
    ) -> LlamaSreOrchestratorObservation:
        nodes: list[NodeMetrics] = []
        for n in self._nodes:
            nodes.append(
                NodeMetrics(
                    id=n.id,
                    traffic_share=float(n.traffic_share if n.is_serving() else 0.0),
                    batch_size=int(n.batch_size),
                    max_concurrency=int(n.max_concurrency),
                    vram_used_pct=float(self._vram_used_pct(n)),
                    rtt_ms=float(n.rtt_ms),
                    oom_rate=float(self._oom_rate(n)),
                    is_healthy=bool(n.is_serving()),
                    draining=bool(n.draining),
                )
            )

        cluster = ClusterMetrics(
            tps=float(tps),
            p95_ms=float(p95_ms),
            error_rate=float(error_rate),
            sla_pass_step=bool(sla_pass),
        )

        return LlamaSreOrchestratorObservation(
            task_id=self._task_id,
            step=int(self._state.step_count),
            max_steps=self.MAX_STEPS,
            step_seconds=self.STEP_SECONDS,
            incoming_rps=self.INCOMING_RPS,
            incident=self._incident,
            cluster=cluster,
            nodes=nodes,
            final_score=final_score,
            uptime=uptime,
            avg_p95_ms=avg_p95,
            avg_error_rate=avg_error,
            restart_count=restart_count,
        )
