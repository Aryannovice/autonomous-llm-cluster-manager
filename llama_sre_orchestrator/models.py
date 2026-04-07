# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Llama Sre Orchestrator environment.

This environment simulates an autonomous SRE managing a 3-node distributed GPU
cluster that serves LLM inference traffic.

The agent acts via structured JSON actions (modeled as a typed OpenEnv Action),
and receives a metrics-rich Observation suitable for deterministic grading.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


TaskId = Literal[
    "vram_recovery_easy",
    "network_spike_medium",
    "mixed_incidents_hard",
]


ActionKind = Literal[
    "noop",
    "set_node_params",
    "drain_node",
    "resume_node",
    "restart_node",
    "rebalance",
]


RebalanceStrategy = Literal["even", "least_rtt", "least_vram", "min_oom"]


class NodeMetrics(BaseModel):
    id: int = Field(..., ge=0, le=2, description="Node id (0..2)")
    traffic_share: float = Field(..., ge=0.0, le=1.0, description="Routed traffic fraction")
    batch_size: int = Field(..., description="Configured batch size")
    max_concurrency: int = Field(..., description="Configured max concurrency")
    vram_used_pct: float = Field(..., ge=0.0, description="VRAM used / VRAM total")
    rtt_ms: float = Field(..., ge=0.0, description="Network round-trip time in ms")
    oom_rate: float = Field(..., ge=0.0, le=1.0, description="OOM-induced failure rate")
    queue_depth: float = Field(..., ge=0.0, description="Estimated queue depth (arbitrary units)")
    is_healthy: bool = Field(..., description="Whether the node is serving")
    draining: bool = Field(..., description="Whether the node is drained (no traffic)")


class ClusterMetrics(BaseModel):
    tps: float = Field(..., ge=0.0, description="Cluster throughput (requests/sec)")
    p95_ms: float = Field(..., ge=0.0, description="Estimated cluster p95 latency (ms)")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Cluster error rate")
    sla_pass_step: bool = Field(..., description="Whether SLA passed at this step")


class LlamaSreOrchestratorAction(Action):
    """Structured JSON action for managing the simulated cluster."""

    kind: ActionKind = Field(..., description="Action type")

    node: Optional[int] = Field(
        default=None,
        ge=0,
        le=2,
        description="Target node for node-scoped actions (0..2)",
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size (allowed: 1,2,4,8,16) for set_node_params",
    )
    max_concurrency: Optional[int] = Field(
        default=None,
        description="Max concurrency (allowed: 1,2,4,8,16,32) for set_node_params",
    )
    strategy: Optional[RebalanceStrategy] = Field(
        default=None,
        description="Rebalance strategy for rebalance action",
    )


class LlamaSreOrchestratorObservation(Observation):
    """Metrics-rich observation for the SRE cluster simulation."""

    task_id: TaskId = Field(..., description="Which task/scenario is active")
    step: int = Field(..., ge=0, le=60, description="Step counter (0..60)")
    max_steps: int = Field(default=60, ge=1, description="Episode horizon")
    step_seconds: int = Field(default=10, ge=1, description="Seconds simulated per step")
    incoming_rps: float = Field(..., ge=0.0, description="Incoming request rate")

    incident: str = Field(default="", description="Human-readable incident summary")
    cluster: ClusterMetrics = Field(..., description="Cluster-level metrics")
    nodes: list[NodeMetrics] = Field(..., min_length=3, max_length=3)

    # Debug/interpretability (kept in observation, not metadata, so it survives serialization)
    last_action: Optional[dict[str, Any]] = Field(
        default=None, description="Most recent action payload applied by the env"
    )
    last_action_impact: Optional[dict[str, Any]] = Field(
        default=None,
        description="Approximate impact deltas for the most recent step (deterministic)",
    )

    # Episode-level values populated when done=True
    final_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Final deterministic score (0..1)"
    )
    uptime: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Mean SLA pass rate over the episode"
    )
    avg_p95_ms: Optional[float] = Field(default=None, ge=0.0)
    avg_error_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    restart_count: Optional[int] = Field(default=None, ge=0)

    # V2: Weighted breakdown shown at episode end.
    score_breakdown: Optional[dict[str, float]] = Field(
        default=None,
        description="Episode-end score components (availability/latency/efficiency) in [0,1]",
    )
