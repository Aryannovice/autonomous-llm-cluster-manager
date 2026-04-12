"""Manifest-addressable graders for hackathon validation.

The Phase 2 validator expects grader class paths in ``openenv.yaml``. These
classes proxy the deterministic per-task grading logic already used by the
environment runtime.
"""

from __future__ import annotations

from openenv.core.rubrics.base import Rubric

from llama_sre_orchestrator.server.llama_sre_orchestrator_environment import (
    LlamaSreOrchestratorEnvironment,
    _TaskGrader,
)


_TASK_SPECS = LlamaSreOrchestratorEnvironment._TASKS


class _ManifestTaskGrader(Rubric):
    """Thin wrapper that exposes an importable rubric class per task."""

    TASK_ID: str = ""

    def __init__(self) -> None:
        super().__init__()
        if self.TASK_ID not in _TASK_SPECS:
            raise ValueError(f"Unknown task grader task id: {self.TASK_ID}")
        spec = _TASK_SPECS[self.TASK_ID]
        self._grader = _TaskGrader(
            self.TASK_ID,
            spec["sla_p95_ms"],
            spec["sla_error_rate"],
        )

    def forward(self, action, observation) -> float:
        return float(self._grader.forward(action, observation))


class VramRecoveryGrader(_ManifestTaskGrader):
    TASK_ID = "vram_recovery_easy"


class NetworkSpikeGrader(_ManifestTaskGrader):
    TASK_ID = "network_spike_medium"


class MixedIncidentsGrader(_ManifestTaskGrader):
    TASK_ID = "mixed_incidents_hard"
