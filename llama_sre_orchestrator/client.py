# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama Sre Orchestrator Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LlamaSreOrchestratorAction, LlamaSreOrchestratorObservation


class LlamaSreOrchestratorEnv(
    EnvClient[LlamaSreOrchestratorAction, LlamaSreOrchestratorObservation, State]
):
    """
    Client for the Llama Sre Orchestrator Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with LlamaSreOrchestratorEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="vram_recovery_easy")
        ...     print(result.observation.task_id)
        ...
        ...     result = client.step(LlamaSreOrchestratorAction(kind="noop"))
        ...     print(result.observation.cluster.p95_ms)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = LlamaSreOrchestratorEnv.from_docker_image("llama_sre_orchestrator-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(LlamaSreOrchestratorAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: LlamaSreOrchestratorAction) -> Dict:
        """
        Convert LlamaSreOrchestratorAction to JSON payload for step message.

        Args:
            action: LlamaSreOrchestratorAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[LlamaSreOrchestratorObservation]:
        """
        Parse server response into StepResult[LlamaSreOrchestratorObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with LlamaSreOrchestratorObservation
        """
        obs_data = payload.get("observation", {})
        observation = LlamaSreOrchestratorObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
