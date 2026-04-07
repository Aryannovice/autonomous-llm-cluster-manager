"""Repo-root FastAPI application for OpenEnv.

This wrapper exists to satisfy common submission validators that expect the
FastAPI entrypoint at `server/app.py` in the repo root.

The environment logic lives in:
- `llama_sre_orchestrator.server.llama_sre_orchestrator_environment`
- `llama_sre_orchestrator.models`
"""

from openenv.core.env_server.http_server import create_app

from llama_sre_orchestrator.models import (
    LlamaSreOrchestratorAction,
    LlamaSreOrchestratorObservation,
)
from server.llama_sre_orchestrator_environment import LlamaSreOrchestratorEnvironment

app = create_app(
    LlamaSreOrchestratorEnvironment,
    LlamaSreOrchestratorAction,
    LlamaSreOrchestratorObservation,
    env_name="llama_sre_orchestrator",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
