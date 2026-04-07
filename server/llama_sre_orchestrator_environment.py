"""Repo-root OpenEnv environment entrypoint.

This wrapper exists to satisfy common submission validators that expect:
- `openenv.yaml` at repo root
- `server/app.py` at repo root

The actual environment implementation lives in
`llama_sre_orchestrator.server.llama_sre_orchestrator_environment`.
"""

from llama_sre_orchestrator.server.llama_sre_orchestrator_environment import (  # noqa: F401
    LlamaSreOrchestratorEnvironment,
)
