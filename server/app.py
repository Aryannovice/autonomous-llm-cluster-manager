"""Repo-root FastAPI application for OpenEnv.

This wrapper exists to satisfy common submission validators that expect the
FastAPI entrypoint at `server/app.py` in the repo root.

The environment logic lives in:
- `llama_sre_orchestrator.server.llama_sre_orchestrator_environment`
- `llama_sre_orchestrator.models`
"""

import os
import json
import time
from pathlib import Path

from openenv.core.env_server.http_server import create_app
from starlette.responses import RedirectResponse

from llama_sre_orchestrator.models import (
    LlamaSreOrchestratorAction,
    LlamaSreOrchestratorObservation,
)
from server.llama_sre_orchestrator_environment import LlamaSreOrchestratorEnvironment

# region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug-f39562.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "f39562",
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


_debug_log(
    "H6",
    "server/app.py:module",
    "server app module loaded",
    {"cwd": os.getcwd(), "debug_log_path": str(_DEBUG_LOG_PATH)},
)
# endregion

app = create_app(
    LlamaSreOrchestratorEnvironment,
    LlamaSreOrchestratorAction,
    LlamaSreOrchestratorObservation,
    env_name="llama_sre_orchestrator",
    max_concurrent_envs=1,
)


@app.get("/", response_model=None, include_in_schema=False)
def root():
    # Spaces and casual users hit the root URL first.
    # Redirect to /web when enabled; otherwise show a tiny hint.
    if str(os.getenv("ENABLE_WEB_INTERFACE", "")).lower() in {"1", "true", "yes", "on"}:
        return RedirectResponse(url="/web")
    return {"status": "ok", "hint": "Open /health or set ENABLE_WEB_INTERFACE=true and open /web"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
