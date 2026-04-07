FROM python:3.12-slim

WORKDIR /app

# System deps for healthcheck and clean installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy repo (so repo-root openenv.yaml + server/ work)
COPY . /app/

# Install runtime deps
RUN pip install --no-cache-dir \
  "openenv-core[core]==0.2.3" \
  "fastapi>=0.115.0" \
  "uvicorn>=0.24.0"

ENV PYTHONPATH="/app:/app/llama_sre_orchestrator"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
