FROM python:3.10-slim

WORKDIR /app

# System deps for healthcheck and clean installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy repo (so repo-root openenv.yaml + server/ work)
COPY . /app/

# Install runtime deps
RUN pip install --no-cache-dir -r llama_sre_orchestrator/server/requirements.txt

ENV PYTHONPATH="/app:/app/llama_sre_orchestrator"
ENV ENABLE_WEB_INTERFACE="true"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
