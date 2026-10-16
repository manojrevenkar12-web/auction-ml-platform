# Serving image: implements the SageMaker inference container contract
# (GET /ping, POST /invocations on :8080, model unpacked at /opt/ml/model).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AUCML_MODEL_DIR=/opt/ml/model

RUN useradd --create-home --uid 1000 model
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[aws]"

USER model
EXPOSE 8080
# SageMaker starts the container with the argument "serve".
ENTRYPOINT ["sh", "-c", "exec uvicorn --factory aucml.serving.app:create_app --host 0.0.0.0 --port 8080 --workers ${WORKERS:-2}"]
