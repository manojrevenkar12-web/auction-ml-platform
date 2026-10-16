"""Local serving container implementing the SageMaker HTTP contract (/ping, /invocations)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request, Response

from aucml.serving.inference import Handler, RequestError, handle, model_fn


def create_app(model_dir: Path | None = None, handler: Handler | None = None) -> FastAPI:
    app = FastAPI(title="aucml model server")
    state: dict[str, Handler] = {}

    def get() -> Handler:
        if "h" not in state:
            state["h"] = handler or model_fn(model_dir or Path(os.environ.get("AUCML_MODEL_DIR", "/opt/ml/model")))
        return state["h"]

    @app.get("/ping")
    def ping() -> Response:
        try:
            h = get()
        except Exception as exc:  # health check must not raise
            return Response(content=f"unhealthy: {exc}", status_code=503)
        return Response(content=h.version, status_code=200)

    @app.post("/invocations")
    async def invocations(request: Request) -> Response:
        body = await request.body()
        try:
            out, ctype = handle(body, get(), request.headers.get("content-type", "application/json"))
        except RequestError as exc:
            return Response(content=str(exc), status_code=400)
        return Response(content=out, media_type=ctype)

    return app
