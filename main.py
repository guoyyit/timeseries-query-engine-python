"""HTTP facade for the local time-series query engine.

Endpoints
---------
GET  /health        liveness probe
POST /api/v1/write  append one or more samples (fsynced)
GET  /api/v1/query  instant query: ?expr=&at=&timeout=

Invalid expressions return HTTP 400 with a character position; semantic
type errors return 422; timeouts return 504 and cancellation 499. None of
the error responses carries a partial result.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from tsengine import (
    QueryCancelledError,
    QueryEngine,
    QuerySyntaxError,
    QueryTimeoutError,
    QueryTypeError,
    Storage,
)
from tsengine.value import Matrix, Scalar, Vector

DATA_DIR = os.environ.get("TSENGINE_DATA_DIR", "./data")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = Storage(DATA_DIR)
    app.state.storage = store
    app.state.engine = QueryEngine(store)
    try:
        yield
    finally:
        store.close()


app = FastAPI(title="Local Time-Series Query Engine", version="0.1.0",
              lifespan=lifespan)


class SampleIn(BaseModel):
    labels: dict[str, str] = Field(
        ..., description="labels, must include __name__")
    timestamp_ms: int
    value: float


class WriteRequest(BaseModel):
    samples: list[SampleIn]


def _json_number(v: float):
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    return v


def _serialize(value: Any) -> dict[str, Any]:
    if isinstance(value, Scalar):
        return {"kind": "scalar", "timestamp_ms": value.timestamp,
                "value": _json_number(value.value)}
    if isinstance(value, Vector):
        return {"kind": "vector", "samples": [
            {"labels": dict(s.labels),
             "value": _json_number(s.value),
             "timestamp_ms": s.timestamp} for s in value]}
    if isinstance(value, Matrix):
        return {"kind": "matrix", "series": [
            {"labels": dict(s.labels),
             "points": [[ts, _json_number(v)] for ts, v in s.points]}
            for s in value]}
    raise QueryTypeError(f"cannot serialize {type(value).__name__}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": str(app.state.storage.version)}


@app.post("/api/v1/write")
def write(req: WriteRequest) -> dict[str, Any]:
    store: Storage = app.state.storage
    batch = [(s.labels, s.timestamp_ms, s.value) for s in req.samples]
    store.append_batch(batch)
    return {"status": "ok", "accepted": len(batch),
            "version": store.version}


@app.get("/api/v1/query")
def query(expr: str = Query(..., description="query expression"),
          at_ms: int | None = Query(None, alias="at"),
          timeout_ms: int | None = Query(None, alias="timeout")):
    eng: QueryEngine = app.state.engine
    if at_ms is None:
        at_ms = int(time.time() * 1000)
    try:
        result = eng.execute(expr, at_ms=at_ms, timeout_ms=timeout_ms)
    except QuerySyntaxError as exc:
        return JSONResponse(status_code=400, content={
            "status": "error", "error_type": "syntax",
            "error": exc.message, "offset": exc.offset,
            "line": exc.line, "column": exc.column})
    except QueryTypeError as exc:
        return JSONResponse(status_code=422, content={
            "status": "error", "error_type": "type", "error": str(exc)})
    except QueryTimeoutError as exc:
        return JSONResponse(status_code=504, content={
            "status": "error", "error_type": "timeout", "error": str(exc)})
    except QueryCancelledError as exc:
        return JSONResponse(status_code=499, content={
            "status": "error", "error_type": "cancelled",
            "error": str(exc)})
    return {"status": "ok", "result": _serialize(result)}
