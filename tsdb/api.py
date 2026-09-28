"""FastAPI HTTP surface.

Endpoints
---------
POST /write   {"metric": str, "labels": {...}, "samples": [{"t": int, "v": float}, ...]}
GET  /query?expr=...&at=<unix>&timeout=<seconds>
GET  /health

A query is parsed and validated before any data is touched, evaluated on a
single snapshot, and cancelled (no partial result, work stops) when either the
deadline expires or the HTTP client disconnects.
"""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .errors import (
    QueryCancelledError,
    QueryTimeoutError,
    StorageError,
    SyntaxErrorTS,
    TSDBError,
    ValidationErrorTS,
)
from .engine import CancellationToken, QueryEngine
from .model import Sample
from .storage import Storage


class SampleIn(BaseModel):
    t: int
    v: float


class WriteIn(BaseModel):
    metric: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)
    samples: list[SampleIn] = Field(min_length=1)


class BatchWriteIn(BaseModel):
    writes: list[WriteIn] = Field(min_length=1)


def _error_response(status: int, error_type: str, message: str, **extra) -> JSONResponse:
    body = {"error": error_type, "message": message}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)


def create_app(storage: Storage | None = None, wal_path: str | None = None) -> FastAPI:
    if storage is None:
        if wal_path is None:
            wal_path = os.environ.get("TSDB_WAL_PATH")
        storage = Storage(wal_path) if wal_path else Storage(None)
        if wal_path is not None:
            storage.recover()
    store = storage
    engine = QueryEngine(store)
    app = FastAPI(title="timeseries-query-engine")

    app.state.storage = store
    app.state.engine = engine

    @app.post("/write")
    async def write(payload: WriteIn) -> JSONResponse:
        try:
            samples = [Sample(s.t, s.v) for s in payload.samples]
            n = await asyncio.get_running_loop().run_in_executor(
                None, store.append, payload.metric, dict(payload.labels), samples
            )
        except StorageError as exc:
            return _error_response(400, "storage_error", str(exc))
        return JSONResponse(status_code=200, content={"written": n})

    @app.post("/write/batch")
    async def write_batch(payload: BatchWriteIn) -> JSONResponse:
        try:
            writes = [
                (
                    w.metric,
                    dict(w.labels),
                    [Sample(s.t, s.v) for s in w.samples],
                )
                for w in payload.writes
            ]
            n = await asyncio.get_running_loop().run_in_executor(
                None, store.append_batch, writes
            )
        except StorageError as exc:
            return _error_response(400, "storage_error", str(exc))
        return JSONResponse(status_code=200, content={"written": n})

    @app.get("/query")
    async def query(request: Request, expr: str, at: int, timeout: float = 5.0) -> JSONResponse:
        if timeout <= 0:
            deadline = time.monotonic()  # already expired -> immediate 504
        else:
            deadline = time.monotonic() + timeout
        token = CancellationToken(deadline)

        async def watch_disconnect() -> None:
            while not token.cancelled():
                if await request.is_disconnected():
                    token.cancel("client disconnected")
                    return
                await asyncio.sleep(0.01)

        watcher = asyncio.create_task(watch_disconnect())
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, engine.instant_query, expr, at, None, token
            )
        except SyntaxErrorTS as exc:
            return _error_response(
                400,
                "syntax_error",
                exc.message,
                pos=exc.pos,
                line=exc.line,
                col=exc.col,
            )
        except ValidationErrorTS as exc:
            return _error_response(400, "validation_error", str(exc))
        except QueryTimeoutError as exc:
            return _error_response(504, "timeout", str(exc))
        except QueryCancelledError as exc:
            return _error_response(499, "cancelled", str(exc))
        except TSDBError as exc:
            return _error_response(422, "query_error", str(exc))
        finally:
            watcher.cancel()
        return JSONResponse(status_code=200, content=result.to_plain())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
