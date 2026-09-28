"""Service entry point: ``uvicorn main:app`` / ``uv run fastapi run``."""

from tsdb.api import app

__all__ = ["app"]
