"""Error types used across the query engine."""

from __future__ import annotations


class TSDBError(Exception):
    """Base class for all engine errors."""


class SyntaxErrorTS(TSDBError):
    """Syntax error carrying a 0-based byte position and line/column info."""

    def __init__(self, message: str, pos: int, line: int, col: int):
        self.message = message
        self.pos = pos
        self.line = line
        self.col = col
        super().__init__(f"syntax error at line {line}, col {col} (offset {pos}): {message}")


class ValidationErrorTS(TSDBError):
    """Semantic validation error for a parsed query."""


class QueryTimeoutError(TSDBError):
    """Raised when a query exceeds its deadline."""


class QueryCancelledError(TSDBError):
    """Raised when a query is cancelled before completion."""


class StorageError(TSDBError):
    """Raised for persistence / recovery failures."""
