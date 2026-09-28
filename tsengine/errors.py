"""Error types for the time-series query engine.

Parse errors always carry a 1-based line/column position so callers can
locate the offending token. Execution-time errors (timeout / cancellation)
are distinct so callers can react differently to them.
"""

from __future__ import annotations


class TSEngineError(Exception):
    """Base class for all engine errors."""


class QuerySyntaxError(TSEngineError):
    """Raised for lexing/parsing/validation failures.

    Positions are 1-based and refer to character offsets in the source
    expression. ``line``/``column`` are convenient for multi-line input;
    ``offset`` is the absolute 0-based character offset.
    """

    def __init__(self, message: str, offset: int = -1,
                 line: int = 1, column: int = -1) -> None:
        self.message = message
        self.offset = offset
        self.line = line
        self.column = column
        location = ""
        if column > 0:
            location = f" (line {line}, col {column})"
        super().__init__(f"{message}{location}")


class QueryTypeError(TSEngineError):
    """Raised when an expression is well formed but has invalid value types."""


class QueryTimeoutError(TSEngineError):
    """Raised when a query exceeds its deadline.

    No partial results are ever returned alongside this error.
    """

    def __init__(self, message: str = "query timed out") -> None:
        super().__init__(message)


class QueryCancelledError(TSEngineError):
    """Raised when a query is cancelled before completion.

    No partial results are ever returned alongside this error.
    """

    def __init__(self, message: str = "query cancelled") -> None:
        super().__init__(message)


class StorageError(TSEngineError):
    """Raised for persistence / recovery failures."""
