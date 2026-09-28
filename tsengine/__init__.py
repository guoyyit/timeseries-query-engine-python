"""Local time-series metrics query engine.

A small PromQL-inspired engine providing expression parsing/validation,
label selectors, range-vector window functions, configurable vector
matching, aggregation/grouping, snapshot-consistent concurrent reads,
query timeout/cancellation and WAL-based restart recovery.
"""

from .engine import QueryContext, QueryEngine
from .errors import (
                     QueryCancelledError,
                     QuerySyntaxError,
                     QueryTimeoutError,
                     QueryTypeError,
                     StorageError,
                     TSEngineError,
)
from .parser import parse
from .storage import Snapshot, Storage
from .value import (
                     Labels,
                     Matrix,
                     Sample,
                     Scalar,
                     Series,
                     Vector,
                     labels_from_dict,
                     labels_to_dict,
)

__all__ = [
                     "Labels",
                     "Matrix",
                     "QueryCancelledError",
                     "QueryContext",
                     "QueryEngine",
                     "QuerySyntaxError",
                     "QueryTimeoutError",
                     "QueryTypeError",
                     "Sample",
                     "Scalar",
                     "Series",
                     "Snapshot",
                     "Storage",
                     "StorageError",
                     "TSEngineError",
                     "Vector",
                     "labels_from_dict",
                     "labels_to_dict",
                     "parse",
]
