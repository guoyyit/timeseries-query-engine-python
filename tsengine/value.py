"""Runtime value types used by the query engine.

Four kinds of values are modelled after PromQL:

* Scalar  - a single float with an evaluation timestamp
* Vector  - a set of samples, each belonging to a distinct label set
* Matrix  - a set of series, each carrying a list of (timestamp, value) points
* String  - reserved for completeness (not produced by current grammar)

Labels are immutable ``frozenset[str, str]`` mappings so that label sets can
be used as dictionary keys for grouping and matching.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

Labels = frozenset[tuple[str, str]]


def labels_from_dict(d: Mapping[str, str]) -> Labels:
    """Build an immutable, canonicalised label set from a mapping."""
    return frozenset((str(k), str(v)) for k, v in d.items())


def labels_to_dict(labels: Labels) -> dict[str, str]:
    return dict(labels)


def labels_get(labels: Labels, name: str) -> str | None:
    for k, v in labels:
        if k == name:
            return v
    return None


def labels_without(labels: Labels, names: frozenset[str]) -> Labels:
    return frozenset((k, v) for k, v in labels if k not in names)


def labels_sorted(labels: Labels) -> list[tuple[str, str]]:
    return sorted(labels, key=lambda kv: kv[0])


@dataclass(frozen=True, slots=True)
class Sample:
    """One instant-vector entry: a label set evaluated at ``timestamp``."""
    labels: Labels
    value: float
    timestamp: int  # unix epoch milliseconds


@dataclass(frozen=True, slots=True)
class Scalar:
    value: float
    timestamp: int


@dataclass(frozen=True, slots=True)
class Series:
    """One matrix row: a label set plus its windowed sample points.

    Points are ordered by timestamp ascending.
    """
    labels: Labels
    points: tuple[tuple[int, float], ...] = field(default_factory=tuple)


class Vector:
    """Instant vector.

    Logically an instant vector maps each label set to at most one sample
    *for a given evaluation timestamp*. The engine guarantees that invariant
    on final results, but intermediate collections (e.g. samples fed into an
    aggregation) may legitimately contain duplicate label sets, so this
    container stores samples in order without silently deduplicating.
    """

    __slots__ = ("_samples",)

    def __init__(self, samples: Iterator[Sample] | None = None) -> None:
        self._samples: list[Sample] = list(samples or ())

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return bool(self._samples)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self._samples == other._samples

    def get(self, labels: Labels) -> Sample | None:
        for s in self._samples:
            if s.labels == labels:
                return s
        return None

    def labels_set(self) -> frozenset[Labels]:
        return frozenset(s.labels for s in self._samples)


class Matrix:
    """Range vector: ordered rows, each a label set plus windowed points."""

    __slots__ = ("_series",)

    def __init__(self, series: Iterator[Series] | None = None) -> None:
        self._series: list[Series] = list(series or ())

    def __iter__(self) -> Iterator[Series]:
        return iter(self._series)

    def __len__(self) -> int:
        return len(self._series)

    def __bool__(self) -> bool:
        return bool(self._series)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._series == other._series


Value = Scalar | Vector | Matrix
