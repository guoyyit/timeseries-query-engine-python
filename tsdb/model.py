"""Core data model: labels, samples, vectors, matrices, matching config."""

from __future__ import annotations

import enum
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Label:
    name: str
    value: str


# A label set maps label name -> label value. "__name__" carries the metric name.
LabelSet = dict[str, str]
NAME_LABEL = "__name__"


class MatchType(enum.Enum):
    EQ = "="
    NEQ = "!="
    RE = "=~"
    NRE = "!~"


@dataclass(frozen=True, slots=True)
class LabelMatcher:
    name: str
    op: MatchType
    value: str


@dataclass(frozen=True, slots=True, order=True)
class Sample:
    """A single timestamp/value pair. Timestamps are unix seconds (int)."""

    t: int
    v: float


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Window [start, end]; both ends are inclusive."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"invalid range: start {self.start} > end {self.end}")


# ---- Execution-time result primitives ---------------------------------------

Scalar = float


@dataclass(frozen=True, slots=True)
class VectorItem:
    """One instant-vector element: labels + value at evaluation instant."""

    labels: LabelSet
    value: float


@dataclass(frozen=True, slots=True)
class MatrixItem:
    """One range-vector element: labels + samples within the window."""

    labels: LabelSet
    values: tuple[Sample, ...]


Vector = tuple[VectorItem, ...]
Matrix = tuple[MatrixItem, ...]


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Result of an instant query at timestamp ``at``.

    ``kind`` is one of: "scalar", "vector", "matrix".
    """

    kind: str
    at: int
    scalar: float | None = None
    vector: Vector = ()
    matrix: Matrix = ()

    def to_plain(self) -> object:
        if self.kind == "scalar":
            return {"kind": "scalar", "at": self.at, "value": self.scalar}
        if self.kind == "vector":
            return {
                "kind": "vector",
                "at": self.at,
                "series": [
                    {"labels": dict(it.labels), "value": it.value} for it in self.vector
                ],
            }
        return {
            "kind": "matrix",
            "at": self.at,
            "series": [
                {
                    "labels": dict(it.labels),
                    "values": [{"t": s.t, "v": s.v} for s in it.values],
                }
                for it in self.matrix
            ],
        }


# ---- Binary-expression vector matching configuration ------------------------


@dataclass(frozen=True, slots=True)
class VectorMatching:
    """Alignment rules for binary operations.

    - ``on``: if non-empty, join only on these labels; otherwise match on the
      full label set.
    - ``ignoring``: labels to drop from both sides before matching.
    - ``group_side``: None = one-to-one; "left" = group_left (the LEFT operand
      is the "many" side, as in Prometheus); "right" = group_right (the RIGHT
      operand is the "many" side).
    - ``include``: labels copied from the "one" side into each result.
    """

    on: tuple[str, ...] = ()
    ignoring: tuple[str, ...] = ()
    group_side: str | None = None
    include: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Write:
    metric: str
    labels: LabelSet
    samples: tuple[Sample, ...]


def labels_key(
    labels: LabelSet,
    on: tuple[str, ...] = (),
    ignoring: tuple[str, ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Deterministic identity key for a label set under a matching projection."""
    skip = set(ignoring)
    if on:
        on_set = set(on)
        items = ((k, v) for k, v in labels.items() if k in on_set and k not in skip)
    else:
               items = ((k, v) for k, v in labels.items() if k not in skip)
    return tuple(sorted(items))


def output_labels(
    left: LabelSet,
    right: LabelSet,
    matching: VectorMatching,
) -> LabelSet:
    """Compute the output label set of a vector-vector binary operation.

    Fixed, explainable rules:

    * one-to-one without ``on``: output = left labels minus ``ignoring``;
      labels existing only on the right are dropped;
    * one-to-one with ``on``: output = the join labels union labels unique to
      the left side (not present on the right). Labels present on both sides
      but not part of the join set are dropped as ambiguous;
    * many-to-one (``group_left``/``group_right``): output = all of the
      "many" side's labels under the same projection plus each ``include``
      label copied from the "one" side.
    """
    ignoring = set(matching.ignoring)
    on = set(matching.on)

    if matching.group_side is None:
        many, one = left, right
        copy_include: tuple[str, ...] = ()
    elif matching.group_side == "left":
        many, one = left, right
        copy_include = matching.include
    elif matching.group_side == "right":
        many, one = right, left
        copy_include = matching.include
    else:  # pragma: no cover - validated by parser
        raise ValueError(f"invalid group side: {matching.group_side}")

    out: LabelSet = {}
    is_group = matching.group_side is not None
    if on:
        for k, v in many.items():
            if k in ignoring:
                continue
            if k in on:
                # Join labels are part of the result; equality holds because
                # the pair was matched on this label.
                if k in one and one[k] == v:
                    out[k] = v
            elif is_group:
                # many-to-one: every "many" side label is preserved.
                out[k] = v
            elif k not in one:
                # one-to-one on-join: preserve only labels unique to the left
                # side. A label present on both sides but absent from the
                # "on" list is ambiguous (whose value wins?) and is dropped.
                out[k] = v
    else:
        for k, v in many.items():
            if k not in ignoring:
                out[k] = v

    for k in copy_include:
        if k in one and k not in ignoring:
            out[k] = one[k]
    return out
