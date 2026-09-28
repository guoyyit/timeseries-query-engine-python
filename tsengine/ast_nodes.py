"""AST node definitions produced by the parser."""

from __future__ import annotations

from dataclasses import dataclass, field

from .matcher import LabelMatcher


@dataclass(frozen=True, slots=True)
class Node:
    # 0-based char offset of the first token of this node; -1 if synthesized
    position: int = -1


@dataclass(frozen=True, slots=True)
class NumberLiteral(Node):
    value: float = 0.0


@dataclass(frozen=True, slots=True)
class StringLiteral(Node):
    value: str = ""


@dataclass(frozen=True, slots=True)
class VectorSelector(Node):
    """Instant-vector selector, e.g. ``http_requests_total{job="api"}``."""
    name: str | None = None
    matchers: tuple[LabelMatcher, ...] = ()
    offset_ms: int = 0


@dataclass(frozen=True, slots=True)
class MatrixSelector(Node):
    """Range-vector selector, e.g. ``metric{...}[5m]``."""
    inner: VectorSelector = field(default_factory=lambda: VectorSelector())
    range_ms: int = 0


@dataclass(frozen=True, slots=True)
class VectorMatching:
    """Configurable alignment rules for binary vector operations.

    * ``on`` / ``ignoring`` are mutually exclusive label lists.
    * ``grouping`` is None for one-to-one, ``("left", labels)`` for many-to-one
      with group_left, or ``("right", labels)`` for group_right.
    """
    on_labels: frozenset[str] = frozenset()
    ignoring_labels: frozenset[str] = frozenset()
    grouping: tuple[str, tuple[str, ...]] | None = None  # (side, extra labels)
    bool_modifier: bool = False


@dataclass(frozen=True, slots=True)
class BinaryExpr(Node):
    op: str = ""
    lhs: Node = field(default_factory=lambda: NumberLiteral())
    rhs: Node = field(default_factory=lambda: NumberLiteral())
    matching: VectorMatching = field(default_factory=VectorMatching)


@dataclass(frozen=True, slots=True)
class AggregateExpr(Node):
    op: str = ""
    expr: Node = field(default_factory=lambda: NumberLiteral())
    param: Node | None = None
    by: bool = True  # True => grouping by ``labels``; False => without
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CallExpr(Node):
    func: str = ""
    args: tuple[Node, ...] = ()


@dataclass(frozen=True, slots=True)
class ParenExpr(Node):
    expr: Node = field(default_factory=lambda: NumberLiteral())


@dataclass(frozen=True, slots=True)
class UnaryExpr(Node):
    op: str = "-"
    expr: Node = field(default_factory=lambda: NumberLiteral())
