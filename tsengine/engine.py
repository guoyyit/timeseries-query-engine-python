"""Query execution engine.

Each query pins one immutable storage snapshot at start (requirement 6) and
cooperatively checks a deadline/cancellation token throughout evaluation
(requirement 7): on timeout or cancel it raises immediately and never
returns partial results.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

from . import ast_nodes as ast
from . import functions as funcs
from .errors import QueryCancelledError, QueryTimeoutError, QueryTypeError
from .parser import (
    _COMPARISON_OPS,
    _SET_OPS,
    parse,
)
from .storage import Snapshot, Storage
from .value import (
    Labels,
    Matrix,
    Sample,
    Scalar,
    Series,
    Value,
    Vector,
    labels_from_dict,
    labels_without,
)

_NAME_LABEL = "__name__"


@dataclass(slots=True)
class QueryContext:
    """Deadline + cancellation tracking shared by one query execution."""
    deadline: float | None = None
    cancel_event: threading.Event | None = None
    _cancelled: bool = False

    def remaining_ms(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, (self.deadline - time.perf_counter()) * 1000.0)

    def check_cancelled(self) -> None:
        """Raise QueryTimeoutError/QueryCancelledError; never return results."""
        if self._cancelled:
            raise QueryCancelledError()
        if self.cancel_event is not None and self.cancel_event.is_set():
            self._cancelled = True
            raise QueryCancelledError()
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self._cancelled = True
            raise QueryTimeoutError()

    def cancel(self) -> None:
        self._cancelled = True
        if self.cancel_event is not None:
            self.cancel_event.set()


_COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}
_ARITH = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "%": lambda a, b: math.fmod(a, b),
    "^": lambda a, b: math.pow(a, b),
}


class QueryEngine:
    def __init__(self, storage: Storage, *, default_timeout_ms: int = 30_000,
                 staleness_ms: int = 5 * 60 * 1000) -> None:
        self._storage = storage
        self._default_timeout_ms = default_timeout_ms
        self._staleness_ms = staleness_ms

    # -- public API --------------------------------------------------------
    def execute(self, expr_or_ast: str | ast.Node, *, at_ms: int | None = None,
                timeout_ms: int | None = None,
                ctx: QueryContext | None = None) -> Value:
        """Parse (if needed), validate and execute an instant query.

        Syntax/type errors are raised *before* any data access. Timeout or
        cancellation always raises and never returns partial results.
        """
        if at_ms is None:
            at_ms = int(time.time() * 1000)
        if isinstance(expr_or_ast, str):
            tree = parse(expr_or_ast)  # full validation first
        else:
            tree = expr_or_ast
        if ctx is None:
            timeout = (self._default_timeout_ms
                       if timeout_ms is None else timeout_ms)
            # Any non-positive timeout means the deadline has already
            # passed; None timeout explicitly disables the deadline.
            if timeout is None:
                deadline = None
            else:
                deadline = time.perf_counter() + timeout / 1000.0
            ctx = QueryContext(deadline=deadline,
                               cancel_event=threading.Event())
        # Pin ONE immutable snapshot for the whole query. Every node is
        # evaluated against this exact object, so concurrent writes cannot
        # leak different commit points into one result.
        snap = self._storage.current_snapshot()
        return self._eval(tree, snap, at_ms, ctx)

    # -- AST evaluation ----------------------------------------------------
    def _eval(self, node: ast.Node, snap: Snapshot, at_ms: int,
              ctx: QueryContext) -> Value:
        ctx.check_cancelled()
        if isinstance(node, ast.NumberLiteral):
            return Scalar(value=node.value, timestamp=at_ms)
        if isinstance(node, ast.StringLiteral):
            raise QueryTypeError("string literals are not query results")
        if isinstance(node, ast.ParenExpr):
            return self._eval(node.expr, snap, at_ms, ctx)
        if isinstance(node, ast.UnaryExpr):
            value = self._eval(node.expr, snap, at_ms, ctx)
            return self._apply_unary(node.op, value)
        if isinstance(node, ast.VectorSelector):
            return self._eval_selector(node, snap, at_ms, ctx)
        if isinstance(node, ast.MatrixSelector):
            return self._eval_matrix(node, snap, at_ms, ctx)
        if isinstance(node, ast.BinaryExpr):
            return self._eval_binary(node, snap, at_ms, ctx)
        if isinstance(node, ast.AggregateExpr):
            return self._eval_aggregate(node, snap, at_ms, ctx)
        if isinstance(node, ast.CallExpr):
            return self._eval_call(node, snap, at_ms, ctx)
        raise QueryTypeError(f"unsupported AST node {type(node).__name__}")

    @staticmethod
    def _apply_unary(op: str, value: Value) -> Value:
        if isinstance(value, Scalar):
            return Scalar(value=-value.value if op == "-" else value.value,
                          timestamp=value.timestamp)
        if isinstance(value, Vector):
            return Vector(
                Sample(labels=s.labels,
                       value=-s.value if op == "-" else s.value,
                       timestamp=s.timestamp)
                for s in value)
        raise QueryTypeError(f"unary {op} invalid on {type(value).__name__}")

    # -- selectors ---------------------------------------------------------
    def _eval_selector(self, node: ast.VectorSelector, snap: Snapshot,
                       at_ms: int, ctx: QueryContext) -> Vector:
        ctx.check_cancelled()
        eval_at = at_ms - node.offset_ms
        series = snap.select(node.name, node.matchers)
        samples: list[Sample] = []
        for sd in series:
            ctx.check_cancelled()
            point = self._latest_at(sd.points, eval_at)
            if point is None:
                continue  # stale / absent series contributes nothing
            ts, value = point
            labels = sd.labels
            # selector without explicit name matcher drops __name__ for
            # pure label-set selectors only; named selectors keep it (the
            # __name__ label is retained like any other label).
            samples.append(Sample(labels=labels, value=value, timestamp=ts))
        return Vector(samples)

    def _latest_at(self, points: tuple[tuple[int, float], ...],
                   eval_at: int) -> tuple[int, float] | None:
        """Return the newest sample at or before ``eval_at`` within the
        staleness window. Missing samples are never zero-filled.
        """
        # bisect_right over timestamps
        lo, hi = 0, len(points)
        while lo < hi:
            mid = (lo + hi) // 2
            if points[mid][0] <= eval_at:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        ts, value = points[lo - 1]
        if eval_at - ts > self._staleness_ms:
            return None
        return ts, value

    def _eval_matrix(self, node: ast.MatrixSelector, snap: Snapshot,
                     at_ms: int, ctx: QueryContext) -> Matrix:
        ctx.check_cancelled()
        inner = node.inner
        window_end = at_ms - inner.offset_ms
        window_start = window_end - node.range_ms  # exclusive lower bound
        series = snap.select(inner.name, inner.matchers)
        rows: list[Series] = []
        for sd in series:
            ctx.check_cancelled()
            pts = self._points_in_window(sd.points, window_start, window_end)
            if pts:
                # only emit series that actually have samples; empty rows
                # would otherwise masquerade as zero values
                rows.append(Series(labels=sd.labels, points=pts))
        return Matrix(rows)

    @staticmethod
    def _points_in_window(
            points: tuple[tuple[int, float], ...], start_exclusive: int,
            end_inclusive: int) -> tuple[tuple[int, float], ...]:
        """Half-open window (start, end]; points outside are excluded."""
        lo, hi = 0, len(points)
        while lo < hi:  # first index with ts > start_exclusive
            mid = (lo + hi) // 2
            if points[mid][0] <= start_exclusive:
                lo = mid + 1
            else:
                hi = mid
        left = lo
        lo, hi = 0, len(points)
        while lo < hi:  # first index with ts > end_inclusive
            mid = (lo + hi) // 2
            if points[mid][0] <= end_inclusive:
                lo = mid + 1
            else:
                hi = mid
        right = lo
        return points[left:right]

    # -- binary operations -------------------------------------------------
    def _eval_binary(self, node: ast.BinaryExpr, snap: Snapshot, at_ms: int,
                     ctx: QueryContext) -> Value:
        ctx.check_cancelled()
        lhs = self._eval(node.lhs, snap, at_ms, ctx)
        rhs = self._eval(node.rhs, snap, at_ms, ctx)
        op = node.op

        if isinstance(lhs, Scalar) and isinstance(rhs, Scalar):
            return self._scalar_scalar(op, lhs, rhs, at_ms, node.matching)
        if isinstance(lhs, Scalar) and isinstance(rhs, Vector):
            return self._vector_scalar(op, rhs, lhs.value, at_ms, ctx,
                                       scalar_on_left=False,
                                       matching=node.matching)
        if isinstance(lhs, Vector) and isinstance(rhs, Scalar):
            return self._vector_scalar(op, lhs, rhs.value, at_ms, ctx,
                                       scalar_on_left=True,
                                       matching=node.matching)
        # vector-vector
        return self._vector_vector(node, lhs, rhs, at_ms, ctx)

    def _scalar_scalar(self, op: str, lhs: Scalar, rhs: Scalar,
                       at_ms: int, matching: ast.VectorMatching) -> Scalar:
        if op in _SET_OPS:
            raise QueryTypeError(f"set op {op} invalid on scalars")
        a, b = lhs.value, rhs.value
        if op in _ARITH:
            if op in ("/", "%") and b == 0:
                # IEEE float semantics: divide/modulo by zero yields NaN
                return Scalar(value=math.nan, timestamp=at_ms)
            return Scalar(value=_ARITH[op](a, b), timestamp=at_ms)
        # comparison without bool: result defined as b if condition holds
        # else the query yields nothing - for scalars we encode "nothing"
        # as NaN with bool off (mirrors PromQL empty scalar result).
        cond = _COMPARISONS[op](a, b)
        if matching.bool_modifier:
            return Scalar(value=1.0 if cond else 0.0, timestamp=at_ms)
        return Scalar(value=a if cond else math.nan, timestamp=at_ms)

    def _vector_scalar(self, op: str, vec: Vector, scalar: float,
                       at_ms: int, ctx: QueryContext, *,
                       scalar_on_left: bool,
                       matching: ast.VectorMatching) -> Vector:
        if op in _SET_OPS:
            raise QueryTypeError(f"set op {op} invalid with scalar operand")
        out: list[Sample] = []
        for s in vec:
            ctx.check_cancelled()
            a, b = (scalar, s.value) if not scalar_on_left else (s.value,
                                                                 scalar)
            if op in _ARITH:
                if op in ("/", "%") and b == 0:
                    continue  # division by zero drops the sample
                val = _ARITH[op](a, b)
            else:
                cond = _COMPARISONS[op](a, b)
                if matching.bool_modifier:
                    val = 1.0 if cond else 0.0
                else:
                    if not cond:
                        continue  # comparison filters the series
                    val = s.value
            out.append(Sample(labels=s.labels, value=val, timestamp=at_ms))
        return Vector(out)

    def _match_key(self, labels: Labels,
                   matching: ast.VectorMatching) -> Labels:
        if matching.on_labels:
            return frozenset((k, v) for k, v in labels
                             if k in matching.on_labels)
        if matching.ignoring_labels:
            return frozenset((k, v) for k, v in labels
                             if k not in matching.ignoring_labels)
        return frozenset(labels)

    def _vector_vector(self, node: ast.BinaryExpr, lhs: Vector,
                       rhs: Vector, at_ms: int, ctx: QueryContext) -> Vector:
        op = node.op
        matching = node.matching
        right_index: dict[Labels, list[Sample]] = {}
        for s in rhs:
            ctx.check_cancelled()
            right_index.setdefault(self._match_key(s.labels, matching),
                                   []).append(s)
        left_index: dict[Labels, list[Sample]] = {}
        for s in lhs:
            ctx.check_cancelled()
            left_index.setdefault(self._match_key(s.labels, matching),
                                  []).append(s)

        if matching.grouping is None:
            # enforce one-to-one cardinality
            for lst in (*left_index.values(), *right_index.values()):
                if len(lst) > 1:
                    raise QueryTypeError(
                        "multiple series matched for the same labels; "
                        "use group_left/group_right to declare many-to-one")
            many_side = "left"
        else:
            many_side, _extra = matching.grouping
            one = right_index if many_side == "left" else left_index
            for lst in one.values():
                if len(lst) > 1:
                    raise QueryTypeError(
                        "found duplicate series on the 'one' side of "
                        f"group_{many_side} matching")

        out: list[Sample] = []

        if many_side == "right" and (op in _ARITH or op in _COMPARISON_OPS):
            # Drive from the right (many) side. The expression order is
            # always lhs OP rhs even though the right side is "many".
            for rs in rhs:
                ctx.check_cancelled()
                key = self._match_key(rs.labels, matching)
                lmatches = left_index.get(key, [])
                if not lmatches:
                    continue
                for ls in lmatches:
                    val, ok = self._combine(op, ls.value, rs.value, matching)
                    if not ok:
                        continue
                    out.append(Sample(
                        labels=self._output_labels(ls.labels, rs.labels,
                                                   matching),
                        value=val, timestamp=at_ms))
            return Vector(out)

        for ls in lhs:
            ctx.check_cancelled()
            key = self._match_key(ls.labels, matching)
            rights = right_index.get(key, [])
            if op in _SET_OPS:
                present = bool(rights)
                if (op == "and" and present) or (op == "or") \
                        or (op == "unless" and not present):
                    out.append(Sample(labels=ls.labels, value=ls.value,
                                      timestamp=at_ms))
                continue
            if not rights:
                continue
            for rs in rights:
                val, ok = self._combine(op, ls.value, rs.value, matching)
                if not ok:
                    continue
                out.append(Sample(
                    labels=self._output_labels(ls.labels, rs.labels,
                                               matching),
                    value=val, timestamp=at_ms))
        if op == "or":
            for rs in rhs:
                ctx.check_cancelled()
                key = self._match_key(rs.labels, matching)
                if key not in left_index:
                    out.append(Sample(labels=rs.labels, value=rs.value,
                                      timestamp=at_ms))
        return Vector(out)

    @staticmethod
    def _combine(op: str, a: float, b: float,
                 matching: ast.VectorMatching) -> tuple[float, bool]:
        """Apply one arithmetic/comparison op. Returns (value, emitted)."""
        if op in _ARITH:
            if op in ("/", "%") and b == 0:
                return 0.0, False
            return _ARITH[op](a, b), True
        cond = _COMPARISONS[op](a, b)
        if matching.bool_modifier:
            return (1.0 if cond else 0.0), True
        # without bool, comparison emits the LEFT value when true
        return (a if cond else 0.0), cond

    def _output_labels(self, left: Labels, right: Labels,
                       matching: ast.VectorMatching) -> Labels:
        if matching.grouping is None:
            # one-to-one: PromQL keeps the left-side labels
            return left
        group_side, _extra = matching.grouping
        if group_side == "left":
            # group_left: many side is the left; retain all its labels
            return left
        # group_right: many side is the right; merge left's join labels
        # into the right label set
        merged = dict(right)
        if matching.on_labels:
            for k, v in left:
                if k in matching.on_labels and k not in merged:
                    merged[k] = v
        else:
            ignore = matching.ignoring_labels
            for k, v in left:
                if k not in ignore and k not in merged:
                    merged[k] = v
        return labels_from_dict(merged)

    # -- aggregation -------------------------------------------------------
    def _eval_aggregate(self, node: ast.AggregateExpr, snap: Snapshot,
                        at_ms: int, ctx: QueryContext) -> Value:
        ctx.check_cancelled()
        vec = self._eval(node.expr, snap, at_ms, ctx)
        if not isinstance(vec, Vector):
            raise QueryTypeError(
                f"aggregation {node.op} requires an instant vector")

        if node.op in ("topk", "bottomk"):
            assert node.param is not None
            k_scalar = self._eval(node.param, snap, at_ms, ctx)
            k = int(k_scalar.value)  # type: ignore[union-attr]
            return self._agg_topk(node.op, k, vec, at_ms, ctx, node)
        if node.op == "count_values":
            assert isinstance(node.param, ast.StringLiteral)
            return self._agg_count_values(node.param.value, vec, at_ms, ctx,
                                          node)

        groups, group_labels = self._group_samples(vec, node, ctx)
        out: list[Sample] = []
        for key, samples in groups.items():
            ctx.check_cancelled()
            values = [s.value for s in samples]
            val = self._aggregate_scalar(node.op, values)
            if val is None:
                continue
            out.append(Sample(labels=group_labels[key], value=val,
                              timestamp=at_ms))
        return Vector(out)

    def _group_samples(self, vec: Vector, node: ast.AggregateExpr,
                       ctx: QueryContext):
        groups: dict[Labels, list[Sample]] = {}
        group_labels: dict[Labels, Labels] = {}
        group_set = frozenset(node.labels)
        for s in vec:
            ctx.check_cancelled()
            if node.by:
                key = frozenset((k, v) for k, v in s.labels
                                if k in group_set)
                out_labels = key
            else:  # without: group on everything not listed
                key = labels_without(s.labels, group_set)
                out_labels = key
            groups.setdefault(key, []).append(s)
            group_labels[key] = out_labels
        return groups, group_labels

    @staticmethod
    def _aggregate_scalar(op: str,
                          values: list[float]) -> float | None:
        if op == "group":
            return values[0]
        if not values and op != "count":
            return None
        n = len(values)
        if op == "sum":
            return math.fsum(values)
        if op == "avg":
            return math.fsum(values) / n if n else None
        if op == "count":
            return float(n)
        if op == "min":
            return min(values) if n else None
        if op == "max":
            return max(values) if n else None
        if op in ("stddev", "stdvar"):
            if n == 0:
                return None
            mean = math.fsum(values) / n
            variance = math.fsum((v - mean) ** 2 for v in values) / n
            return math.sqrt(variance) if op == "stddev" else variance
        raise QueryTypeError(f"unsupported aggregation {op}")

    def _agg_topk(self, op: str, k: int, vec: Vector, at_ms: int,
                  ctx: QueryContext, node: ast.AggregateExpr) -> Vector:
        if k <= 0:
            return Vector([])
        samples = list(vec)
        ctx.check_cancelled()
        samples.sort(key=lambda s: (-s.value if op == "topk" else s.value))
        chosen = samples[:k]
        # topk/bottomk do NOT collapse labels: original label sets survive
        return Vector(Sample(labels=s.labels, value=s.value,
                             timestamp=at_ms) for s in chosen)

    def _agg_count_values(self, new_label: str, vec: Vector, at_ms: int,
                          ctx: QueryContext,
                          node: ast.AggregateExpr) -> Vector:
        # group by (original grouping labels + value bucket label)
        counts: dict[Labels, float] = {}
        group_set = frozenset(node.labels)
        for s in vec:
            ctx.check_cancelled()
            base = (frozenset((k, v) for k, v in s.labels
                              if k in group_set)
                    if node.by else labels_without(s.labels, group_set))
            key = base | {(new_label, _format_number(s.value))}
            counts[key] = counts.get(key, 0.0) + 1.0
        return Vector(Sample(labels=k, value=v, timestamp=at_ms)
                      for k, v in counts.items())

    # -- function calls ----------------------------------------------------
    def _eval_call(self, node: ast.CallExpr, snap: Snapshot, at_ms: int,
                   ctx: QueryContext) -> Value:
        ctx.check_cancelled()
        name = node.func
        args = [self._eval(a, snap, at_ms, ctx) for a in node.args]
        if name in funcs.RANGE_FUNCTIONS:
            matrix = args[0]
            assert isinstance(matrix, Matrix)
            sel = node.args[0]
            range_ms = sel.range_ms if isinstance(sel, ast.MatrixSelector) \
                else 0
            return funcs.call_range(name, matrix, at_ms, range_ms)
        if name == "scalar":
            vec = args[0]
            assert isinstance(vec, Vector)
            samples = list(vec)
            # scalar() of one series -> its value; otherwise NaN
            if len(samples) == 1:
                return Scalar(value=samples[0].value, timestamp=at_ms)
            return Scalar(value=math.nan, timestamp=at_ms)
        if name in ("abs", "ceil", "floor", "sqrt", "sgn"):
            arg = args[0]
            if isinstance(arg, Scalar):
                return Scalar(value=funcs._SCALAR_FUNCS[name](arg.value),
                              timestamp=at_ms)
            return funcs.call_unary_numeric(name, arg, at_ms)
        if name == "round":
            return self._eval_round(args, at_ms)
        if name in ("sort", "sort_desc"):
            vec = args[0]
            if isinstance(vec, Scalar):
                return vec
            samples = sorted(vec, key=lambda s: s.value,
                             reverse=name == "sort_desc")
            return Vector(samples)
        raise QueryTypeError(f"unsupported function {name}")

    def _eval_round(self, args: list[Value], at_ms: int) -> Value:
        arg = args[0]
        nearest = 1.0
        if len(args) == 2 and isinstance(args[1], Scalar):
            nearest = args[1].value

        def round_to(v: float) -> float:
            if nearest == 0:
                return v
            return round(v / nearest) * nearest

        if isinstance(arg, Scalar):
            return Scalar(value=round_to(arg.value), timestamp=at_ms)
        return Vector(Sample(labels=s.labels, value=round_to(s.value),
                             timestamp=at_ms) for s in arg)  # type: ignore[union-attr]


def _format_number(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return repr(v)
