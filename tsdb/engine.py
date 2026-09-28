"""Query engine: parse + validate, then evaluate an AST on one snapshot.

Result kinds: ``"scalar"`` (float), ``"vector"`` (tuple[VectorItem]),
``"matrix"`` (tuple[MatrixItem]). Evaluation is single-pass over an immutable
:class:`~tsdb.storage.Snapshot`; every loop boundary checks the
:class:`CancellationToken`, so timeout/cancel stops work immediately and no
partial result is returned to the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .errors import QueryCancelledError, QueryTimeoutError, TSDBError
from .functions import AGG_FUNCTIONS, RANGE_FUNCTIONS
from .model import (
    Matrix,
    MatrixItem,
    QueryResult,
    Vector,
    VectorItem,
    VectorMatching,
    labels_key,
    output_labels,
)
from . import parser as ast_mod
from .parser import (
    Aggregation,
    BinaryExpr,
    Call,
    MatrixSelector,
    NumberLiteral,
    ParenExpr,
    StringLiteral,
    UnaryExpr,
    VectorSelector,
)
from .storage import Storage

_INF = float("inf")
_SCALAR = "scalar"
_VECTOR = "vector"
_MATRIX = "matrix"


class CancellationToken:
    """Cooperative cancellation + deadline support."""

    __slots__ = ("_event", "_deadline", "_reason")

    def __init__(self, deadline: float | None = None) -> None:
        import threading

        self._event = threading.Event()
        self._deadline = deadline
        self._reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self._reason = reason
        self._event.set()

    def cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return True
        return False

    def check(self) -> None:
        if self._event.is_set():
            raise QueryCancelledError(self._reason or "query cancelled")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise QueryTimeoutError("query timed out")


@dataclass(slots=True)
class Evaluator:
    snapshot: object  # tsdb.storage.Snapshot
    at: int
    token: CancellationToken
    lookup_delta: int = 300

    # ---- entry -----------------------------------------------------------
    def eval(self, node: ast_mod.Expr) -> tuple[str, object]:
        self.token.check()
        if isinstance(node, NumberLiteral):
            return _SCALAR, node.value
        if isinstance(node, StringLiteral):
            # Strings only appear inside selector matchers; top-level strings
            # are rejected by parser validation. Defensive guard:
            raise TSDBError("strings are not queryable expressions")
        if isinstance(node, VectorSelector):
            return _VECTOR, self._eval_vector_selector(node)
        if isinstance(node, MatrixSelector):
            return _MATRIX, self._eval_matrix_selector(node)
        if isinstance(node, ParenExpr):
            return self.eval(node.expr)
        if isinstance(node, UnaryExpr):
            kind, val = self.eval(node.expr)
            if kind == _SCALAR:
                return _SCALAR, (-val if node.op == "-" else val)
            if kind == _VECTOR:
                return _VECTOR, tuple(
                    VectorItem(it.labels, -it.value if node.op == "-" else it.value)
                    for it in val
                )
            raise TSDBError("unary +/- on matrix is not supported")
        if isinstance(node, Call):
            return self._eval_call(node)
        if isinstance(node, Aggregation):
            return _VECTOR, self._eval_aggregation(node)
        if isinstance(node, BinaryExpr):
            return self._eval_binary(node)
        raise TSDBError(f"unsupported node {type(node).__name__}")  # pragma: no cover

    # ---- selectors -------------------------------------------------------
    def _eval_vector_selector(self, node: VectorSelector) -> Vector:
        self.token.check()
        at = self.at - node.offset
        rows = self.snapshot.select(node.metric, node.matchers, at, self.lookup_delta)
        out: list[VectorItem] = []
        for labels, sample in rows:
            self.token.check()
            if sample is None:
                # staleness: series exists but has no recent sample -> omitted,
                # it must not appear as zero
                continue
            out.append(VectorItem(labels, sample.v))
        return tuple(out)

    def _eval_matrix_selector(self, node: MatrixSelector) -> Matrix:
        self.token.check()
        end = self.at - node.offset
        start = end - node.range_seconds
        # inclusive [start, end]
        rows = self.snapshot.select_range(node.metric, node.matchers, start, end)
        out: list[MatrixItem] = []
        for labels, samples in rows:
            self.token.check()
            out.append(MatrixItem(labels, samples))
        return tuple(out)

    # ---- calls -----------------------------------------------------------
    def _eval_call(self, node: Call) -> tuple[str, object]:
        self.token.check()
        if node.func == "time":
            return _SCALAR, float(self.at)
        if node.func == "abs":
            kind, val = self.eval(node.args[0])
            if kind == _SCALAR:
                return _SCALAR, abs(val)
            return _VECTOR, tuple(VectorItem(it.labels, abs(it.value)) for it in val)
        func = RANGE_FUNCTIONS[node.func]
        kind, matrix = self.eval(node.args[0])
        assert kind == _MATRIX
        return _VECTOR, func(matrix)

    # ---- aggregations ----------------------------------------------------
    def _eval_aggregation(self, node: Aggregation) -> Vector:
        kind, vec = self.eval(node.expr)
        if kind != _VECTOR:
            if kind == _SCALAR:
                # Scalar aggregation is identity: single unlabelled output.
                return (VectorItem({}, vec),)
            raise TSDBError("aggregation requires an instant vector")
        vec: Vector = vec

        if node.op in ("topk", "bottomk"):
            k = max(0, int(node.param.value))
            ordered = sorted(
                vec,
                key=lambda it: (it.value, _labels_sort_key(it.labels)),
                reverse=(node.op == "topk"),
            )
            chosen = ordered[:k]
            # output keeps the ORIGINAL labels of each element, deterministically
            # ordered by (value desc/asc, labels)
            chosen = sorted(
                chosen,
                key=lambda it: (-it.value if node.op == "topk" else it.value,
                                _labels_sort_key(it.labels)),
            )
            return tuple(chosen)

        reducer = AGG_FUNCTIONS[node.op]

        def group_key(it: VectorItem) -> tuple[tuple[str, str], ...]:
            if node.without:
                # without: group on all labels except the listed ones and
                # __name__; output retains all grouped labels.
                drop = set(node.grouping) | {"__name__"}
                return tuple((k, v) for k, v in sorted(it.labels.items()) if k not in drop)
            # by: group exactly on the listed labels; output label set is
            # exactly the grouping labels (explainable shrink).
            gset = set(node.grouping)
            return tuple(sorted((k, v) for k, v in it.labels.items() if k in gset))

        groups: dict[tuple[tuple[str, str], ...], list[VectorItem]] = {}
        for it in vec:
            self.token.check()
            groups.setdefault(group_key(it), []).append(it)

        out: list[VectorItem] = []
        for key, members in groups.items():
            self.token.check()
            value = reducer([m.value for m in members])
            if node.without:
                labels = dict(key)
            else:
                labels = dict(key)
            out.append(VectorItem(labels, value))
        # deterministic ordering: one output per group, sorted by group key
        out.sort(key=lambda it: _labels_sort_key(it.labels))
        return tuple(out)

    # ---- binary operations ----------------------------------------------
    def _eval_binary(self, node: BinaryExpr) -> tuple[str, object]:
        lk, lv = self.eval(node.lhs)
        rk, rv = self.eval(node.rhs)

        if node.op in ("and", "or"):
            if lk != _VECTOR or rk != _VECTOR:
                raise TSDBError("'and'/'or' require instant vectors on both sides")
            return _VECTOR, self._logical(node.op, lv, rv, node.matching)

        is_cmp = node.op in ("==", "!=", ">", "<", ">=", "<=")

        if lk == _SCALAR and rk == _SCALAR:
            value = self._apply(node.op, lv, rv, is_cmp, node.bool_modifier)
            if value is None:
                # filtered comparison between scalars -> unlabelled empty vector
                return _VECTOR, ()
            return _SCALAR, value

        if lk == _SCALAR and rk == _VECTOR:
            return _VECTOR, self._scalar_vector(node, lv, rv, is_cmp, scalar_on_left=True)
        if lk == _VECTOR and rk == _SCALAR:
            return _VECTOR, self._scalar_vector(node, rv, lv, is_cmp, scalar_on_left=False)
        if lk == _VECTOR and rk == _VECTOR:
            return _VECTOR, self._vector_vector(node, lv, rv, is_cmp)
        raise TSDBError(f"cannot combine {lk} and {rk}")

    def _apply(self, op: str, a: float, b: float, is_cmp: bool, bool_modifier: bool):
        if not is_cmp:
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b if b != 0 else _nan()
            if op == "%":
                return a % b if b != 0 else _nan()
            if op == "^":
                return a ** b
        result = _COMPARE[op](a, b)
        if bool_modifier:
            return 1.0 if result else 0.0
        return result  # True -> keep, False -> None (filter)

    def _scalar_vector(
        self,
        node: BinaryExpr,
        scalar: float,
        vec: Vector,
        is_cmp: bool,
        scalar_on_left: bool,
    ) -> Vector:
        out: list[VectorItem] = []
        for it in vec:
            self.token.check()
            if scalar_on_left:
                res = self._apply(node.op, scalar, it.value, is_cmp, node.bool_modifier)
            else:
                res = self._apply(node.op, it.value, scalar, is_cmp, node.bool_modifier)
            if res is None:
                continue
            if is_cmp and isinstance(res, bool):
                if not res:
                    continue
                out.append(VectorItem(it.labels, it.value))
            else:
                out.append(VectorItem(it.labels, float(res)))
        return tuple(out)

    def _vector_vector(
        self, node: BinaryExpr, left: Vector, right: Vector, is_cmp: bool
    ) -> Vector:
        m = node.matching
        left_index = self._index(left, m)
        right_index = self._index(right, m)

        if m.group_side is None:
            # validate one-to-one: each key unique on both sides
            for key in set(left_index) | set(right_index):
                self.token.check()
                if len(left_index.get(key, ())) > 1 or len(right_index.get(key, ())) > 1:
                    raise TSDBError(
                        "multiple series matched one-to-one; use group_left/group_right"
                    )
            many_index, one_index = left_index, right_index
            group_right = False
        elif m.group_side == "left":
            # group_left: LEFT side is the "many" side; right must be unique.
            self._require_unique(right_index, "right")
            many_index, one_index = left_index, right_index
            group_right = False
        else:
            # group_right: RIGHT side is the "many" side; left must be unique.
            self._require_unique(left_index, "left")
            many_index, one_index = right_index, left_index
            group_right = True

        out: list[VectorItem] = []
        for key, many_items in sorted(many_index.items()):
            self.token.check()
            one_items = one_index.get(key)
            if not one_items:
                continue
            one_item = one_items[0]
            for it in many_items:
                self.token.check()
                if group_right:
                    # operator orientation: left op right
                    a, b = one_item.value, it.value
                    # labels are based on the "many" side (right), with the
                    # one side (left) feeding join/include labels.
                    labels = output_labels(one_item.labels, it.labels, m)
                else:
                    a, b = it.value, one_item.value
                    labels = output_labels(it.labels, one_item.labels, m)                # A computed series no longer belongs to either input metric;
                # the metric name is always dropped from binary results.
                labels.pop("__name__", None)
                res = self._apply(node.op, a, b, is_cmp, node.bool_modifier)
                if res is None:
                    continue
                if is_cmp and isinstance(res, bool):
                    if not res:
                        continue
                    out.append(VectorItem(labels, a))
                else:
                    out.append(VectorItem(labels, float(res)))
        out.sort(key=lambda x: _labels_sort_key(x.labels))
        seen: set = set()
        for item in out:
            self.token.check()
            key = tuple(sorted(item.labels.items()))
            if key in seen:
                raise TSDBError(
                    "binary operation would emit multiple series with the same "
                    "label set; refine on()/ignoring()/grouping"
                )
            seen.add(key)
        return tuple(out)

    def _require_unique(self, index: dict, side: str) -> None:
        for key, items in index.items():
            self.token.check()
            if len(items) > 1:
                raise TSDBError(
                    f"duplicate series on {side} side for group match (key={key}); "
                    f"the one side must be unique"
                )

    def _index(self, vec: Vector, m: VectorMatching) -> dict:
        idx: dict[tuple, list[VectorItem]] = {}
        for it in vec:
            self.token.check()
            idx.setdefault(_match_key(it.labels, m), []).append(it)
        return idx

    def _logical(self, op: str, left: Vector, right: Vector, m: VectorMatching) -> Vector:
        right_keys = {_match_key(it.labels, m) for it in right}
        if op == "and":
            kept = [it for it in left if _match_key(it.labels, m) in right_keys]
        else:
            # set union: all left series plus right series not matching any left
            left_keys = {_match_key(it.labels, m) for it in left}
            kept = list(left) + [
                it for it in right if _match_key(it.labels, m) not in left_keys
            ]
        kept.sort(key=lambda x: _labels_sort_key(x.labels))
        return tuple(kept)


def _match_key(labels: dict[str, str], m: VectorMatching) -> tuple[tuple[str, str], ...]:
    """Identity key for binary matching.

    The metric name (``__name__``) never participates in aligning two
    operands: it is implicitly ignored unless the user explicitly lists it in
    an ``on(__name__)`` clause.
    """
    ignoring = m.ignoring
    if not (m.on and "__name__" in m.on):
        ignoring = tuple(set(ignoring) | {"__name__"})
    return labels_key(labels, m.on, ignoring)


def _labels_sort_key(labels: dict[str, str]):
    return tuple(sorted(labels.items()))


def _nan() -> float:
    return float("nan")


_COMPARE = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}


class QueryEngine:
    def __init__(self, storage: Storage, lookup_delta: int = 300) -> None:
        self.storage = storage
        self.lookup_delta = lookup_delta

    def instant_query(
        self,
        expr: str,
        at: int,
        timeout: float | None = None,
        token: CancellationToken | None = None,
    ) -> QueryResult:
        # 1) parse + validate fully BEFORE touching any data: an invalid query
        #    can never partially execute.
        ast = ast_mod.Parser.parse(expr)
        # 2) take exactly one snapshot for the whole evaluation.
        snap = self.storage.snapshot()
        if token is None:
            deadline = time.monotonic() + timeout if timeout is not None else None
            token = CancellationToken(deadline)
        elif timeout is not None:
            raise ValueError("pass either timeout= or a token, not both")
        token.check()
        ev = Evaluator(snapshot=snap, at=at, token=token, lookup_delta=self.lookup_delta)
        kind, value = ev.eval(ast)
        token.check()
        if kind == "scalar":
            return QueryResult(kind="scalar", at=at, scalar=value)
        if kind == "vector":
            return QueryResult(kind="vector", at=at, vector=value)
        return QueryResult(kind="matrix", at=at, matrix=value)
