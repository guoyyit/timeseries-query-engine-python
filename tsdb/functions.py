"""Built-in range-vector functions and aggregation operators.

Window semantics (fully deterministic):
  * the window is the inclusive interval supplied by the matrix selector;
  * samples are the stored samples inside that interval — missing samples are
    NEVER synthesized as zeros, and series whose window is empty (or which do
    not meet the minimum-sample requirement of the function) produce NO
    output element;
  * ``rate``/``increase`` use the first and last samples of the window only.
    ``increase`` = last-first with counter-reset correction (see below);
    ``rate`` = corrected increase / (t_last - t_first).
  * counter reset: if the last value is below the first value the counter is
    assumed to have reset, and the increase is the last value itself.
"""

from __future__ import annotations

from .model import (
    Matrix,
    Vector,
    VectorItem,
)


def _counter_increase(first_v: float, last_v: float) -> float:
    delta = last_v - first_v
    if delta < 0:
        # counter reset: the only observable increase after reset is last_v
        return last_v
    return delta


def rate(matrix: Matrix) -> Vector:
    out: list[VectorItem] = []
    for item in matrix:
        if len(item.values) < 2:
            continue
        first, last = item.values[0], item.values[-1]
        elapsed = last.t - first.t
        if elapsed <= 0:
            continue
        out.append(VectorItem(item.labels, _counter_increase(first.v, last.v) / elapsed))
    return tuple(out)


def increase(matrix: Matrix) -> Vector:
    out: list[VectorItem] = []
    for item in matrix:
        if len(item.values) < 2:
            continue
        first, last = item.values[0], item.values[-1]
        if last.t - first.t <= 0:
            continue
        out.append(VectorItem(item.labels, _counter_increase(first.v, last.v)))
    return tuple(out)


def avg_over_time(matrix: Matrix) -> Vector:
    out: list[VectorItem] = []
    for item in matrix:
        if not item.values:
            continue
        total = sum(s.v for s in item.values)
        out.append(VectorItem(item.labels, total / len(item.values)))
    return tuple(out)


def sum_over_time(matrix: Matrix) -> Vector:
    return tuple(
        VectorItem(item.labels, sum(s.v for s in item.values))
        for item in matrix
        if item.values
    )


def max_over_time(matrix: Matrix) -> Vector:
    return tuple(
        VectorItem(item.labels, max(s.v for s in item.values))
        for item in matrix
        if item.values
    )


def min_over_time(matrix: Matrix) -> Vector:
    return tuple(
        VectorItem(item.labels, min(s.v for s in item.values))
        for item in matrix
        if item.values
    )


def count_over_time(matrix: Matrix) -> Vector:
    return tuple(
        VectorItem(item.labels, float(len(item.values)))
        for item in matrix
        if item.values
    )


RANGE_FUNCTIONS = {
    "rate": rate,
    "increase": increase,
    "avg_over_time": avg_over_time,
    "sum_over_time": sum_over_time,
    "max_over_time": max_over_time,
    "min_over_time": min_over_time,
    "count_over_time": count_over_time,
}


# ---- aggregation operators --------------------------------------------------
# Each aggregator reduces a list of float values to one float.

def agg_sum(items: list[float]) -> float:
    return sum(items)


def agg_avg(items: list[float]) -> float:
    return sum(items) / len(items)


def agg_count(items: list[float]) -> float:
    return float(len(items))


def agg_min(items: list[float]) -> float:
    return min(items)


def agg_max(items: list[float]) -> float:
    return max(items)


AGG_FUNCTIONS = {
    "sum": agg_sum,
    "avg": agg_avg,
    "count": agg_count,
    "min": agg_min,
    "max": agg_max,
}

# topk/bottomk are handled specially in the engine because they emit more
# than one group (parameter + per-group ordering).
