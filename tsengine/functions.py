"""Built-in range/vector functions.

Window semantics (requirement 3)
--------------------------------
Range functions consume a :class:`Matrix` whose rows already contain exactly
the samples inside the selector's **half-open** window
``(t - offset - range, t - offset]``. Only points actually present in that
window are used; missing samples are never substituted with zero. A series
that has fewer points than an operator needs (e.g. ``rate`` with a single
point) yields no output sample for that series rather than a fabricated one.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .errors import QueryTypeError
from .value import Matrix, Sample, Series, Vector

RangeFunc = Callable[[Matrix, int], Vector]


def _emit(matrix: Matrix, at_ms: int,
          calc: Callable[[Series], float | None]) -> Vector:
    samples: list[Sample] = []
    for s in matrix:
        if not s.points:
            continue  # no samples in window -> no output (not zero!)
        result = calc(s)
        if result is None:
            continue
        samples.append(Sample(labels=s.labels, value=result,
                              timestamp=at_ms))
    return Vector(samples)


def _extrapolated_rate(s: Series, range_ms: int,
                       is_increase: bool) -> float | None:
    """Per-second rate / counter increase over the points in the window.

    Deterministic simplification (no edge extrapolation):

    * requires at least two points;
    * counter resets (a decrease between adjacent points) are compensated
      by adding the post-reset value (the counter is assumed to restart at
      zero), so the total increase sums every monotonic segment;
    * ``rate`` divides that total by the elapsed time between the first
      and last points present;
    * ``increase`` scales the rate by the requested window length.
    """
    pts = s.points
    if len(pts) < 2:
        return None
    first_ts, first_v = pts[0]
    last_ts, _ = pts[-1]
    duration_s = (last_ts - first_ts) / 1000.0
    if duration_s <= 0:
        return None
    total = 0.0
    prev = first_v
    for _, v in pts[1:]:
        if v >= prev:
            total += v - prev
        else:
            # counter reset: drop to zero then climb to v
            total += v
        prev = v
    rate_per_second = total / duration_s
    if is_increase:
        return rate_per_second * (range_ms / 1000.0)
    return rate_per_second


def rate(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms,
                 lambda s: _extrapolated_rate(s, range_ms, False))


def irate(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    """Instant rate from the last two points only; needs >= 2 points."""
    def calc(s: Series) -> float | None:
        if len(s.points) < 2:
            return None
        (t0, v0), (t1, v1) = s.points[-2], s.points[-1]
        dt_s = (t1 - t0) / 1000.0
        if dt_s <= 0:
            return None
        diff = v1 - v0 if v1 >= v0 else v1
        return diff / dt_s
    return _emit(matrix, at_ms, calc)


def increase(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms,
                 lambda s: _extrapolated_rate(s, range_ms, True))


def delta(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    """Simple difference between first and last value (no reset handling)."""
    def calc(s: Series) -> float | None:
        if len(s.points) < 2:
            return None
        return s.points[-1][1] - s.points[0][1]
    return _emit(matrix, at_ms, calc)


def avg_over_time(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms,
                 lambda s: sum(v for _, v in s.points) / len(s.points))


def sum_over_time(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms, lambda s: sum(v for _, v in s.points))


def max_over_time(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms, lambda s: max(v for _, v in s.points))


def min_over_time(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms, lambda s: min(v for _, v in s.points))


def count_over_time(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    # counts samples actually present; never zero-fills
    return _emit(matrix, at_ms, lambda s: float(len(s.points)))


def last_over_time(matrix: Matrix, at_ms: int, range_ms: int = 0) -> Vector:
    return _emit(matrix, at_ms, lambda s: s.points[-1][1])


_RANGE_IMPL: dict[str, RangeFunc] = {
    "rate": rate,
    "irate": irate,
    "increase": increase,
    "delta": delta,
    "avg_over_time": avg_over_time,
    "sum_over_time": sum_over_time,
    "max_over_time": max_over_time,
    "min_over_time": min_over_time,
    "count_over_time": count_over_time,
    "last_over_time": last_over_time,
}

# names the parser knows about (kept in sync; parser is the source of truth
# for which names accept a matrix argument)
RANGE_FUNCTIONS: frozenset[str] = frozenset(_RANGE_IMPL)


def call_range(name: str, matrix: Matrix, at_ms: int,
               range_ms: int) -> Vector:
    func = _RANGE_IMPL.get(name)
    if func is None:
        raise QueryTypeError(f"not a range function: {name}")
    return func(matrix, at_ms, range_ms)


# -- instant vector / scalar functions -------------------------------------

def _map_values(vec: Vector, fn: Callable[[float], float]) -> Vector:
    return Vector(Sample(labels=s.labels, value=fn(s.value),
                         timestamp=s.timestamp) for s in vec)


def call_unary_numeric(name: str, value, at_ms: int):
    if isinstance(value, (int, float)):
        return _SCALAR_FUNCS[name](float(value))
    if name in ("sort", "sort_desc"):
        return value
    return _map_values(value, _SCALAR_FUNCS[name])


_SCALAR_FUNCS: dict[str, Callable[[float], float]] = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "sgn": lambda v: (v > 0) - (v < 0),
}
