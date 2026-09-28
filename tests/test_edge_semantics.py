"""Additional deterministic edge semantics."""

from __future__ import annotations

import math

import pytest

from tsengine.engine import QueryEngine
from tsengine.storage import Storage
from tsengine.value import Scalar, Vector


def _eng(samples, staleness_ms=10**12, at=1000):
    store = Storage()
    for labels, value in samples:
        store.append(labels, at, value)
    eng = QueryEngine(store, staleness_ms=staleness_ms)
    # bind every unparameterised execute() to the dataset timestamp
    _orig = eng.execute

    def execute(expr, *, at_ms=None, timeout_ms=None, ctx=None):
        return _orig(expr, at_ms=at if at_ms is None else at_ms,
                     timeout_ms=timeout_ms, ctx=ctx)

    eng.execute = execute  # type: ignore[method-assign]
    return eng


def test_staleness_boundary_inclusive_within_window():
    store = Storage()
    store.append({"__name__": "m"}, 1000, 5.0)
    eng = QueryEngine(store, staleness_ms=1000)
    # age exactly == staleness is still visible (age > staleness hides)
    assert len(eng.execute("m", at_ms=2000)) == 1
    assert len(eng.execute("m", at_ms=2001)) == 0


def test_bool_after_matching_clause():
    eng = _eng([
        ({"__name__": "l", "job": "a", "code": "200"}, 5.0),
        ({"__name__": "r", "job": "a"}, 10.0),
    ])
    r = eng.execute("l > on(job) group_left bool r")
    assert isinstance(r, Vector)
    samples = list(r)
    assert len(samples) == 1 and samples[0].value == 0.0


def test_double_bool_rejected():
    from tsengine.errors import QuerySyntaxError
    from tsengine.parser import parse
    with pytest.raises(QuerySyntaxError):
        parse("a > bool bool 1")


def test_scalar_function_on_empty_and_multiple_is_nan():
    eng = _eng([
        ({"__name__": "m", "job": "a"}, 1.0),
        ({"__name__": "m", "job": "b"}, 2.0),
    ])
    r0 = eng.execute("scalar(nonexistent)")
    assert isinstance(r0, Scalar) and math.isnan(r0.value)
    r2 = eng.execute("scalar(m)")
    assert isinstance(r2, Scalar) and math.isnan(r2.value)  # 2 series
    r1 = eng.execute('scalar(m{job="a"})')
    assert isinstance(r1, Scalar) and r1.value == 1.0


def test_unary_minus_on_vector_and_scalar():
    eng = _eng([({"__name__": "m", "job": "a"}, 3.0)])
    assert list(eng.execute("-m"))[0].value == -3.0
    assert eng.execute("-5 + 2").value == -3.0


def test_numeric_functions_preserve_kind():
    eng = _eng([({"__name__": "m", "job": "a"}, 4.0)])
    assert eng.execute("abs(-m)").__class__.__name__ == "Vector"
    assert eng.execute("sqrt(16)").value == 4.0
    assert list(eng.execute("ceil(m)"))[0].value == 4.0
    assert list(eng.execute("round(m * 1.4)"))[0].value == 6.0
    assert list(eng.execute("round(m * 1.4, 5)"))[0].value == 5.0


def test_modulo_and_power():
    eng = _eng([({"__name__": "m"}, 7.0)])
    assert list(eng.execute("m % 3"))[0].value == pytest.approx(1.0)
    assert list(eng.execute("m ^ 2"))[0].value == pytest.approx(49.0)


def test_out_of_order_and_duplicate_timestamps():
    store = Storage()
    labels = {"__name__": "m"}
    for ts, v in [(3000, 3.0), (1000, 1.0), (2000, 2.0), (2000, 22.0)]:
        store.append(labels, ts, v)
    eng = QueryEngine(store, staleness_ms=10**12)
    # duplicate timestamp overwrites; order is ascending
    r = list(eng.execute("m", at_ms=3000))
    assert r[0].value == 3.0  # latest point
    series = list(store.current_snapshot().series.values())[0]
    assert series.points == ((1000, 1.0), (2000, 22.0), (3000, 3.0))


def test_final_vector_results_have_unique_label_sets():
    """Even constructed results must not emit the same label set twice."""
    eng = _eng([
        ({"__name__": "l", "job": "a"}, 1.0),
        ({"__name__": "l", "job": "b"}, 2.0),
        ({"__name__": "r", "job": "a"}, 9.0),
        ({"__name__": "r", "job": "b"}, 8.0),
    ])
    for expr in [
        "l + on(job) r",
        "l or on(job) r",
        "l and on(job) r",
    ]:
        r = eng.execute(expr)
        keys = [tuple(sorted(s.labels)) for s in r]
        assert len(keys) == len(set(keys)), expr
