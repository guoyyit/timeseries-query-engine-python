"""Requirements 4 & 5: binary operations with configurable vector matching,
and aggregation/grouping semantics."""

from __future__ import annotations

import pytest

from tsengine.engine import QueryEngine
from tsengine.errors import QueryTypeError
from tsengine.storage import Storage
from tsengine.value import Scalar, Vector


def make_engine(samples):
    """samples: list of (labels-dict, value) at t=1000."""
    store = Storage()
    for labels, value in samples:
        store.append(labels, 1000, value)
    return QueryEngine(store, staleness_ms=10**12)


def result_map(engine, expr, key="labels"):
    r = engine.execute(expr, at_ms=1000)
    assert isinstance(r, Vector)
    out = {}
    for s in r:
        d = dict(s.labels)
        out[tuple(sorted(d.items()))] = s.value
    return out


L = lambda **kw: {"__name__": "m", **kw}


# -- requirement 4: scalar/vector determinism ------------------------------

def test_vector_plus_scalar_applies_to_each_series():
    eng = make_engine([(L(job="a"), 1.0), (L(job="b"), 2.0)])
    r = result_map(eng, "m + 10")
    assert sorted(r.values()) == [11.0, 12.0]


def test_scalar_vector_sides_are_deterministic():
    eng = make_engine([(L(job="a"), 10.0)])
    assert result_map(eng, "20 - m")[("__name__", "m"), ("job", "a")] == 10.0
    assert result_map(eng, "m - 20")[("__name__", "m"), ("job", "a")] == -10.0


def test_scalar_scalar_arithmetic():
    eng = make_engine([])
    r = eng.execute("1 + 2 * 3", at_ms=1000)
    assert isinstance(r, Scalar) and r.value == 7.0


def test_comparison_filters_series():
    eng = make_engine([(L(job="a"), 5.0), (L(job="b"), 15.0)])
    r = result_map(eng, "m > 10")
    jobs = {dict(k)["job"] for k in r}
    assert jobs == {"b"}


def test_comparison_bool_keeps_all_series():
    eng = make_engine([(L(job="a"), 5.0), (L(job="b"), 15.0)])
    r = result_map(eng, "m > bool 10")
    vals = {dict(k)["job"]: v for k, v in r.items()}
    assert vals == {"a": 0.0, "b": 1.0}


def test_division_by_zero_drops_sample():
    eng = make_engine([(L(job="a"), 4.0)])
    r = eng.execute("m / 0", at_ms=1000)
    assert isinstance(r, Vector) and len(r) == 0


# -- requirement 4: vector-vector matching ---------------------------------

def test_default_one_to_one_requires_full_label_match():
    eng = make_engine([
        ({"__name__": "l", "job": "a", "code": "200"}, 1.0),
        ({"__name__": "r", "job": "a", "code": "500"}, 2.0),
    ])
    # full label sets differ (incl __name__ and code) => nothing matches
    r = eng.execute("l + r", at_ms=1000)
    assert len(r) == 0


def test_ignoring_aligns_series():
    eng = make_engine([
        ({"__name__": "l", "job": "a", "code": "200"}, 100.0),
        ({"__name__": "r", "job": "a", "code": "500"}, 5.0),
        ({"__name__": "r", "job": "b", "code": "500"}, 7.0),
    ])
    r = result_map(eng, 'l + ignoring(code, __name__) r')
    # only job=a has both sides; labels come from the LEFT
    assert len(r) == 1
    (labels, value), = r.items()
    d = dict(labels)
    assert d["job"] == "a" and d["code"] == "200"
    assert value == 105.0


def test_on_clause_matches_on_subset():
    eng = make_engine([
        ({"__name__": "l", "job": "a", "instance": "x"}, 3.0),
        ({"__name__": "r", "job": "a", "instance": "y"}, 4.0),
    ])
    r = result_map(eng, "l + on(job) r")
    assert len(r) == 1
    assert next(iter(r.values())) == 7.0


def test_many_to_many_without_grouping_is_rejected():
    eng = make_engine([
        ({"__name__": "l", "job": "a", "code": "200"}, 1.0),
        ({"__name__": "l", "job": "a", "code": "500"}, 1.0),
        ({"__name__": "r", "job": "a", "code": "200"}, 2.0),
        ({"__name__": "r", "job": "a", "code": "500"}, 2.0),
    ])
    with pytest.raises(QueryTypeError):
        eng.execute("l + on(job) r", at_ms=1000)


def test_group_left_many_to_one():
    eng = make_engine([
        # left (many side): one series per code
        ({"__name__": "l", "job": "a", "code": "200"}, 100.0),
        ({"__name__": "l", "job": "a", "code": "500"}, 5.0),
        ({"__name__": "l", "job": "b", "code": "200"}, 200.0),
        # right (one side): per job totals
        ({"__name__": "r", "job": "a"}, 105.0),
        ({"__name__": "r", "job": "b"}, 200.0),
    ])
    r = result_map(eng, "l / on(job) group_left r")
    vals = {(dict(k)["job"], dict(k)["code"]): v for k, v in r.items()}
    assert set(vals) == {("a", "200"), ("a", "500"), ("b", "200")}
    assert vals[("a", "200")] == pytest.approx(100 / 105)
    assert vals[("a", "500")] == pytest.approx(5 / 105)
    assert vals[("b", "200")] == pytest.approx(1.0)


def test_group_right_many_to_one():
    eng = make_engine([
        ({"__name__": "l", "job": "a"}, 105.0),
        ({"__name__": "r", "job": "a", "code": "200"}, 100.0),
        ({"__name__": "r", "job": "a", "code": "500"}, 5.0),
    ])
    r = result_map(eng, "l / on(job) group_right r")
    codes = {dict(k)["code"] for k in r}
    assert codes == {"200", "500"}
    vals = {dict(k)["code"]: v for k, v in r.items()}
    # expression order is always l / r (105 / 100 and 105 / 5)
    assert vals["200"] == pytest.approx(105 / 100)
    assert vals["500"] == pytest.approx(105 / 5)


def test_set_operators_and_or_unless():
    eng = make_engine([
        ({"__name__": "m", "job": "a"}, 1.0),
        ({"__name__": "m", "job": "b"}, 2.0),
        ({"__name__": "m", "job": "a", "code": "x"}, 3.0),
    ])
    and_r = eng.execute('m{job="a"} and m{code="x"}', at_ms=1000)
    # intersection by FULL label set => only the series with code=x
    assert len(and_r) == 1 and dict(list(and_r)[0].labels)["code"] == "x"

    or_r = result_map(eng, 'm{job="b"} or m{job="a"}')
    # union: job b (1) + both job a (2) = 3
    assert len(or_r) == 3

    unless_r = eng.execute('m unless m{code="x"}', at_ms=1000)
    # everything except series with code=x
    assert len(unless_r) == 2
    assert all("code" not in dict(s.labels) for s in unless_r)


def test_unmatched_series_dropped_not_zero():
    eng = make_engine([
        ({"__name__": "l", "job": "a"}, 1.0),
        ({"__name__": "r", "job": "zzz"}, 9.0),
    ])
    r = eng.execute("l + on(job) r", at_ms=1000)
    assert len(r) == 0  # not a 0-filled result


# -- requirement 5: aggregation / grouping ---------------------------------

def test_group_emits_once_per_group():
    eng = make_engine([
        (L(job="a", code="200"), 1.0),
        (L(job="a", code="500"), 2.0),
        (L(job="b", code="200"), 3.0),
        (L(job="b", code="500"), 4.0),
    ])
    r = result_map(eng, "sum(m) by (job)")
    assert len(r) == 2
    vals = {dict(k)["job"]: v for k, v in r.items()}
    assert vals == {"a": 3.0, "b": 7.0}


def test_by_keeps_only_grouping_labels():
    eng = make_engine([(L(job="a", code="200", instance="x"), 1.0)])
    r = result_map(eng, "sum(m) by (job)")
    (labels, _), = r.items()
    assert dict(labels) == {"job": "a"}


def test_without_removes_named_labels_keeps_rest():
    eng = make_engine([(L(job="a", code="200", instance="x"), 1.0)])
    r = result_map(eng, "sum(m) without (code)")
    (labels, _), = r.items()
    assert dict(labels) == {"__name__": "m", "job": "a", "instance": "x"}


def test_aggregate_all_drops_labels():
    eng = make_engine([(L(job="a"), 1.0), (L(job="b"), 2.0)])
    r = result_map(eng, "sum(m)")
    (labels, value), = r.items()
    assert dict(labels) == {} and value == 3.0


def test_aggregation_functions():
    eng = make_engine([
        (L(job="a", instance="x"), 2.0),
        (L(job="a", instance="y"), 4.0),
        (L(job="a", instance="z"), 6.0),
    ])
    assert list(result_map(eng, "avg(m) by (job)").values()) == [4.0]
    assert list(result_map(eng, "count(m) by (job)").values()) == [3.0]
    assert list(result_map(eng, "min(m) by (job)").values()) == [2.0]
    assert list(result_map(eng, "max(m) by (job)").values()) == [6.0]
    # population stddev of [2,4,6] is sqrt(8/3)
    assert list(result_map(eng, "stddev(m) by (job)").values()) == \
        [pytest.approx((8 / 3) ** 0.5)]
    assert list(result_map(eng, "stdvar(m) by (job)").values()) == \
        [pytest.approx(8 / 3)]


def test_topk_preserves_labels():
    eng = make_engine([
        (L(job="a"), 1.0), (L(job="b"), 3.0), (L(job="c"), 2.0),
    ])
    r = result_map(eng, "topk(2, m)")
    jobs = {dict(k)["job"] for k in r}
    assert jobs == {"b", "c"}  # labels preserved, not grouped away


def test_bottomk():
    eng = make_engine([
        (L(job="a"), 1.0), (L(job="b"), 3.0), (L(job="c"), 2.0),
    ])
    r = result_map(eng, "bottomk(1, m)")
    assert {dict(k)["job"] for k in r} == {"a"}


def test_count_values_buckets():
    eng = make_engine([
        (L(job="a", instance="x"), 1.0),
        (L(job="a", instance="y"), 1.0),
        (L(job="b", instance="z"), 2.0),
    ])
    r = eng.execute('count_values("val", m) by (job, val)', at_ms=1000)
    keys = {(dict(s.labels)["job"], dict(s.labels)["val"]): s.value
            for s in r}
    assert keys == {("a", "1"): 2.0, ("b", "2"): 1.0}


def test_empty_group_results_are_empty_not_zero_vector():
    eng = make_engine([])
    r = eng.execute("sum(m) by (job)", at_ms=1000)
    assert isinstance(r, Vector) and len(r) == 0
