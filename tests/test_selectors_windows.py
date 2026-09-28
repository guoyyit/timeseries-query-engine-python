"""Requirements 2 & 3: selectors/label matching and deterministic window
semantics (half-open window, missing samples are never zero)."""

from __future__ import annotations

import pytest

from tsengine.engine import QueryEngine
from tsengine.storage import Storage
from tsengine.value import Vector


@pytest.fixture
def engine():
    store = Storage()
    t0 = 60_000
    samples = [
        # job=a, one sample every 10s
        ({"__name__": "m", "job": "a"}, 0, 0.0),
        ({"__name__": "m", "job": "a"}, 10_000, 10.0),
        ({"__name__": "m", "job": "a"}, 20_000, 20.0),
        ({"__name__": "m", "job": "a"}, 30_000, 30.0),
        ({"__name__": "m", "job": "a"}, 40_000, 40.0),
        ({"__name__": "m", "job": "a"}, 50_000, 50.0),
        ({"__name__": "m", "job": "a"}, 60_000, 60.0),
        # job=b has a GAP: missing samples between 20s and 60s
        ({"__name__": "m", "job": "b"}, 0, 1.0),
        ({"__name__": "m", "job": "b"}, 10_000, 2.0),
        ({"__name__": "m", "job": "b"}, 20_000, 4.0),
        ({"__name__": "m", "job": "b"}, 60_000, 8.0),
        # job=c's only sample is 10 minutes before query time (stale)
        ({"__name__": "m", "job": "c"}, -10 * 60_000, 99.0),
    ]
    for labels, ts, val in samples:
        store.append(labels, t0 + ts, val)
    return QueryEngine(store, staleness_ms=5 * 60_000), t0 + 60_000


def values(result) -> dict:
    assert isinstance(result, Vector)
    return {dict(s.labels)["job"]: s.value for s in result}


# -- requirement 2: selectors ----------------------------------------------

def test_select_by_name_returns_multiple_series(engine):
    eng, at = engine
    r = eng.execute("m", at_ms=at)
    assert set(values(r)) == {"a", "b"}  # c is stale, excluded


def test_select_by_equality_label(engine):
    eng, at = engine
    r = eng.execute('m{job="a"}', at_ms=at)
    assert set(values(r)) == {"a"}
    assert values(r)["a"] == 60.0


def test_select_by_inequality(engine):
    eng, at = engine
    r = eng.execute('m{job!="a"}', at_ms=at)
    assert set(values(r)) == {"b"}


def test_select_regex_and_negated_regex(engine):
    eng, at = engine
    r = eng.execute('m{job=~"a|b"}', at_ms=at)
    assert set(values(r)) == {"a", "b"}
    r2 = eng.execute('m{job!~"a"}', at_ms=at)
    assert set(values(r2)) == {"b"}


def test_selector_label_presence():
    store = Storage()
    store.append({"__name__": "m", "env": "prod"}, 1000, 1.0)
    store.append({"__name__": "m"}, 1000, 2.0)
    eng = QueryEngine(store, staleness_ms=10**12)
    r = eng.execute('m{env!=""}', at_ms=1000)
    envs = {dict(s.labels).get("env") for s in r}
    assert envs == {"prod"}
    # env="" matches series without the env label
    r2 = eng.execute('m{env=""}', at_ms=1000)
    assert len(r2) == 1 and "env" not in dict(list(r2)[0].labels)


# -- requirement 3: window boundaries --------------------------------------

def test_window_is_half_open(engine):
    """Window (t-30s, t] must EXCLUDE the boundary sample at t-30s."""
    eng, at = engine
    r = eng.execute("avg_over_time(m{job=\"a\"}[30s])", at_ms=at)
    v = values(r)
    # included: 40,50,60 ; excluded: 30 (at t-30s, lower bound exclusive)
    assert v["a"] == pytest.approx((40 + 50 + 60) / 3)


def test_window_includes_upper_boundary(engine):
    eng, at = engine
    r = eng.execute("count_over_time(m{job=\"a\"}[10s])", at_ms=at)
    # (t-10s, t] -> 50 excluded, 60 included => 1 point
    assert values(r)["a"] == 1.0


def test_window_point_count(engine):
    eng, at = engine
    r = eng.execute("count_over_time(m{job=\"a\"}[60s])", at_ms=at)
    # (t-60s, t] -> points at 10..60 => 6 points; t=0 excluded
    assert values(r)["a"] == 6.0


def test_missing_samples_are_not_zero_in_avg(engine):
    eng, at = engine
    # job=b window (t-40s,t] contains ONLY t=60 sample (value 8).
    # Missing samples at 30/40/50 must NOT be treated as zero.
    r = eng.execute("avg_over_time(m{job=\"b\"}[40s])", at_ms=at)
    assert values(r)["b"] == 8.0  # not (8+0+0+0)/4


def test_missing_samples_are_not_zero_in_sum(engine):
    eng, at = engine
    r = eng.execute("sum_over_time(m{job=\"b\"}[60s])", at_ms=at)
    # only t=60 (8.0) is in (0, 60]; t=0 is excluded, 10/20 are at boundary
    # window start is t-60 -> t=0 excluded; points at 10,20 are >0 so...
    # window (0,60s relative] => 10,20,60 => 2+4+8 = 14
    assert values(r)["b"] == pytest.approx(14.0)


def test_rate_requires_two_points_and_emits_nothing_for_one(engine):
    eng, at = engine
    r = eng.execute("rate(m{job=\"b\"}[30s])", at_ms=at)
    # single point in window -> series must be absent, not zero
    assert "b" not in values(r)


def test_rate_uses_only_present_points(engine):
    eng, at = engine
    r = eng.execute("rate(m{job=\"a\"}[20s])", at_ms=at)
    # points at 40,60 (50 also): (60-40)/20s = 1.0
    assert values(r)["a"] == pytest.approx(1.0)


def test_rate_counter_reset():
    at = 120_000
    store = Storage()
    labels = {"__name__": "c"}
    store.append(labels, at - 30_000, 100.0)
    store.append(labels, at - 20_000, 90.0)   # reset
    store.append(labels, at - 10_000, 20.0)   # reset
    eng = QueryEngine(store, staleness_ms=10**12)
    r = eng.execute("rate(c[35s])", at_ms=at)
    # reset: 90 + 20 over 20s = 5.5
    assert list(r)[0].value == pytest.approx(5.5)


def test_stale_series_not_in_instant_vector(engine):
    eng, at = engine
    r = eng.execute('m{job="c"}', at_ms=at)
    assert len(r) == 0


def test_offset_shifts_window(engine):
    eng, at = engine
    # evaluate 30s ago: window (0,30s] relative to t-30s => absolute
    # (at-60s, at-30s]; job a points 10,20,30 relative to t0 => count 3
    r = eng.execute(
        "count_over_time(m{job=\"a\"}[30s] offset 30s)", at_ms=at)
    assert values(r)["a"] == 3.0


def test_last_over_time(engine):
    eng, at = engine
    r = eng.execute("last_over_time(m{job=\"b\"}[10s])", at_ms=at)
    assert values(r)["b"] == 8.0


def test_min_max_over_time(engine):
    eng, at = engine
    rmin = eng.execute("min_over_time(m{job=\"a\"}[40s])", at_ms=at)
    rmax = eng.execute("max_over_time(m{job=\"a\"}[40s])", at_ms=at)
    # (20,60] => 30,40,50,60
    assert values(rmin)["a"] == 30.0
    assert values(rmax)["a"] == 60.0


def test_instant_selector_returns_latest_before_query(engine):
    eng, at = engine
    # value at query time is the latest sample <= at within staleness
    r = eng.execute('m{job="a"}', at_ms=at)
    assert values(r)["a"] == 60.0


def test_invalid_window_expr_raises_not_empty(engine):
    """A malformed range query must raise, never silently return an empty
    vector that a caller could mistake for 'no data'."""
    eng, at = engine
    from tsengine.errors import QuerySyntaxError
    with pytest.raises(QuerySyntaxError):
        eng.execute("rate(m[)", at_ms=at)


def test_missing_points_not_zero_in_increase_and_count():
    """Gap in the middle of a window must not be zero-filled: increase is
    derived only from the points present, and count counts actual points."""
    store = Storage()
    L = {"__name__": "g"}
    # gap between t=10s and t=60s
    store.append(L, 0, 100.0)
    store.append(L, 10_000, 100.0)
    store.append(L, 60_000, 160.0)
    eng = QueryEngine(store, staleness_ms=10**12)
    at = 60_000
    # window (20s,60s] contains only the 160 point => count 1, not 5
    cnt = eng.execute("count_over_time(g[40s])", at_ms=at)
    assert list(cnt)[0].value == 1.0
    avg = eng.execute("avg_over_time(g[40s])", at_ms=at)
    # average over ONE present sample is 160, not 160/5 with zero fill
    assert list(avg)[0].value == 160.0
    # rate with a single in-window point yields nothing (needs 2 points)
    r = eng.execute("rate(g[40s])", at_ms=at)
    assert len(r) == 0
