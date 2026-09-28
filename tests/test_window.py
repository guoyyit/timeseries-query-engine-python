"""Window boundary semantics and missing-sample handling."""

from __future__ import annotations

from tsdb.model import Sample


def _vals(engine, expr, at):
    res = engine.instant_query(expr, at=at)
    return {
        tuple(sorted({k: v for k, v in it.labels.items() if k != "__name__"}.items())): it.value
        for it in res.vector
    }


def _matrix_labels(engine, expr, at):
    res = engine.instant_query(expr, at=at)
    return res.matrix


def test_window_is_inclusive_on_both_boundaries(engine, storage):
    # samples exactly on the window edges must both be included
    storage.append("m", {"h": "a"}, [Sample(100, 1.0), Sample(130, 2.0), Sample(160, 3.0)])
    res = _vals(engine, "avg_over_time(m[60s])", at=160)
    # window [100,160]: mean(1,2,3) == 2
    assert res == {(("h", "a"),): 2.0}
    res = _vals(engine, "sum_over_time(m[60s])", at=159)
    # window [99,159]: samples 100,130 -> 3
    assert res == {(("h", "a"),): 3.0}


def test_rate_uses_first_and_last_window_samples(engine, storage):
    storage.append("m", {"h": "a"}, [Sample(100, 10.0), Sample(120, 999.0), Sample(160, 40.0)])
    res = _vals(engine, "rate(m[60s])", at=160)
    # (40 - 10) / 60 = 0.5; interior point ignored
    assert res == {(("h", "a"),): 0.5}
    res = _vals(engine, "increase(m[60s])", at=160)
    assert res == {(("h", "a"),): 30.0}


def test_rate_requires_two_samples_no_zero_fill(engine, storage):
    # only one sample in window -> series dropped, not treated as 0
    storage.append("m", {"h": "a"}, [Sample(160, 7.0)])
    storage.append("m", {"h": "b"}, [Sample(100, 1.0), Sample(160, 3.0)])
    res = _vals(engine, "rate(m[60s])", at=160)
    assert res == {(("h", "b"),): (3.0 - 1.0) / 60}


def test_empty_window_series_dropped_not_zero(engine, storage):
    storage.append("m", {"h": "a"}, [Sample(5, 1.0)])  # far outside window
    res = _vals(engine, "sum_over_time(m[60s])", at=1000)
    assert res == {}
    res = _vals(engine, "count_over_time(m[60s])", at=1000)
    assert res == {}
    res = _vals(engine, "max_over_time(m[60s])", at=1000)
    assert res == {}


def test_counter_reset_increase(engine, storage):
    storage.append("m", {"h": "a"}, [Sample(100, 90.0), Sample(160, 5.0)])
    res = _vals(engine, "increase(m[60s])", at=160)
    # reset observed: increase is last value (5), not 5-90 = -85
    assert res == {(("h", "a"),): 5.0}
    res = _vals(engine, "rate(m[60s])", at=160)
    assert res == {(("h", "a"),): 5.0 / 60}


def test_instant_lookback_staleness_is_none_not_zero(engine, storage):
    storage.append("m", {"h": "a"}, [Sample(0, 9.0)])
    # default staleness delta 300s: at=300 still visible (inclusive edge)
    assert _vals(engine, "m", at=300) == {(("h", "a"),): 9.0}
    # at=301 outside window -> series omitted entirely
    assert _vals(engine, "m", at=301) == {}


def test_matrix_selector_returns_real_samples_only(engine, storage):
    storage.append("m", {"h": "a"}, [Sample(100, 1.0), Sample(160, 2.0)])
    # matrix results are only reachable through functions, but verify via
    # count_over_time that gaps never synthesize points.
    res = _vals(engine, "count_over_time(m[1000s])", at=160)
    assert res == {(("h", "a"),): 2.0}  # 2 real samples, not 1001 zero-filled slots


def test_offset_shifts_window(engine, storage):
    storage.append("m", {"h": "a"}, [
        Sample(40, 1.0), Sample(100, 2.0), Sample(160, 100.0),
    ])
    # current window [100,160] mean 51; offset 60 -> [40,100] mean 1.5
    res = _vals(engine, "avg_over_time(m[60s] offset 60s)", at=160)
    assert res == {(("h", "a"),): 1.5}


def test_aggregations_over_window_functions(engine, storage):
    storage.append("m", {"h": "a"}, [Sample(100, 1.0), Sample(160, 3.0)])
    storage.append("m", {"h": "b"}, [Sample(100, 10.0), Sample(160, 20.0)])
    res = engine.instant_query("sum(rate(m[60s]))", at=160).vector
    assert len(res) == 1
    assert dict(res[0].labels) == {}
    # (3-1)/60 + (20-10)/60
    assert abs(res[0].value - ((2.0 + 10.0) / 60)) < 1e-12
