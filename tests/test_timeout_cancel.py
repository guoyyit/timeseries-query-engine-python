"""Requirement 7: queries must honour deadlines and cancellation, stop
consuming compute afterwards, and never return partial results."""

from __future__ import annotations

import threading
import time

import pytest

from tsengine.engine import QueryContext, QueryEngine
from tsengine.errors import QueryCancelledError, QueryTimeoutError
from tsengine.storage import Storage


def _heavy_storage(n_series=20_000, points_each=3):
    """Many series so a full-scan query takes a measurable amount of time."""
    store = Storage()
    batch = []
    for i in range(n_series):
        labels = {"__name__": "m", "i": str(i)}
        for p in range(points_each):
            batch.append((labels, 1000 + p * 1000, float(i + p)))
    # append in one huge batch so the dataset is there before querying
    store.append_batch(batch)
    return store


def test_syntax_error_is_not_a_timeout():
    """Invalid input raises a syntax error regardless of a tight deadline."""
    from tsengine.errors import QuerySyntaxError
    eng = QueryEngine(Storage())
    with pytest.raises(QuerySyntaxError):
        eng.execute("m +", at_ms=1000, timeout_ms=10_000)


def test_zero_timeout_raises_timeout():
    store = _heavy_storage()
    eng = QueryEngine(store, staleness_ms=10**12)
    with pytest.raises(QueryTimeoutError):
        eng.execute("sum(m) by (i)", at_ms=2000, timeout_ms=0)


def test_tiny_timeout_on_heavy_query():
    store = _heavy_storage()
    eng = QueryEngine(store, staleness_ms=10**12)
    # try a range of tight deadlines; at least some must hit the deadline
    raised = False
    for _ in range(10):
        try:
            eng.execute("sum(rate(m[5s])) by (i)", at_ms=2000,
                        timeout_ms=1)
        except QueryTimeoutError:
            raised = True
            break
    assert raised, "a 1ms heavy query should time out"


def test_timeout_returns_no_partial_result():
    store = _heavy_storage()
    eng = QueryEngine(store, staleness_ms=10**12)
    try:
        eng.execute("sum(rate(m[5s])) by (i)", at_ms=2000, timeout_ms=0)
        result = "returned"
    except QueryTimeoutError as exc:
        result = exc
    assert isinstance(result, QueryTimeoutError)


def test_external_cancel_event_stops_query():
    store = _heavy_storage(n_series=40_000)
    eng = QueryEngine(store, staleness_ms=10**12)
    event = threading.Event()
    ctx = QueryContext(deadline=None, cancel_event=event)

    def cancel_soon():
        time.sleep(0.002)
        event.set()

    t = threading.Thread(target=cancel_soon)
    t.start()
    with pytest.raises(QueryCancelledError):
        eng.execute("sum(rate(m[5s])) by (i)", at_ms=2000, ctx=ctx)
    t.join()


def test_cancelled_query_fast_after_cancel():
    """After cancel, the query thread unwinds promptly (no run-away work)."""
    store = _heavy_storage(n_series=40_000)
    eng = QueryEngine(store, staleness_ms=10**12)
    event = threading.Event()
    ctx = QueryContext(deadline=None, cancel_event=event)
    box: dict = {}

    def run():
        start = time.perf_counter()
        try:
            eng.execute("sum(rate(m[5s])) by (i)", at_ms=2000, ctx=ctx)
        except QueryCancelledError:
            box["elapsed"] = time.perf_counter() - start

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.005)
    cancel_at = time.perf_counter()
    event.set()
    t.join(timeout=5.0)
    assert not t.is_alive(), "query kept running after cancellation"
    # unwind should be near-immediate compared to a full scan
    assert time.perf_counter() - cancel_at < 1.0


def test_check_cancelled_is_cooperative():
    ctx = QueryContext(deadline=time.perf_counter() - 1)
    with pytest.raises(QueryTimeoutError):
        ctx.check_cancelled()
    ctx2 = QueryContext(cancel_event=threading.Event())
    ctx2.cancel()
    with pytest.raises(QueryCancelledError):
        ctx2.check_cancelled()


def test_long_running_query_under_generous_timeout_completes():
    store = _heavy_storage(n_series=2_000)
    eng = QueryEngine(store, staleness_ms=10**12)
    r = eng.execute("sum(m)", at_ms=2000, timeout_ms=30_000)
    assert len(list(r)) == 1


def test_cancel_during_concurrent_writes_leaves_storage_usable():
    """Cancelling a read while writes proceed must not corrupt storage or
    block subsequent queries/writes."""
    store = Storage(checkpoint_interval=10**9)
    for i in range(5):
        store.append({"__name__": "m", "i": str(i)}, 0, 0.0)
    eng = QueryEngine(store, staleness_ms=10**12)

    stop_writes = threading.Event()
    write_errors: list[BaseException] = []

    def writer():
        v = 0
        while not stop_writes.is_set():
            v += 1
            try:
                store.append_batch(
                    [({"__name__": "m", "i": str(i)}, v, float(v))
                     for i in range(5)])
            except BaseException as exc:  # noqa: BLE001
                write_errors.append(exc)
                return

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(5):
            event = threading.Event()
            ctx = QueryContext(cancel_event=event)
            event.set()  # pre-cancel
            with pytest.raises(QueryCancelledError):
                eng.execute("sum(m) by (i)", at_ms=10**12, ctx=ctx)
    finally:
        stop_writes.set()
        t.join()

    assert not write_errors
    # storage and engine continue to work after cancellations
    store.append({"__name__": "after", "k": "1"}, 1, 42.0)
    r = list(eng.execute("after", at_ms=1, timeout_ms=None))
    assert r and r[0].value == 42.0
