"""Concurrent query/write consistency, timeout and cancellation."""

from __future__ import annotations

import threading
import time

import pytest

from tsdb.engine import CancellationToken, QueryEngine
from tsdb.errors import QueryCancelledError, QueryTimeoutError
from tsdb.model import LabelMatcher, MatchType, Sample


def test_query_observes_single_consistent_snapshot(storage, engine):
    """Every query result must be explainable as one of the two states."""
    # initial state: cpu host=a = 10 for all timestamps
    storage.append("cpu", {"host": "a"}, [Sample(t, 10.0) for t in range(0, 601, 60)])

    stop = threading.Event()
    results: list[dict] = []
    errors: list[Exception] = []

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            # both series change in ONE atomic batch, so a snapshot can only
            # observe the previous or the new state, never an intermediate one
            storage.append_batch([
                ("cpu", {"host": "a"}, [Sample(600, 10.0 + i)]),
                ("cpu", {"host": "b"}, [Sample(600, 10.0 + i)]),
            ])
            time.sleep(0.0005)

    def reader():
        while not stop.is_set():
            try:
                r = engine.instant_query(
                    'sum by (host) (cpu{host=~"a|b"})', at=600
                )
                vals = {it.labels.get("host"): it.value for it in r.vector}
                results.append(vals)
            except Exception as exc:  # pragma: no cover - queries shouldn't fail
                errors.append(exc)

    tw = threading.Thread(target=writer)
    readers = [threading.Thread(target=reader) for _ in range(4)]
    tw.start()
    for t in readers:
        t.start()
    time.sleep(0.5)
    stop.set()
    tw.join()
    for t in readers:
        t.join()

    assert not errors, errors
    assert results
    # Result must be one of: {a only} before host b first exists, or both
    # hosts with EQUAL values (writes to a and b are separate appends but the
    # snapshot must never expose an intermediate single-host-changed state).
    for vals in results:
        if "b" in vals:
            assert vals.get("a") == vals.get("b"), vals
        else:
            assert set(vals) == {"a"}, vals


def test_snapshot_never_mixes_versions(storage):
    """A snapshot taken before a write is unaffected by later writes."""
    storage.append("m", {"k": "1"}, [Sample(100, 1.0)])
    snap = storage.snapshot()
    for i in range(50):
        storage.append("m", {"k": str(i)}, [Sample(100, float(i))])
    assert snap.version == 1
    rows = snap.select("m", (), at=100)
    assert len(rows) == 1


def test_timeout_raises_and_returns_no_partial_result(engine, storage):
    for i in range(5000):
        storage.append("m", {"k": f"k{i}"}, [Sample(100, float(i))])
    with pytest.raises(QueryTimeoutError):
        engine.instant_query("sum(m)", at=100, timeout=0.0)
    # a still-positive but tight deadline on a smaller set works
    r = engine.instant_query("sum(m)", at=100, timeout=10.0)
    assert r.kind == "vector"


def test_cancellation_token_stops_computation(engine, storage):
    for i in range(20000):
        storage.append("m", {"k": f"k{i}"}, [Sample(100, float(i))])
    token = CancellationToken()
    token.cancel("test cancel")
    with pytest.raises(QueryCancelledError):
        engine.instant_query("sum(m)", at=100, token=token)


def test_cancel_during_long_query_releases(engine, storage):
    for i in range(20000):
        storage.append("m", {"k": f"k{i}"}, [Sample(t, float(i)) for t in range(0, 600, 30)])

    token = CancellationToken()
    box: dict = {}

    def runner():
        start = time.monotonic()
        try:
            engine.instant_query("sum(rate(m[500s]))", at=600, token=token)
        except QueryCancelledError:
            box["elapsed"] = time.monotonic() - start

    t = threading.Thread(target=runner)
    t.start()
    time.sleep(0.02)
    token.cancel()
    t.join(timeout=5.0)
    assert not t.is_alive(), "query thread kept running after cancel"
    assert box["elapsed"] < 2.0


def test_parallel_writes_and_queries_preserve_sample_integrity(storage, engine):
    """Heavy mixed load: every sample is either fully visible or not."""
    def writer(n):
        for i in range(100):
            storage.append(
                "m",
                {"w": str(n), "i": str(i)},
                [Sample(i, float(n * 1000 + i))],
            )

    def reader():
        end = time.time() + 0.4
        while time.time() < end:
            r = engine.instant_query("count by (w) (m)", at=99)
            for it in r.vector:
                # count for any observed writer must be between 1 and 100;
                # partial appends cannot create duplicates/missing rows
                assert 1 <= it.value <= 100

    ws = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    rs = [threading.Thread(target=reader) for _ in range(3)]
    for t in ws + rs:
        t.start()
    for t in ws:
        t.join()
    for t in rs:
        t.join()
    # after all writers finish: exactly 400 distinct series
    r = engine.instant_query("count(m)", at=99)
    assert r.vector[0].value == 400.0


def test_concurrent_batch_writes_never_split(storage):
    """Each batch must appear indivisibly across many concurrent snapshots."""
    n_writers = 8
    per = 100
    seen_versions: set[int] = set()
    stop = threading.Event()

    def batch_writer(w):
        for i in range(per):
            storage.append_batch([
                ("m", {"w": str(w), "pair": "x", "i": str(i)}, [Sample(1, 1.0)]),
                ("m", {"w": str(w), "pair": "y", "i": str(i)}, [Sample(1, 1.0)]),
            ])

    def observer():
        while not stop.is_set():
            snap = storage.snapshot()
            seen_versions.add(snap.version)
            for w in range(n_writers):
                matcher = (LabelMatcher("w", MatchType.EQ, str(w)),)
                rows = snap.matching_series("m", matcher)
                # every observed writer appears with 0 or an even number of
                # rows (x/y pairs), never a torn single half of the batch
                assert len(rows) % 2 == 0

    threads = [threading.Thread(target=batch_writer, args=(w,)) for w in range(n_writers)]
    obs = [threading.Thread(target=observer) for _ in range(2)]
    for t in threads + obs:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    for t in obs:
        t.join()
    assert storage.version == n_writers * per
