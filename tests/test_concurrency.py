"""Requirement 6: concurrent reads/writes must each observe one consistent
snapshot - never a mixture of two commit points or a half-applied batch."""

from __future__ import annotations

import threading

from tsengine.engine import QueryEngine
from tsengine.storage import Storage


def _series_values(result):
    return {tuple(sorted(s.labels)): s.value for s in result}


def test_snapshot_immutable_against_later_writes():
    """A pinned snapshot stays identical after subsequent commits."""
    store = Storage()
    for i in range(5):
        store.append({"__name__": "m", "i": str(i)}, 1000, float(i))
    snap = store.current_snapshot()
    before = {ls: len(sd.points) for ls, sd in snap.series.items()}
    for i in range(5, 20):
        store.append({"__name__": "m", "i": str(i)}, 1000, float(i))
        store.append({"__name__": "m", "i": "0"}, 2000 + i, float(i))
    after = {ls: len(sd.points) for ls, sd in snap.series.items()}
    assert before == after
    # but the live storage moved on
    assert store.current_snapshot().version > snap.version


def test_batch_is_atomic():
    """All samples in a batch become visible together, never partially."""
    store = Storage()
    # pre-populate one point per series so selectors match
    for k in range(10):
        store.append({"__name__": "m", "k": str(k)}, 0, 0.0)

    seen_versions: list[int] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            snap = store.current_snapshot()
            # Within one snapshot every series must have the SAME number of
            # points because each batch writes to every series. A torn batch
            # would expose differing counts.
            counts = {len(sd.points)
                      for sd in snap.series.values()
                      if sd.metric_name == "m"}
            assert len(counts) == 1, f"torn batch: {counts}"
            seen_versions.append(snap.version)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for v in range(1, 51):
            batch = [({"__name__": "m", "k": str(k)}, v, float(v))
                     for k in range(10)]
            store.append_batch(batch)
    finally:
        stop.set()
        t.join()
    assert seen_versions  # reader actually ran


def test_concurrent_queries_see_consistent_aggregates():
    """sum computed by a query must equal some committed prefix total."""
    store = Storage()
    for k in range(20):
        store.append({"__name__": "m", "k": str(k)}, 0, 0.0)
    eng = QueryEngine(store, staleness_ms=10**12)

    prefix_totals: set[float] = {0.0}
    running_total = 0.0

    results: list[float] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def query_loop():
        try:
            while not stop.is_set():
                r = eng.execute("sum(m)", at_ms=10**12)
                samples = list(r)
                # a full aggregation yields exactly one sample, no labels
                assert len(samples) == 1
                assert dict(samples[0].labels) == {}
                results.append(samples[0].value)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=query_loop) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for v in range(1, 41):
            batch = [({"__name__": "m", "k": str(k)}, 10**12, 1.0)
                     for k in range(20)]
            store.append_batch(batch)
            running_total += 20.0
            prefix_totals.add(running_total)
    finally:
        stop.set()
        for t in threads:
            t.join()

    assert not errors, errors
    assert results
    # every observed total must correspond to a fully committed prefix
    assert set(results) <= prefix_totals, (
        set(results) - prefix_totals)


def test_query_pins_snapshot_for_whole_evaluation():
    """A range query started before writes must not include the new writes."""
    store = Storage()
    labels = {"__name__": "m"}
    for ts in range(0, 100_000, 10_000):
        store.append(labels, ts, 1.0)
    eng = QueryEngine(store, staleness_ms=10**12)

    # grab snapshot then write before evaluating
    snap = store.current_snapshot()
    from tsengine.engine import QueryContext
    ctx = QueryContext(cancel_event=threading.Event())
    store.append_batch([(labels, 5_000, 99.0),
                        (labels, 15_000, 99.0)])
    r = eng._eval_matrix(
        __import__("tsengine.parser", fromlist=["parse"]).parse(
            "m[100s]"), snap, 90_000, ctx)
    rows = list(r)
    assert len(rows) == 1
    assert all(v == 1.0 for _, v in rows[0].points)
    assert 99.0 not in [v for _, v in rows[0].points]


def test_many_readers_and_writers_thread_safety():
    """Smoke stress: no exceptions, counts stay consistent."""
    store = Storage(checkpoint_interval=10**9)
    eng = QueryEngine(store, staleness_ms=10**12, default_timeout_ms=10_000)
    n_series = 8
    for k in range(n_series):
        store.append({"__name__": "m", "k": str(k)}, 0, 0.0)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer(seed):
        v = 0
        while not stop.is_set():
            v += 1
            batch = [({"__name__": "m", "k": str(k), "w": str(seed)},
                      v, float(seed + k)) for k in range(n_series)]
            store.append_batch(batch)

    def reader():
        while not stop.is_set():
            try:
                snap = store.current_snapshot()
                # each writer writes one point to EACH series per commit;
                # within a snapshot, per-writer point counts must be uniform
                per_writer: dict[str, set[int]] = {}
                for sd in snap.series.values():
                    w = dict(sd.labels).get("w")
                    if w is None:
                        continue
                    per_writer.setdefault(w, set()).add(len(sd.points))
                for w, counts in per_writer.items():
                    assert len(counts) == 1, (w, counts)
                eng.execute("sum(m) by (w)", at_ms=10**12)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return

    threads = [threading.Thread(target=writer, args=(i,))
               for i in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    import time
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
