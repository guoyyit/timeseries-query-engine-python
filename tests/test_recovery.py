"""Requirement 8: data must survive process restart, and queries over the
recovered state must yield the same results as before the restart."""

from __future__ import annotations

import json

import pytest

from tsengine.engine import QueryEngine
from tsengine.errors import StorageError
from tsengine.storage import CHECKPOINT_NAME, WAL_NAME, Storage
from tsengine.value import Vector


def _populate(store: Storage):
    store.append_batch([
        ({"__name__": "cpu", "host": "h1", "region": "us"}, 1000, 0.1),
        ({"__name__": "cpu", "host": "h1", "region": "us"}, 2000, 0.2),
        ({"__name__": "cpu", "host": "h1", "region": "us"}, 3000, 0.3),
        ({"__name__": "cpu", "host": "h2", "region": "eu"}, 1000, 0.9),
        ({"__name__": "cpu", "host": "h2", "region": "eu"}, 2000, 0.8),
        ({"__name__": "mem", "host": "h1", "region": "us"}, 1000, 512.0),
    ])


def _query_state(eng: QueryEngine):
    """Capture query results used to compare pre/post restart."""
    cpu = eng.execute("cpu", at_ms=10_000, timeout_ms=None)
    rate = eng.execute("rate(cpu[3s])", at_ms=3000, timeout_ms=None)
    total = eng.execute("sum(cpu) by (host)", at_ms=10_000, timeout_ms=None)
    def snap(v):
        assert isinstance(v, Vector)
        return sorted((tuple(sorted(s.labels)), round(s.value, 9))
                      for s in v)
    return snap(cpu), snap(rate), snap(total)


def test_wal_replay_restores_all_writes(tmp_path):
    d = tmp_path / "data"
    store = Storage(d)
    _populate(store)
    store.close()

    # simulate process restart: a brand-new Storage on the same directory
    recovered = Storage(d)
    assert recovered.version == 1  # one atomic batch commit replayed
    eng = QueryEngine(recovered, staleness_ms=10**12)
    cpu = list(eng.execute("cpu", at_ms=10_000))
    assert len(cpu) == 2
    h1 = [s for s in cpu if dict(s.labels)["host"] == "h1"][0]
    h2 = [s for s in cpu if dict(s.labels)["host"] == "h2"][0]
    assert h1.value == 0.3 and h2.value == 0.8
    mem = list(eng.execute("mem", at_ms=10_000))
    assert mem[0].value == 512.0
    recovered.close()


def test_query_results_identical_before_and_after_restart(tmp_path):
    d = tmp_path / "data"
    store = Storage(d)
    _populate(store)
    eng = QueryEngine(store, staleness_ms=10**12)
    before = _query_state(eng)
    store.close()

    recovered = Storage(d)
    eng2 = QueryEngine(recovered, staleness_ms=10**12)
    after = _query_state(eng2)
    assert before == after
    recovered.close()


def test_checkpoint_then_wal_replay(tmp_path):
    d = tmp_path / "data"
    store = Storage(d, checkpoint_interval=10**9)
    _populate(store)
    store.checkpoint()
    # writes after checkpoint live only in the WAL
    store.append({"__name__": "cpu", "host": "h3", "region": "ap"},
                 1000, 0.5)
    store.append({"__name__": "net", "host": "h1"}, 1000, 42.0)
    store.close()

    recovered = Storage(d)
    eng = QueryEngine(recovered, staleness_ms=10**12)
    hosts = {dict(s.labels)["host"]
             for s in eng.execute("cpu", at_ms=10_000)}
    assert hosts == {"h1", "h2", "h3"}
    assert len(list(eng.execute("net", at_ms=10_000))) == 1
    # WAL beyond checkpoint was replayed and version is continuous
    assert recovered.version == 3
    recovered.close()


def test_multiple_restarts_are_stable(tmp_path):
    d = tmp_path / "data"
    store = Storage(d)
    _populate(store)
    store.close()
    expected = None
    for _ in range(3):
        s = Storage(d)
        eng = QueryEngine(s, staleness_ms=10**12)
        state = _query_state(eng)
        if expected is None:
            expected = state
        else:
            assert state == expected
        s.close()


def test_automatic_checkpoint_compacts_wal(tmp_path):
    d = tmp_path / "data"
    store = Storage(d, checkpoint_interval=3)
    for i in range(7):
        store.append({"__name__": "m", "i": str(i)}, 1000, float(i))
    store.close()
    wal = (d / WAL_NAME).read_bytes()
    # checkpoint fires at commits 3 and 6; after commit 6 only commit 7
    # (series i=6) remains in the WAL
    assert wal.count(b"\n") == 1
    ckpt = json.loads((d / CHECKPOINT_NAME).read_text())
    assert len(ckpt["series"]) == 6  # i=0..5 compacted
    recovered = Storage(d)
    # recovery merges checkpoint (6 series) + WAL (1 series)
    assert len(recovered.current_snapshot().series) == 7
    assert recovered.version == 7
    recovered.close()


def test_torn_final_wal_record_keeps_committed_prefix(tmp_path):
    """A crash mid-append may leave a partial trailing line. The records
    before it were fsynced and acknowledged, so recovery keeps them and
    discards only the torn tail."""
    d = tmp_path / "data"
    store = Storage(d)
    _populate(store)
    store.close()
    good = json.dumps(
        ("c", [[{"__name__": "late", "k": "1"}, 9000, 7.0]]))
    with open(d / WAL_NAME, "ab") as f:
        f.write((good + "\n").encode())
        f.write(b'["c", [[{"__name__": "brok')  # no newline, truncated
    recovered = Storage(d)
    eng = QueryEngine(recovered, staleness_ms=10**12)
    # all acknowledged data survived
    assert len(list(eng.execute("cpu", at_ms=10_000))) == 2
    assert len(list(eng.execute("late", at_ms=10_000))) == 1
    recovered.close()
    # after recovery the WAL is clean and reopening is stable
    again = Storage(d)
    assert len(list(QueryEngine(again, staleness_ms=10**12)
                    .execute("late", at_ms=10_000))) == 1
    again.close()


def test_corrupt_middle_wal_record_is_hard_error(tmp_path):
    """Corruption in the MIDDLE of the WAL (valid records after it) cannot
    be interpreted as a torn tail and must fail loudly."""
    d = tmp_path / "data"
    store = Storage(d)
    _populate(store)
    store.close()
    good = json.dumps(
        ("c", [[{"__name__": "late", "k": "1"}, 9000, 7.0]]))
    with open(d / WAL_NAME, "ab") as f:
        f.write(b"{not valid json\n")          # corrupt middle line
        f.write((good + "\n").encode())        # valid record after it
    with pytest.raises(StorageError):
        Storage(d)


def test_no_data_dir_is_purely_in_memory(tmp_path):
    """Without a directory the engine still works, keeping recovery opt-in."""
    store = Storage()
    _populate(store)
    eng = QueryEngine(store, staleness_ms=10**12)
    assert len(list(eng.execute("cpu", at_ms=10_000))) == 2


def test_recovered_points_remain_time_ordered(tmp_path):
    d = tmp_path / "data"
    store = Storage(d)
    # out-of-order append order
    for ts, v in [(3000, 3.0), (1000, 1.0), (2000, 2.0), (2000, 22.0)]:
        store.append({"__name__": "m"}, ts, v)
    store.close()
    rec = Storage(d)
    sd = list(rec.current_snapshot().series.values())[0]
    assert sd.points == ((1000, 1.0), (2000, 22.0), (3000, 3.0))
    rec.close()


def test_rate_query_after_restart_matches(tmp_path):
    d = tmp_path / "data"
    store = Storage(d)
    labels = {"__name__": "counter"}
    for ts, v in [(0, 0.0), (1000, 10.0), (2000, 30.0), (3000, 35.0)]:
        store.append(labels, ts, v)
    store.close()

    rec = Storage(d)
    eng = QueryEngine(rec, staleness_ms=10**12)
    r = list(eng.execute("rate(counter[3s])", at_ms=3000))
    assert len(r) == 1
    # half-open window (0,3s] excludes t=0: points 10,30,35
    # rate = (35 - 10) / (3s - 1s) = 12.5
    assert r[0].value == pytest.approx(12.5, rel=1e-6)
    rec.close()
