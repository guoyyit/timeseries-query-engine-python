"""WAL-based restart recovery and post-restart query consistency."""

from __future__ import annotations

import json
import os

import pytest

from tsdb.engine import QueryEngine
from tsdb.errors import StorageError
from tsdb.model import Sample
from tsdb.storage import Storage


def _populate(s: Storage) -> None:
    s.append("cpu", {"host": "a"}, [Sample(100, 1.0), Sample(200, 2.0)])
    s.append("mem", {"host": "b"}, [Sample(150, 7.0)])
    s.append_batch([
        ("cpu", {"host": "c"}, [Sample(200, 3.0)]),
        ("cpu", {"host": "d"}, [Sample(200, 4.0)]),
    ])


def test_data_survives_restart(tmp_wal):
    s1 = Storage(tmp_wal)
    _populate(s1)
    s1.close()

    s2 = Storage(tmp_wal)
    n = s2.recover()
    assert n == 5
    eng = QueryEngine(s2)
    r = eng.instant_query("sum(sum_over_time(cpu[1000s]))", at=200)
    assert r.vector[0].value == 1.0 + 2.0 + 3.0 + 4.0
    r = eng.instant_query("mem", at=200)
    assert r.vector[0].value == 7.0
    s2.close()


def test_queries_identical_before_and_after_restart(tmp_wal):
    s1 = Storage(tmp_wal)
    _populate(s1)
    eng1 = QueryEngine(s1)
    expected = eng1.instant_query('sum by (host) ({__name__=~"cpu|mem"})', at=200).to_plain()
    s1.close()

    s2 = Storage(tmp_wal)
    s2.recover()
    eng2 = QueryEngine(s2)
    after = eng2.instant_query('sum by (host) ({__name__=~"cpu|mem"})', at=200).to_plain()
    assert after == expected
    s2.close()


def test_atomic_batch_replays_atomically(tmp_wal):
    s1 = Storage(tmp_wal)
    s1.append_batch([
        ("m", {"k": "1"}, [Sample(10, 1.0)]),
        ("m", {"k": "2"}, [Sample(10, 2.0)]),
    ])
    s1.close()
    # the WAL contains a single batch record
    lines = [l for l in open(tmp_wal, "rb").read().decode().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["type"] == "batch" and len(rec["writes"]) == 2

    s2 = Storage(tmp_wal)
    s2.recover()
    eng = QueryEngine(s2)
    r = eng.instant_query("sum(m)", at=10)
    assert r.vector[0].value == 3.0
    s2.close()


def test_recover_is_idempotent_on_fresh_storage(tmp_wal):
    assert Storage(tmp_wal).recover() == 0
    assert Storage(tmp_wal).recover() == 0


def test_appends_after_recovery_continue_log(tmp_wal):
    s1 = Storage(tmp_wal)
    s1.append("m", {"k": "1"}, [Sample(1, 1.0)])
    s1.close()

    s2 = Storage(tmp_wal)
    assert s2.recover() == 1
    s2.append("m", {"k": "2"}, [Sample(2, 2.0)])
    s2.close()

    s3 = Storage(tmp_wal)
    assert s3.recover() == 2
    eng = QueryEngine(s3)
    assert eng.instant_query("sum(m)", at=2).vector[0].value == 3.0
    s3.close()


def test_checkpoint_then_restart(tmp_wal):
    s1 = Storage(tmp_wal)
    _populate(s1)
    s1.checkpoint()
    # overwrite a sample, append after checkpoint
    s1.append("cpu", {"host": "a"}, [Sample(200, 20.0)])
    eng1 = QueryEngine(s1)
    expected = eng1.instant_query("sum(sum_over_time(cpu[1000s]))", at=200).vector[0].value
    s1.close()

    s2 = Storage(tmp_wal)
    s2.recover()
    eng2 = QueryEngine(s2)
    assert eng2.instant_query("sum(sum_over_time(cpu[1000s]))", at=200).vector[0].value == expected
    # samples: a=1@100 and 20@200, c=3@200, d=4@200
    assert expected == 1.0 + 20.0 + 3.0 + 4.0
    s2.close()


def test_corrupt_wal_record_is_rejected_not_silently_ignored(tmp_wal):
    with open(tmp_wal, "wb") as f:
        f.write(b'{"type":"batch","writes":[]}\n')
        f.write(b'{"type":"append","metric":"m","labels":{},"samples":[{"t":"oops","v":1}]}\n')
    s = Storage(tmp_wal)
    with pytest.raises(StorageError, match="corrupt WAL"):
        s.recover()
    s.close()


def test_recovered_storage_supports_snapshot_consistency(tmp_wal):
    s1 = Storage(tmp_wal)
    _populate(s1)
    s1.close()
    s2 = Storage(tmp_wal)
    s2.recover()
    snap = s2.snapshot()
    s2.append("late", {"x": "1"}, [Sample(300, 99.0)])
    # pre-write snapshot unaffected
    assert snap.select("late", (), at=300) == []
    assert {lab["host"] for lab, _ in snap.matching_series("cpu", ())} == {"a", "c", "d"}
    s2.close()
