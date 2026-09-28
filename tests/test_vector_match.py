"""Binary operations and configurable vector matching."""

from __future__ import annotations

import pytest

from tsdb.errors import TSDBError
from tsdb.model import Sample


def _map(engine, expr, at=100, key=("job",)):
    res = engine.instant_query(expr, at=at)
    out = {}
    for it in res.vector:
        k = tuple(it.labels.get(name) for name in key)
        out[k] = it.value
    return out


def _labels(engine, expr, at=100):
    res = engine.instant_query(expr, at=at)
    return [dict(it.labels) for it in res.vector]


@pytest.fixture
def populated(engine, storage):
    storage.append("a", {"job": "j1", "host": "h1"}, [Sample(100, 10.0)])
    storage.append("a", {"job": "j1", "host": "h2"}, [Sample(100, 20.0)])
    storage.append("a", {"job": "j2", "host": "h3"}, [Sample(100, 30.0)])
    storage.append("b", {"job": "j1"}, [Sample(100, 2.0)])
    storage.append("b", {"job": "j2"}, [Sample(100, 5.0)])
    storage.append("b", {"job": "j3"}, [Sample(100, 9.0)])
    return engine, storage


# ---- scalar / scalar --------------------------------------------------------

def test_scalar_scalar_arithmetic(engine):
    assert engine.instant_query("1 + 2", 100).scalar == 3.0
    assert engine.instant_query("10 % 3", 100).scalar == 1.0
    assert engine.instant_query("2 ^ 3", 100).scalar == 8.0
    # comparison without bool on scalars that fails -> empty vector
    assert engine.instant_query("1 > 2", 100).vector == ()
    assert engine.instant_query("2 > 1", 100).scalar is True or engine.instant_query("2 > 1", 100).scalar == 1.0
    # bool comparison always yields 0/1 scalar
    assert engine.instant_query("1 > bool 2", 100).scalar == 0.0
    assert engine.instant_query("2 > bool 1", 100).scalar == 1.0


# ---- scalar / vector --------------------------------------------------------

def test_vector_scalar_preserves_labels(engine, storage):
    storage.append("m", {"k": "x"}, [Sample(100, 3.0)])
    storage.append("m", {"k": "y"}, [Sample(100, 4.0)])
    res = _map(engine, "m + 1", key=("k",))
    assert res == {("x",): 4.0, ("y",): 5.0}
    res = _map(engine, "2 * m", key=("k",))
    assert res == {("x",): 6.0, ("y",): 8.0}
    # filter comparisons keep element values; bool comparisons give 0/1
    assert _map(engine, "m > bool 3", key=("k",)) == {("x",): 0.0, ("y",): 1.0}
    kept = _map(engine, "m > 3", key=("k",))
    assert kept == {("y",): 4.0}


# ---- vector / vector --------------------------------------------------------

def test_default_matching_requires_identical_labels(populated):
    engine, _ = populated
    # full label sets differ (__name__, host) -> no matches -> empty
    assert engine.instant_query("a + b", 100).vector == ()


def test_on_join_one_to_one(populated):
    engine, _ = populated
    # job j1 has TWO hosts -> one-to-many must be rejected as ambiguous
    with pytest.raises(TSDBError):
        engine.instant_query("a / on(job) b", 100)


def test_one_to_one_unique_match(engine, storage):
    storage.append("a", {"job": "j1", "host": "h1"}, [Sample(100, 10.0)])
    storage.append("a", {"job": "j2", "host": "h2"}, [Sample(100, 30.0)])
    storage.append("b", {"job": "j1"}, [Sample(100, 2.0)])
    storage.append("b", {"job": "j2"}, [Sample(100, 5.0)])
    res = _map(engine, "a / on(job) b", key=("job", "host"))
    assert res == {("j1", "h1"): 5.0, ("j2", "h2"): 6.0}
    # output labels: join label + left-only host; no __name__ (ambiguous, dropped)
    labs = _labels(engine, "a / on(job) b")
    for l in labs:
        assert "__name__" not in l
        assert "job" in l and "host" in l


def test_group_left_many_to_one(populated):
    engine, _ = populated
    res = _map(engine, "a / on(job) group_left b", key=("job", "host"))
    assert res == {("j1", "h1"): 5.0, ("j1", "h2"): 10.0, ("j2", "h3"): 6.0}
    # unmatched left (none here) / unmatched right j3 simply don't appear
    labs = _labels(engine, "a / on(job) group_left b")
    assert all("__name__" not in l for l in labs)


def test_group_right_many_to_one(engine, storage):
    # right operand (b) is the many side under group_right
    storage.append("a", {"job": "j1"}, [Sample(100, 200.0)])
    storage.append("b", {"job": "j1", "host": "h1"}, [Sample(100, 10.0)])
    storage.append("b", {"job": "j1", "host": "h2"}, [Sample(100, 20.0)])
    res = _map(engine, "a / on(job) group_right b", key=("job", "host"))
    assert res == {("j1", "h1"): 20.0, ("j1", "h2"): 10.0}


def test_group_left_include_copies_one_side_labels(engine, storage):
    storage.append("a", {"job": "j1", "host": "h1"}, [Sample(100, 10.0)])
    storage.append("b", {"job": "j1", "zone": "z1"}, [Sample(100, 2.0)])
    labs = _labels(engine, "a / on(job) group_left(zone) b")
    assert labs == [{"job": "j1", "host": "h1", "zone": "z1"}]


def test_group_left_requires_unique_one_side(engine, storage):
    storage.append("a", {"job": "j1", "host": "h1"}, [Sample(100, 1.0)])
    storage.append("b", {"job": "j1", "x": "1"}, [Sample(100, 1.0)])
    storage.append("b", {"job": "j1", "x": "2"}, [Sample(100, 1.0)])
    with pytest.raises(TSDBError, match="one side must be unique"):
        engine.instant_query("a / on(job) group_left b", 100)


def test_ignoring_drops_labels_for_matching(populated):
    engine, _ = populated
    # ignoring(host) collapses two a-rows onto the same output key; producing
    # two identically-labelled results would be ambiguous, so it is rejected.
    with pytest.raises(TSDBError, match="same label set"):
        engine.instant_query("a / ignoring(host) group_left b", 100).vector

    # The unambiguous version: on(job) keeps the left-only host label.
    res = _map(engine, "a / on(job) group_left b", key=("job", "host"))
    assert res[("j1", "h1")] == 5.0
    assert res[("j1", "h2")] == 10.0


def test_logical_and_or(populated):
    engine, _ = populated
    res = engine.instant_query("a and on(job) b", 100).vector
    # every a row has a matching job in b
    assert len(res) == 3
    res = engine.instant_query("b and on(job) a", 100).vector
    jobs = {it.labels["job"] for it in res}
    assert jobs == {"j1", "j2"}  # j3 has no match in a
    res = engine.instant_query("b or on(job) a", 100).vector
    # b contributes j1,j2,j3 (on-job keys), a contributes nothing new
    assert {it.labels["job"] for it in res} == {"j1", "j2", "j3"}


def test_results_are_deterministically_ordered(engine, storage):
    for i, v in enumerate([3.0, 1.0, 2.0]):
        storage.append("m", {"k": chr(122 - i)}, [Sample(100, v)])
    first = _labels(engine, "m + 0", )
    second = _labels(engine, "m + 0")
    assert first == second
    assert [l["k"] for l in first] == sorted(l["k"] for l in first)
