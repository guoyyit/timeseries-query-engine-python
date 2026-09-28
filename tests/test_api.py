"""HTTP API integration: write/query, error payloads, batch atomicity."""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from tsdb.api import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(wal_path=str(tmp_path / "wal.log"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_write_and_query(client):
    r = client.post("/write", json={
        "metric": "cpu",
        "labels": {"host": "a"},
        "samples": [{"t": 100, "v": 3.5}],
    })
    assert r.status_code == 200 and r.json()["written"] == 1
    r = client.get("/query", params={"expr": "cpu", "at": 100})
    body = r.json()
    assert body["kind"] == "vector"
    assert body["series"][0]["labels"]["host"] == "a"
    assert body["series"][0]["value"] == 3.5


def test_batch_write_is_atomic(client):
    r = client.post("/write/batch", json={"writes": [
        {"metric": "cpu", "labels": {"host": "a"}, "samples": [{"t": 100, "v": 1}]},
        {"metric": "cpu", "labels": {"host": "b"}, "samples": [{"t": 100, "v": 2}]},
    ]})
    assert r.status_code == 200 and r.json()["written"] == 2
    body = client.get("/query", params={"expr": "sum(cpu)", "at": 100}).json()
    assert body["series"][0]["value"] == 3.0


def test_syntax_error_payload_has_position(client):
    r = client.get("/query", params={"expr": "1 +", "at": 100})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "syntax_error"
    assert body["pos"] == 3 and body["line"] == 1 and body["col"] == 4


def test_validation_error_payload(client):
    r = client.get("/query", params={"expr": "cpu[5m]", "at": 100})
    assert r.status_code == 400
    assert r.json()["error"] == "validation_error"


def test_invalid_write_rejected(client):
    r = client.post("/write", json={
        "metric": "",
        "labels": {},
        "samples": [{"t": 1, "v": 1}],
    })
    assert r.status_code == 422


def test_timeout_returns_504(client):
    # timeout=0 -> deadline already in the past at evaluation time
    r = client.get("/query", params={"expr": "1", "at": 100, "timeout": 0.0})
    assert r.status_code == 504
    assert r.json()["error"] == "timeout"
