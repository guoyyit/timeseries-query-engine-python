"""End-to-end HTTP API tests using FastAPI's TestClient with a temp dir."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TSENGINE_DATA_DIR", str(tmp_path / "data"))
    import main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_write_and_query_roundtrip(client):
    r = client.post("/api/v1/write", json={"samples": [
        {"labels": {"__name__": "m", "job": "a"}, "timestamp_ms": 1000,
         "value": 3.0},
        {"labels": {"__name__": "m", "job": "b"}, "timestamp_ms": 1000,
         "value": 7.0},
    ]})
    assert r.status_code == 200
    assert r.json()["accepted"] == 2

    r = client.get("/api/v1/query", params={
        "expr": "sum(m)", "at": 1000, "timeout": 5000})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["result"]["kind"] == "vector"
    assert body["result"]["samples"][0]["value"] == 10.0


def test_window_query_over_http(client):
    samples = [
        {"labels": {"__name__": "c"}, "timestamp_ms": ts, "value": v}
        for ts, v in [(0, 0.0), (1000, 10.0), (2000, 30.0), (3000, 35.0)]]
    client.post("/api/v1/write", json={"samples": samples})
    r = client.get("/api/v1/query",
                   params={"expr": "rate(c[3s])", "at": 3000})
    assert r.status_code == 200
    val = r.json()["result"]["samples"][0]["value"]
    assert val == 12.5


def test_syntax_error_returns_position(client):
    r = client.get("/api/v1/query", params={"expr": "m{", "at": 1000})
    assert r.status_code == 400
    body = r.json()
    assert body["error_type"] == "syntax"
    assert body["column"] > 0 and body["offset"] >= 0


def test_type_error_returns_422(client):
    r = client.get("/api/v1/query",
                   params={"expr": "rate(metric)", "at": 1000})
    assert r.status_code == 422
    assert r.json()["error_type"] == "type"


def test_timeout_returns_504(client):
    # seed many series to make a zero-timeout query always expire
    client.post("/api/v1/write", json={"samples": [
        {"labels": {"__name__": "m", "i": str(i)},
         "timestamp_ms": 1000, "value": float(i)} for i in range(2000)]})
    r = client.get("/api/v1/query", params={
        "expr": "sum(m) by (i)", "at": 1000, "timeout": 0})
    assert r.status_code == 504
    assert r.json()["error_type"] == "timeout"
