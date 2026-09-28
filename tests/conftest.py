"""Shared fixtures: fresh storage + engine per test, temp WAL."""

from __future__ import annotations

import os
import tempfile

import pytest

from tsdb.engine import QueryEngine
from tsdb.model import Sample
from tsdb.storage import Storage


@pytest.fixture
def tmp_wal(tmp_path):
    return str(tmp_path / "wal.log")


@pytest.fixture
def storage(tmp_wal):
    s = Storage(tmp_wal)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def engine(storage):
    return QueryEngine(storage)


@pytest.fixture
def make_engine(tmp_path):
    """Factory allowing multiple engines over the same WAL path (restart)."""

    def _make(name: str = "wal.log"):
        path = str(tmp_path / name)
        s = Storage(path)
        return s, QueryEngine(s), path

    return _make
