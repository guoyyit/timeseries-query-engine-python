"""In-memory time-series storage with immutable snapshots and WAL recovery.

Concurrency model
-----------------
- ``append`` serialises on ``_lock``. Each append creates new immutable
  tuples/series entries; existing samples are never mutated.
- ``snapshot`` copies the current mapping under the lock and returns an
  immutable :class:`Snapshot`. A query therefore sees exactly one committed
  state for its whole duration, regardless of concurrent writes.

Persistence
-----------
Every successful append is first written as one line-based JSON record to a
write-ahead log (``append<N>`` records) and flushed + fsynced. On startup
``recover`` replays the log; ``checkpoint`` truncates it.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass

from .errors import StorageError
from .model import LabelMatcher, LabelSet, MatchType, NAME_LABEL, Sample


_KEY_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


def validate_labels(metric: str, labels: LabelSet) -> None:
    if not isinstance(metric, str) or not metric:
        raise StorageError("metric name must be a non-empty string")
    if not _KEY_RE.match(metric):
        raise StorageError(f"invalid metric name {metric!r}")
    if NAME_LABEL in labels:
        raise StorageError(f"label {NAME_LABEL!r} is reserved")
    for k, v in labels.items():
        if not isinstance(k, str) or not _KEY_RE.match(k):
            raise StorageError(f"invalid label name {k!r}")
        if not isinstance(v, str):
            raise StorageError(f"label {k!r} value must be a string")


def _series_key(metric: str, labels: LabelSet) -> tuple[tuple[str, str], ...]:
    items = [(NAME_LABEL, metric), *labels.items()]
    return tuple(sorted(items))


def _full_labels(metric: str, labels: LabelSet) -> LabelSet:
    out: LabelSet = {NAME_LABEL: metric}
    out.update(labels)
    return out


class Snapshot:
    """Immutable point-in-time view of the database contents."""

    __slots__ = ("_data", "_version")

    def __init__(
        self,
        data: tuple[tuple[tuple[tuple[str, str], ...], LabelSet, tuple[Sample, ...]], ...],
        version: int,
    ) -> None:
        # data is a tuple of (key, full_labels, immutable samples tuple)
        self._data = data
        self._version = version

    @property
    def version(self) -> int:
        return self._version

    def _match(self, labels: LabelSet, name: str | None, matchers: tuple[LabelMatcher, ...]) -> bool:
        if name is not None and labels.get(NAME_LABEL) != name:
            return False
        for m in matchers:
            if m.name == NAME_LABEL and name is not None:
                return False  # conflicting selector is rejected at parser layer; defensive
            actual = labels.get(m.name)
            if not _match_one(actual, m):
                return False
        return True

    def matching_series(
        self, name: str | None, matchers: tuple[LabelMatcher, ...]
    ) -> tuple[tuple[LabelSet, tuple[Sample, ...]], ...]:
        out: list[tuple[LabelSet, tuple[Sample, ...]]] = []
        for _key, labels, samples in self._data:
            if self._match(labels, name, matchers):
                out.append((labels, samples))
        out.sort(key=lambda item: tuple(sorted(item[0].items())))
        return tuple(out)

    def select(
        self,
        name: str | None,
        matchers: tuple[LabelMatcher, ...],
        at: int,
        lookback_delta: int = 300,
    ) -> list[tuple[LabelSet, Sample | None]]:
        """Instant selection: newest sample with t in [at-delta, at].

        ``None`` is returned (never a synthesized zero) when the series has
        no sample in the staleness window.
        """
        result: list[tuple[LabelSet, Sample | None]] = []
        for labels, samples in self.matching_series(name, matchers):
            found: Sample | None = None
            # samples are stored sorted by timestamp; scan from the end
            for s in reversed(samples):
                if s.t > at:
                    continue
                if s.t >= at - lookback_delta:
                    found = s
                break
            result.append((labels, found))
        return result

    def select_range(
        self,
        name: str | None,
        matchers: tuple[LabelMatcher, ...],
        start: int,
        end: int,
    ) -> list[tuple[LabelSet, tuple[Sample, ...]]]:
        """Range selection: samples within inclusive [start, end]."""
        result: list[tuple[LabelSet, tuple[Sample, ...]]] = []
        for labels, samples in self.matching_series(name, matchers):
            window = tuple(s for s in samples if start <= s.t <= end)
            # Even series with zero samples in the window are returned;
            # functions decide how to treat the empty window.
            result.append((labels, window))
        return result


def _match_one(actual: str | None, m: LabelMatcher) -> bool:
    import re

    if m.op is MatchType.EQ:
        return actual == m.value
    if m.op is MatchType.NEQ:
        # absent label: != "x" matches (label does not equal x)
        return actual != m.value
    if m.op is MatchType.RE:
        if actual is None:
            return False
        return re.fullmatch(m.value, actual) is not None
    # NRE: absent label matches ("does not match" is vacuously true)
    if actual is None:
        return True
    return re.fullmatch(m.value, actual) is None


class Storage:
    def __init__(self, wal_path: str | None = None) -> None:
        self._lock = threading.RLock()
        # key -> [full_labels, samples_tuple]; mapping itself is replaced on
        # each append so snapshots never observe partial mutation.
        self._series: dict[tuple[tuple[str, str], ...], list] = {}
        self._version = 0
        self._wal_path = wal_path
        self._wal_file = None
        if wal_path is not None:
            d = os.path.dirname(os.path.abspath(wal_path))
            os.makedirs(d, exist_ok=True)
            self._wal_file = open(wal_path, "ab", buffering=0)

    # lifecycle -------------------------------------------------------------
    def recover(self) -> int:
        """Replay the WAL. Returns number of recovered samples."""
        if self._wal_path is None or not os.path.exists(self._wal_path):
            return 0
        count = 0
        with self._lock, open(self._wal_path, "rb") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rtype = rec.get("type")
                    if rtype == "batch":
                        entries = [
                            (
                                w["metric"],
                                dict(w["labels"]),
                                [Sample(int(s["t"]), float(s["v"])) for s in w["samples"]],
                            )
                            for w in rec["writes"]
                        ]
                    elif rtype == "append":
                        entries = [
                            (
                                rec["metric"],
                                dict(rec["labels"]),
                                [Sample(int(s["t"]), float(s["v"])) for s in rec["samples"]],
                            )
                        ]
                    else:
                        raise ValueError("unknown record type")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise StorageError(
                        f"corrupt WAL at {self._wal_path}:{lineno}: {exc}"
                    ) from None
                for metric, labels, samples in entries:
                    self._apply_append(metric, labels, samples)
                    count += len(samples)
                self._version += 1
        return count

    def close(self) -> None:
        with self._lock:
            if self._wal_file is not None:
                self._wal_file.flush()
                os.fsync(self._wal_file.fileno())
                self._wal_file.close()
                self._wal_file = None

    # writes ----------------------------------------------------------------
    def append(self, metric: str, labels: LabelSet, samples: list[Sample]) -> int:
        """Append a single series' samples (one atomic commit)."""
        return self.append_batch([(metric, dict(labels), samples)])

    def append_batch(self, writes: list[tuple[str, LabelSet, list[Sample]]]) -> int:
        """Atomically append samples for multiple series.

        All series in ``writes`` become visible in a single version step with
        a single WAL record, so readers observe either the whole batch or
        none of it. Duplicate timestamps overwrite deterministically
        (last entry wins).
        """
        normalized: list[tuple[str, LabelSet, list[Sample]]] = []
        total = 0
        for metric, labels, samples in writes:
            validate_labels(metric, dict(labels))
            norm: dict[int, Sample] = {}
            for s in samples:
                if not isinstance(s, Sample):
                    raise StorageError("append expects Sample objects")
                norm[s.t] = s
            ordered = sorted(norm.values(), key=lambda s: s.t)
            normalized.append((metric, dict(labels), ordered))
            total += len(ordered)

        with self._lock:
            if self._wal_file is not None:
                record = json.dumps(
                    {
                        "type": "batch",
                        "writes": [
                            {
                                "metric": metric,
                                "labels": labels,
                                "samples": [{"t": s.t, "v": s.v} for s in ordered],
                            }
                            for metric, labels, ordered in normalized
                        ],
                    },
                    separators=(",", ":"),
                ).encode() + b"\n"
                self._wal_file.write(record)
                self._wal_file.flush()
                os.fsync(self._wal_file.fileno())
            for metric, labels, ordered in normalized:
                self._apply_append(metric, labels, ordered)
            self._version += 1
            return total

    def _apply_append(self, metric: str, labels: LabelSet, ordered: list[Sample]) -> None:
        key = _series_key(metric, labels)
        entry = self._series.get(key)
        if entry is None:
            full = _full_labels(metric, labels)
            merged = tuple(ordered)
            self._series[key] = [full, merged]
            return
        full, existing = entry
        merged_map: dict[int, Sample] = {s.t: s for s in existing}
        for s in ordered:
            merged_map[s.t] = s
        entry[1] = tuple(sorted(merged_map.values(), key=lambda s: s.t))

    def checkpoint(self) -> None:
        """Persist current state and truncate the WAL."""
        if self._wal_path is None:
            return
        with self._lock:
            if self._wal_file is not None:
                self._wal_file.close()
            with open(self._wal_path, "wb") as f:
                for _key, entry in sorted(self._series.items()):
                    full, samples = entry
                    metric = full[NAME_LABEL]
                    labels = {k: v for k, v in full.items() if k != NAME_LABEL}
                    rec = {
                        "type": "append",
                        "metric": metric,
                        "labels": labels,
                        "samples": [{"t": s.t, "v": s.v} for s in samples],
                    }
                    f.write(json.dumps(rec, separators=(",", ":")).encode() + b"\n")
                f.flush()
                os.fsync(f.fileno())
            self._wal_file = open(self._wal_path, "ab", buffering=0)

    # reads -----------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        with self._lock:
            data = tuple(
                (key, entry[0], entry[1])
                for key, entry in sorted(self._series.items())
            )
            return Snapshot(data, self._version)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version
