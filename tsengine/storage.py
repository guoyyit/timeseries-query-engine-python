"""In-memory time-series storage with MVCC-style snapshots and WAL recovery.

Concurrency model
-----------------
* A single :class:`Storage` accepts concurrent readers and writers.
* Every committed batch of writes atomically replaces the *single*
  immutable :class:`Snapshot` object referenced by ``self._snapshot``.
  Reference assignment is atomic under the GIL, so a reader that captures
  the reference once at query start sees one consistent, immutable view for
  the whole query - it can never observe a half-applied batch or mix data
  from different commit points.
* Each series' point tuple and the snapshot's series mapping are immutable;
  a write builds brand-new tuples/dicts rather than mutating shared state.

Durability
----------
Before a batch becomes visible it is appended to the write-ahead log and
fsynced. On (re)start the storage loads the latest checkpoint (if any) and
then replays the surviving WAL records, reconstructing every acknowledged
write. Checkpoints compact the log and atomically replace prior state.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from .errors import StorageError
from .matcher import LabelMatcher
from .value import Labels, labels_from_dict

WAL_NAME = "wal.log"
CHECKPOINT_NAME = "checkpoint.json"
_NAME_LABEL = "__name__"


@dataclass(frozen=True, slots=True)
class SeriesData:
    """Immutable per-series view inside a snapshot."""
    labels: Labels
    metric_name: str  # value of the __name__ label
    points: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable point-in-time view of the whole data set."""
    series: dict[Labels, SeriesData]
    version: int = 0

    def select(self, metric_name: str | None,
               matchers: tuple[LabelMatcher, ...]) -> list[SeriesData]:
        """Return all series matching the (optional) name and matchers.

        Selection is a pure read over immutable state; ordering is by
        sorted labels so results are deterministic.
        """
        out: list[SeriesData] = []
        for sd in self.series.values():
            if metric_name is not None and sd.metric_name != metric_name:
                continue
            if all(m.matches(sd.labels) for m in matchers):
                out.append(sd)
        out.sort(key=lambda sd: sorted(sd.labels))
        return out


def _normalize_labels(labels: dict[str, str] | Labels) -> Labels:
    if isinstance(labels, frozenset):
        d = {k: v for k, v in labels}
    else:
        d = dict(labels)
    for k in d:
        if not isinstance(k, str):
            raise StorageError(f"label names must be strings, got {k!r}")
        d[k] = str(d[k])
    if _NAME_LABEL not in d:
        raise StorageError("sample labels must include '__name__'")
    return labels_from_dict(d)


class Storage:
    """Append-oriented local time-series storage with WAL persistence."""

    def __init__(self, data_dir: str | os.PathLike[str] | None = None,
                 *, checkpoint_interval: int = 1000) -> None:
        self._lock = threading.RLock()
        self._snapshot = Snapshot(series={}, version=0)
        self._commits_since_checkpoint = 0
        self._checkpoint_interval = checkpoint_interval
        self._closed = False
        if data_dir is None:
            self._dir: Path | None = None
            self._wal = None
        else:
            self._dir = Path(data_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._recover()
            wal_path = self._dir / WAL_NAME
            # long-lived append handle; closed in close(), not a context here
            self._wal = open(wal_path, "ab", buffering=0)  # noqa: SIM115

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._wal is not None:
                self._wal.flush()
                os.fsync(self._wal.fileno())
                self._wal.close()
                self._wal = None

    def __enter__(self) -> Storage:  # noqa: PYI034
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def version(self) -> int:
        return self._snapshot.version

    # -- writes ------------------------------------------------------------
    def append(self, labels: dict[str, str] | Labels, timestamp_ms: int,
               value: float, *, durable: bool = True) -> None:
        """Append one sample. Labels must include ``__name__``.

        The write is durable (WAL fsync) and visible atomically together
        with any other samples in the same ``append_batch`` call.
        """
        self.append_batch([(labels, timestamp_ms, value)], durable=durable)

    def append_batch(self, samples: list[tuple[dict[str, str] | Labels,
                                               int, float]],
                     *, durable: bool = True) -> None:
        if self._closed:
            raise StorageError("storage is closed")
        if not samples:
            return
        norm: list[tuple[Labels, int, float]] = []
        for labels, ts, value in samples:
            if not isinstance(ts, int) or isinstance(ts, bool):
                raise StorageError(f"timestamp must be int milliseconds, "
                                   f"got {ts!r}")
            norm.append((_normalize_labels(labels), ts, float(value)))

        with self._lock:
            if durable and self._wal is not None:
                self._append_wal([("c", [[dict(ls), ts, val]
                                         for ls, ts, val in norm])])
            current = self._snapshot
            new_series = dict(current.series)
            for ls, ts, val in norm:
                sd = new_series.get(ls)
                if sd is None:
                    metric_name = dict(ls)[_NAME_LABEL]
                    sd = SeriesData(labels=ls, metric_name=metric_name,
                                    points=())
                old_points = sd.points
                # binary search insertion keeps points ascending by time;
                # identical timestamps overwrite the previous value.
                lo, hi = 0, len(old_points)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if old_points[mid][0] < ts:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(old_points) and old_points[lo][0] == ts:
                    new_points = (old_points[:lo]
                                  + ((ts, val),)
                                  + old_points[lo + 1:])
                else:
                    new_points = (old_points[:lo]
                                  + ((ts, val),)
                                  + old_points[lo:])
                new_series[ls] = replace(sd, points=new_points)
            self._snapshot = Snapshot(series=new_series,
                                      version=current.version + 1)
            self._commits_since_checkpoint += 1
            if (self._dir is not None
                    and self._commits_since_checkpoint
                    >= self._checkpoint_interval):
                self._checkpoint_locked()

    # -- reads -------------------------------------------------------------
    def current_snapshot(self) -> Snapshot:
        """Return the current immutable snapshot (atomic reference read).

        Callers retain this object for the whole query; it is never mutated
        by subsequent writes.
        """
        return self._snapshot

    # -- persistence -------------------------------------------------------
    def _append_wal(self, records: list[tuple]) -> None:
        assert self._wal is not None
        buf = bytearray()
        for rec in records:
            buf.extend(json.dumps(rec, separators=(",", ":")).encode("utf-8"))
            buf.append(b"\n"[0])
        self._wal.write(bytes(buf))
        self._wal.flush()
        os.fsync(self._wal.fileno())

    def checkpoint(self) -> None:
        """Force an immediate checkpoint + WAL truncation."""
        with self._lock:
            self._checkpoint_locked()

    def _checkpoint_locked(self) -> None:
        if self._dir is None:
            self._commits_since_checkpoint = 0
            return
        snap = self._snapshot
        payload = {
            "version": snap.version,
            "series": [
                {
                    "labels": dict(sd.labels),
                    "points": [[ts, val] for ts, val in sd.points],
                }
                for sd in snap.series.values()
            ],
        }
        # atomic publish: temp file in same dir, fsync, os.replace
        fd, tmp_name = tempfile.mkstemp(prefix=".ckpt-", dir=self._dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(json.dumps(payload, separators=(",", ":")).encode())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self._dir / CHECKPOINT_NAME)
            # fsync directory entry
            dfd = os.open(self._dir, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        # truncate WAL: everything up to snap.version is now in checkpoint
        if self._wal is not None:
            self._wal.close()
        with open(self._dir / WAL_NAME, "wb") as f:
            f.flush()
            os.fsync(f.fileno())
        # long-lived append handle replacing the closed one above
        self._wal = open(self._dir / WAL_NAME, "ab", buffering=0)  # noqa: SIM115
        self._commits_since_checkpoint = 0

    def _recover(self) -> None:
        assert self._dir is not None
        series: dict[Labels, SeriesData] = {}
        version = 0
        ckpt = self._dir / CHECKPOINT_NAME
        if ckpt.exists():
            try:
                payload = json.loads(ckpt.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise StorageError(
                    f"corrupt checkpoint {ckpt}: {exc}") from exc
            version = int(payload.get("version", 0))
            for row in payload.get("series", []):
                ls = labels_from_dict(row["labels"])
                points = tuple((int(ts), float(val))
                               for ts, val in row["points"])
                series[ls] = SeriesData(
                    labels=ls, metric_name=dict(ls)[_NAME_LABEL],
                    points=points)

        wal = self._dir / WAL_NAME
        if wal.exists():
            with open(wal, "rb") as f:
                lines = f.readlines()
            replayed_records = 0
            for lineno, raw in enumerate(lines, 1):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                    self._apply_record(rec, series)
                except (json.JSONDecodeError, ValueError, KeyError,
                        TypeError) as exc:
                    # Records are fsynced as complete newline-terminated
                    # JSON lines. Therefore every successfully parsed
                    # prefix record is an acknowledged commit; a malformed
                    # line can only be the torn tail left by a crash during
                    # append. Such a tail was never acknowledged, so it is
                    # truncated rather than failing recovery. A malformed
                    # record in the MIDDLE (more data following it) is a
                    # real corruption and must not be silently ignored.
                    if lineno < len(lines):
                        raise StorageError(
                            f"corrupt WAL record at line {lineno}: {exc}"
                        ) from exc
                    break
                replayed_records += 1
                version += 1
            # rewrite a clean WAL containing only the replayed records so a
            # later checkpoint/append does not trip over the torn tail
            if replayed_records != sum(1 for ln in lines if ln.strip()):
                clean = b"".join(ln for ln in lines[:replayed_records])
                tmp = wal.with_suffix(".wal.tmp")
                tmp.write_bytes(clean)
                os.replace(tmp, wal)
        self._snapshot = Snapshot(series=series, version=version)
        self._commits_since_checkpoint = 0

    @staticmethod
    def _apply_record(rec: tuple, series: dict[Labels, SeriesData]) -> None:
        kind = rec[0]
        if kind != "c":
            raise StorageError(f"unknown WAL record kind {kind!r}")
        # one record == one atomic commit containing one or more samples
        for labels_d, ts, val in rec[1]:
            ls = labels_from_dict(labels_d)
            ts, val = int(ts), float(val)
            sd = series.get(ls)
            if sd is None:
                sd = SeriesData(labels=ls, metric_name=dict(ls)[_NAME_LABEL],
                                points=())
            points = sd.points
            lo, hi = 0, len(points)
            while lo < hi:
                mid = (lo + hi) // 2
                if points[mid][0] < ts:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(points) and points[lo][0] == ts:
                points = points[:lo] + ((ts, val),) + points[lo + 1:]
            else:
                points = points[:lo] + ((ts, val),) + points[lo:]
            series[ls] = replace(sd, points=points)
