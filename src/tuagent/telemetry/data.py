import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from .contracts import PERIOD_MS, PROFILE, RETENTION_MS, SCHEMA, SourceArgs, TelemetryError
from .mysql import Row, TelemetryReader


BEIJING = timezone(timedelta(hours=8))
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def beijing(ms: int) -> str:
    return (EPOCH + timedelta(milliseconds=ms)).astimezone(BEIJING).isoformat(
        timespec="milliseconds"
    )


def time_range(start_ms: int, end_ms: int) -> Row:
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "start_beijing": beijing(start_ms), "end_beijing": beijing(end_ms),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        aware = value.replace(tzinfo=BEIJING) if value.tzinfo is None else value
        return aware.astimezone(BEIJING).isoformat(timespec="milliseconds")
    raise TypeError(f"Unsupported telemetry value type: {type(value).__name__}")


def load_metadata(
    db: TelemetryReader, args: SourceArgs, fields: tuple[str, ...], expected_hash: str | None
) -> tuple[dict[str, Row], Row]:
    if args.profile_version != PROFILE:
        raise TelemetryError("version_error", f"Unsupported profile {args.profile_version!r}; expected {PROFILE}")
    rows = db.metadata(args.table_name, args.profile_version)
    by_field = {row["field_name"]: row for row in rows}
    missing = [name for name in fields if name not in by_field]
    if missing:
        raise TelemetryError(
            "version_error", f"Missing definitions in {args.table_name}/{PROFILE}: {', '.join(missing)}"
        )
    hashes = [row.get("source_sha256") for row in rows]
    if (
        len(rows) > 32 or len(by_field) != len(rows)
        or not all(isinstance(value, str) and re.fullmatch("[0-9a-fA-F]{64}", value) for value in hashes)
        or len({value.lower() for value in hashes}) != 1
    ):
        raise TelemetryError(
            "version_error", f"Invalid or inconsistent source_sha256/definitions for {args.table_name}/{PROFILE}"
        )
    source_hash = hashes[0].lower()
    if expected_hash and expected_hash.lower() != source_hash:
        raise TelemetryError(
            "version_error", f"source_sha256 differs from the deployment expectation for {args.table_name}/{PROFILE}"
        )

    metadata = {}
    for name in fields:
        definition = dict(by_field[name])
        try:
            for key in ("value_conventions", "invalid_values"):
                raw = definition[key]
                if isinstance(raw, (str, bytes)):
                    definition[key] = json.loads(raw)
            if not isinstance(definition["invalid_values"], list):
                raise ValueError
            if not all(math.isfinite(float(definition[key])) for key in ("scale_factor", "offset_value")):
                raise ValueError
            if definition["verification_status"] not in ("verified_simulator", "simulation_assumption"):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            raise TelemetryError(
                "version_error", f"Invalid field definition for {args.table_name}/{PROFILE}/{name}"
            ) from None
        metadata[name] = definition

    return metadata, {
        "schema_name": SCHEMA, "table_name": args.table_name, "profile_version": PROFILE,
        "source_sha256": source_hash,
        "model_verification": ("matched_deployment_hash" if expected_hash else "dictionary_consistency_only"),
        "model_verification_note": (
            "Dictionary hash matches the configured deployment hash." if expected_hash
            else "Dictionary consistency checked; the model file was not independently verified."
        ),
    }


def valid_value(name: str, value: Any, definition: Row) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, Decimal)) and not math.isfinite(value):
        return False
    if name == "air_intake_temp" and value == -200:
        return False
    if name in ("fault_code", "alarm_code") and (not isinstance(value, str) or not value.strip()):
        return False
    return not any(
        value == invalid and isinstance(value, bool) == isinstance(invalid, bool)
        for invalid in definition["invalid_values"]
    )


def sample_key(row: Row) -> tuple[int, int]:
    return int(row["timestamp"]), row["id"]


def sample_time(row: Row) -> Row:
    ms, row_id = sample_key(row)
    return {"timestamp_ms": ms, "time_beijing": beijing(ms), "id": row_id}


@dataclass(frozen=True, slots=True)
class Window:
    start_ms: int
    end_ms: int
    read_ms: int

    @property
    def cutoff_ms(self) -> int:
        return max(0, self.read_ms - RETENTION_MS)

    @property
    def available_start(self) -> int:
        return min(self.end_ms, max(self.start_ms, self.cutoff_ms))

    @property
    def available_end(self) -> int:
        return max(self.available_start, min(self.end_ms, self.read_ms + 1))

    def unavailable(self) -> list[Row]:
        result = []
        if self.start_ms < self.cutoff_ms:
            result.append({
                **time_range(self.start_ms, min(self.end_ms, self.cutoff_ms)),
                "reason": "outside_retention",
            })
        if self.end_ms > self.read_ms + 1:
            result.append({
                **time_range(max(self.start_ms, self.read_ms + 1), self.end_ms),
                "reason": "after_read_time",
            })
        return result


class Coverage:
    """Track full-window quality on the ordered stream, independent of pagination."""

    def __init__(self, fields: tuple[str, ...], *, gap_limit: int = 10):
        self.points = self.unique_timestamps = self.irregular_intervals = 0
        self.first = self.last = None
        self.valid_points = dict.fromkeys(fields, 0)
        self._gap_limit = gap_limit
        self._gaps: list[Row] = []
        self._gap_count = self._missing = 0

    def add(self, ms: int, validity: dict[str, bool]) -> None:
        self.points += 1
        for name, valid in validity.items():
            self.valid_points[name] += valid
        if self.first is None:
            self.first = ms
        if self.last is None or self.last != ms:
            self.unique_timestamps += 1
        if self.last is not None and ms - self.last != PERIOD_MS:
            self.irregular_intervals += 1
            missing = (ms - self.last - 1) // PERIOD_MS
            if missing > 0:
                self._gap_count += 1
                self._missing += missing
                if len(self._gaps) < self._gap_limit:
                    self._gaps.append(self._gap("internal", self.last + PERIOD_MS, ms, missing))
        self.last = ms

    @staticmethod
    def _gap(kind: str, start: int, end: int, points: int) -> Row:
        return {"kind": kind, "start_ms": start, "end_ms": end, "missing_points": points}

    def result(self, window: Window, *, phase: int | None = None) -> Row:
        start, end = window.available_start, window.available_end
        if phase is None and self.first is not None:
            phase = self.first % PERIOD_MS

        def expected(left: int, right: int) -> int:
            first = left if phase is None else left + (phase - left) % PERIOD_MS
            return max(0, (right - first + PERIOD_MS - 1) // PERIOD_MS)

        gaps = [gap.copy() for gap in self._gaps]
        gap_count, missing = self._gap_count, self._missing
        if self.first is None or self.last is None:
            missing = expected(start, end)
            if missing:
                gaps.append(self._gap("empty", start, end, missing))
                gap_count += 1
        else:
            leading = expected(start, self.first)
            if leading:
                gaps.insert(0, self._gap("leading", start, self.first, leading))
                gap_count += 1
                missing += leading
            trailing = max(0, (end - self.last - 1) // PERIOD_MS)
            if trailing:
                gaps.append(self._gap("trailing", self.last + PERIOD_MS, end, trailing))
                gap_count += 1
                missing += trailing
        unavailable = window.unavailable()
        status = "complete"
        if not self.points:
            status = "no_coverage"
        elif missing or unavailable or self.irregular_intervals:
            status = "partial_coverage"
        return {
            "status": status, "expected_points": expected(window.start_ms, window.end_ms),
            "expected_available_points": expected(start, end), "points": self.points,
            "unique_timestamps": self.unique_timestamps,
            "valid_points": self.valid_points.copy(),
            "invalid_points": {name: self.points - count for name, count in self.valid_points.items()},
            "first_ms": self.first, "last_ms": self.last,
            "first_beijing": beijing(self.first) if self.first is not None else None,
            "last_beijing": beijing(self.last) if self.last is not None else None,
            "gap_count": gap_count, "missing_available_points": missing, "gaps": gaps,
            "gaps_truncated": len(gaps) < gap_count, "irregular_intervals": self.irregular_intervals,
            "unavailable_ranges": unavailable,
        }
