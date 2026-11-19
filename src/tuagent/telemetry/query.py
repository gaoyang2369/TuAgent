import base64
import binascii
import math
from decimal import Decimal
from threading import Lock
from typing import Annotated

from pydantic import Field, ValidationError

from .contracts import (
    MAX_TIMESTAMP_MS, PERIOD_MS, RETENTION_MS, Arguments, EpochMs, FieldName,
    QueryArgs, Seconds, TableName, TelemetryError, WindowArgs,
)
from .data import Coverage, Window, sample_key, sample_time, valid_value
from .mysql import MAX_WINDOW_ROWS, Row, TelemetryReader


class PageCursor(Arguments):
    table: TableName
    profile: str
    fields: tuple[FieldName, ...]
    sha256: str = Field(pattern="^[0-9a-f]{64}$")
    start_ms: EpochMs
    end_ms: EpochMs
    relative_seconds: Seconds | None
    after_ms: EpochMs
    after_id: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
    revision: tuple[int, int] | None = None


class HistoryCoverageCache:
    """Keep recent full-window quality scans. A cache miss can always be rescanned."""

    def __init__(self) -> None:
        self._entries: dict[tuple, Coverage] = {}
        self._lock = Lock()

    def get(self, key: tuple) -> Coverage | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: tuple, coverage: Coverage) -> None:
        with self._lock:
            if key not in self._entries and len(self._entries) >= 16:
                del self._entries[next(iter(self._entries))]
            self._entries[key] = coverage


def query_window(args: WindowArgs, read_ms: int, cursor: PageCursor | None = None) -> Window:
    if cursor:
        return Window(cursor.start_ms, cursor.end_ms, read_ms)
    if args.last_seconds is not None:
        return Window(max(0, read_ms - args.last_seconds * 1000), read_ms, read_ms)
    assert args.start_ms is not None and args.end_ms is not None
    return Window(args.start_ms, args.end_ms, read_ms)


def parse_cursor(args: QueryArgs) -> PageCursor | None:
    if args.cursor is None:
        return None
    try:
        raw = base64.b64decode(args.cursor, altchars=b"-_", validate=True)
        cursor = PageCursor.model_validate_json(raw)
    except (binascii.Error, ValidationError, ValueError):
        raise TelemetryError("invalid_cursor", "Invalid history cursor; start a new window query") from None
    if cursor.revision is None:
        raise TelemetryError("stale_cursor", "This cursor has no window revision. Restart from the first page.")
    fields = tuple(dict.fromkeys(args.fields))
    if (
        (cursor.table, cursor.profile, cursor.fields) != (args.table_name, args.profile_version, fields)
        or not 0 < cursor.end_ms - cursor.start_ms <= RETENTION_MS
        or not cursor.start_ms <= cursor.after_ms < cursor.end_ms
        or (args.last_seconds is not None and args.last_seconds != cursor.relative_seconds)
        or (args.start_ms is not None and (args.start_ms, args.end_ms) != (cursor.start_ms, cursor.end_ms))
    ):
        raise TelemetryError("invalid_cursor", "History cursor does not match the requested source, fields or window")
    return cursor


def make_cursor(result: Row, point: Row) -> str:
    cursor = PageCursor(
        table=result["source"]["table_name"], profile=result["source"]["profile_version"],
        fields=tuple(result["metadata"]), sha256=result["source"]["source_sha256"],
        start_ms=result["requested_range"]["start_ms"], end_ms=result["requested_range"]["end_ms"],
        relative_seconds=result["relative_seconds"], after_ms=point["timestamp_ms"], after_id=point["id"],
        revision=(result["coverage"]["points"], result["window_max_id"]),
    )
    return base64.urlsafe_b64encode(cursor.model_dump_json().encode()).decode()


def _validity(row: Row, metadata: dict[str, Row]) -> dict[str, bool]:
    return {name: valid_value(name, row[name], definition) for name, definition in metadata.items()}


def _point(row: Row, metadata: dict[str, Row], args: QueryArgs) -> Row:
    values = {name: row[name] for name in metadata}
    for name, value in values.items():
        if isinstance(value, (float, Decimal)) and not math.isfinite(value):
            values[name] = str(value)  # JSON has no non-finite numeric literal.
    ms, row_id = sample_key(row)
    return {
        **sample_time(row), "ref": f"{args.table_name}/{args.profile_version}/point/{ms}/{row_id}",
        "values": values, "valid": _validity(row, metadata),
    }


def _history_page(
    db: TelemetryReader, args: QueryArgs, window: Window, metadata: dict[str, Row],
    cursor: PageCursor | None, cache: HistoryCoverageCache,
) -> tuple[list[Row], Coverage, int]:
    fields = tuple(metadata)
    source_hash = metadata[fields[0]]["source_sha256"].lower()
    key = (args.table_name, args.profile_version, source_hash, fields, window.start_ms, window.end_ms)
    after = (cursor.after_ms, cursor.after_id) if cursor else None
    bounds = (window.available_start, window.available_end)
    cached = cache.get((*key, cursor.revision)) if cursor else None
    # Readers providing the optional fast path check the revision and fetch rows in one transaction.
    fetch_page = getattr(db, "history_page", None)
    if cached is not None and fetch_page is not None:
        revision, rows = fetch_page(args.table_name, fields, *bounds, after=after, limit=args.limit + 1)
        coverage = cached
    else:
        coverage = Coverage(fields)
        rows = []
        max_id = 0
        for row in db.samples(args.table_name, fields, *bounds):
            if coverage.points == MAX_WINDOW_ROWS:
                raise TelemetryError("window_row_limit", "Window has too many rows; use smaller windows")
            row_key = sample_key(row)
            coverage.add(row_key[0], _validity(row, metadata))
            max_id = max(max_id, row_key[1])
            if len(rows) <= args.limit and (after is None or row_key > after):
                rows.append(row)
        revision = (coverage.points, max_id)
    if cursor is not None and cursor.revision != revision:
        raise TelemetryError("stale_cursor", "The rows in this window changed. Restart the original window from the first page.")
    if cached is None:
        cache.put((*key, revision), coverage)
    return rows, coverage, revision[1]


def read_points(
    db: TelemetryReader, args: QueryArgs, read_ms: int, metadata: dict[str, Row],
    cursor: PageCursor | None, cache: HistoryCoverageCache,
) -> tuple[Window, Row]:
    fields = tuple(metadata)
    latest_age_ms = None
    window_max_id = None
    if args.mode == "latest":
        rows = list(db.samples(
            args.table_name, fields, max(0, read_ms - RETENTION_MS),
            min(MAX_TIMESTAMP_MS, read_ms + 1), limit=args.limit + 1, newest_first=True,
        ))
        more = len(rows) > args.limit
        newest_ms = sample_key(rows[0])[0] if rows else read_ms
        end = newest_ms + 1
        window = Window(max(0, end - args.limit * PERIOD_MS), end, read_ms)
        # The most recent N stored points can extend further back when samples are missing.
        if rows:
            window = Window(min(window.start_ms, sample_key(rows[min(len(rows), args.limit) - 1])[0]), end, read_ms)
            latest_age_ms = read_ms - newest_ms
        rows = list(reversed(rows[:args.limit]))
        coverage = Coverage(fields)
        for row in rows:
            coverage.add(sample_key(row)[0], _validity(row, metadata))
    else:
        window = query_window(args, read_ms, cursor)
        rows, coverage, window_max_id = _history_page(db, args, window, metadata, cursor, cache)
        more = len(rows) > args.limit
    page = [_point(row, metadata, args) for row in rows[:args.limit]]

    return window, {
        "mode": args.mode,
        "window_max_id": window_max_id,
        "relative_seconds": cursor.relative_seconds if cursor else args.last_seconds,
        "data": page, "values_are": "stored", "coverage": coverage.result(window),
        "coverage_scope": "full_window" if args.mode == "window" else "latest_selection",
        "latest_sample_age_ms": latest_age_ms,
        "page": {
            "limit": args.limit, "returned_points": len(page), "truncated": more,
            "first_ms": page[0]["timestamp_ms"] if page else None,
            "last_ms": page[-1]["timestamp_ms"] if page else None,
            "next_cursor": None,
        },
    }
