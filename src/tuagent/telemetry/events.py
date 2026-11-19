from collections.abc import Iterable, Iterator
from itertools import groupby

from .contracts import PERIOD_MS, EventArgs, TelemetryError
from .data import Coverage, Window, sample_key, sample_time, valid_value
from .mysql import MAX_WINDOW_ROWS, Row, TelemetryReader


_EVENT_LIMIT = 1000


def _code_observations(
    rows: Iterable[Row], name: str, definition: Row, window_start: int, coverage: Coverage,
) -> Iterator[tuple[Row, str | None, bool]]:
    scanned = 0
    for ms, group in groupby(rows, key=lambda row: int(row["timestamp"])):
        first = None
        first_code = None
        conflict = False
        for row in group:
            scanned += 1
            if scanned > MAX_WINDOW_ROWS:
                raise TelemetryError("window_row_limit", "Too many rows including the predecessor; use a smaller window")
            valid = valid_value(name, row[name], definition)
            code = row[name] if valid else None
            if ms >= window_start:
                coverage.add(ms, {name: valid})
            if first is None:
                first, first_code = row, code
            elif code != first_code:
                conflict = True
        assert first is not None
        yield first, None if conflict else first_code, conflict


def find_events(db: TelemetryReader, args: EventArgs, window: Window, definition: Row) -> Row:
    name = args.code_field
    fields = (name,)
    prior = None
    scan_start = window.available_start
    if window.available_start < window.available_end:
        prior_rows = list(db.samples(
            args.table_name, fields, window.cutoff_ms, window.available_start,
            limit=1, newest_first=True,
        ))
        if prior_rows:
            # Include every row at the predecessor timestamp when reconciling its state.
            scan_start = sample_key(prior_rows[0])[0]
    previous_ms = previous_code = None
    coverage = Coverage(fields)
    events: list[Row] = []
    changes: list[Row] = []
    event_count = change_count = new_trigger_count = 0
    conflicting_times = 0
    first_in_window = True
    active = None

    def begin(code: str, row: Row, boundary: str, *, prior_code: str | None = None) -> Row:
        when = sample_time(row)
        event: Row = {
            "ref": f"{args.table_name}/{args.profile_version}/{name}/{code}/{when['timestamp_ms']}/{when['id']}",
            "code": code, "start_boundary": boundary,
            "crosses_window_start": boundary == "continued_from_before_window",
            "first_observed": when, "last_observed": when,
            "trigger": when if boundary == "observed_trigger" else None, "clear": None,
            "confirmed_duration_in_window_ms": 0,
        }
        if prior_code is not None:
            event["previous_code"] = prior_code
            event["code_change_at"] = when
        return event

    def finish(event: Row, boundary: str, *, row: Row | None = None, next_code: str | None = None) -> None:
        nonlocal event_count, new_trigger_count
        if args.target_code and event["code"] != args.target_code:
            return
        event["end_boundary"] = boundary
        event["crosses_window_end_unknown"] = boundary in ("window_open", "sampling_gap_at_end")
        if boundary == "observed_clear" and row is not None:
            event["clear"] = sample_time(row)
        if next_code is not None and row is not None:
            event["next_code"] = next_code
            event["changed_to_at"] = sample_time(row)
        trigger, clear = event["trigger"], event["clear"]
        event["observed_trigger_to_clear_ms"] = (
            clear["timestamp_ms"] - trigger["timestamp_ms"] if trigger and clear else None
        )
        if trigger is not None:
            new_trigger_count += 1
        event_count += 1
        if len(events) < _EVENT_LIMIT:
            events.append(event)

    rows = db.samples(args.table_name, fields, scan_start, window.available_end)
    for row, code, conflict in _code_observations(rows, name, definition, window.available_start, coverage):
        ms, _ = sample_key(row)
        if ms < window.available_start:
            prior, previous_ms, previous_code = row, ms, code
            continue
        conflicting_times += conflict
        adjacent = previous_ms is not None and ms - previous_ms == PERIOD_MS
        if first_in_window and adjacent and code is not None and previous_code not in (None, "0"):
            assert prior is not None
            active = begin(previous_code, prior, "continued_from_before_window")
        first_in_window = False

        if not adjacent or previous_code is None or code is None:
            if active is not None:
                finish(active, "invalid_code" if adjacent else "sampling_gap")
                active = None
            if code not in (None, "0"):
                if previous_ms is None:
                    reason = "unknown_missing_predecessor"
                elif not adjacent:
                    reason = "unknown_sampling_gap"
                else:
                    reason = "unknown_invalid_predecessor"
                active = begin(code, row, reason)
        elif code == previous_code:
            if active is not None:
                assert previous_ms is not None
                active["confirmed_duration_in_window_ms"] += ms - max(previous_ms, window.start_ms)
                active["last_observed"] = sample_time(row)
        elif code == "0":
            if active is not None:
                finish(active, "observed_clear", row=row)
                active = None
        elif previous_code == "0":
            active = begin(code, row, "observed_trigger")
        else:
            if active is not None:
                finish(active, "code_change", row=row, next_code=code)
            if args.target_code in (None, previous_code, code):
                change_count += 1
                if len(changes) < _EVENT_LIMIT:
                    changes.append({"from": previous_code, "to": code, "at": sample_time(row)})
            active = begin(code, row, "code_change", prior_code=previous_code)
        previous_ms, previous_code = ms, code

    if active is not None:
        assert previous_ms is not None
        boundary = "sampling_gap_at_end" if previous_ms + PERIOD_MS < window.available_end else "window_open"
        finish(active, boundary)
    return {
        "coverage": coverage.result(window), "values_are": "stored",
        "code_field": name, "target_code": args.target_code,
        "events": events, "event_count": event_count, "new_trigger_count": new_trigger_count,
        "conflicting_code_timestamps": conflicting_times,
        "code_changes": changes, "code_change_count": change_count,
        "events_truncated": len(events) < event_count,
        "code_changes_truncated": len(changes) < change_count,
        "duration_method": "Sum in-window intervals between adjacent samples carrying the same code.",
    }
