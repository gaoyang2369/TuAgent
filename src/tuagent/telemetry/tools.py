import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..tools import ToolContext, ToolOutput, ToolRegistry
from .analytics import summarize
from .contracts import EventArgs, MetadataArgs, PROFILE, QueryArgs, SummaryArgs, TelemetryError
from .data import beijing, json_default, load_metadata, time_range
from .events import find_events
from .mysql import FIELDS, MySQLTelemetrySource, Row, TelemetrySource
from .query import HistoryCoverageCache, make_cursor, parse_cursor, query_window, read_points


MODEL_CONTENT_BYTES = 16 * 1024
_COMPACT_METADATA = (
    "stored_unit", "display_unit", "canonical_unit", "scale_factor", "offset_value",
    "invalid_values", "verification_status", "source_sha256",
)


def _dump(result: Row) -> str:
    return json.dumps(result, ensure_ascii=False, default=json_default, separators=(",", ":"), allow_nan=False)


def _model_content(result: Row) -> str:
    payload = dict(result)
    if "coverage" not in result:
        items = list(result["metadata"].items())
        payload["field_names"] = list(result["metadata"])
        collection = "metadata"
    else:
        payload["metadata"] = {
            name: {key: row[key] for key in _COMPACT_METADATA}
            for name, row in result["metadata"].items()
        }
        payload["metadata_view"] = "units_and_validity; call get_signal_metadata for full definitions"
        collection = next(name for name in ("data", "buckets", "events") if name in result)
        items = result[collection]

    is_latest = result.get("mode") == "latest"
    if "code_changes" in result:
        payload["code_changes"] = []

    def candidate(size: int, change_size: int = 0) -> str:
        visible = items[-size:] if is_latest and size else items[:size]
        payload[collection] = dict(visible) if collection == "metadata" else visible
        truncated = size < len(items)
        if "code_changes" in result:
            payload["code_changes"] = result["code_changes"][:change_size]
            truncated = truncated or change_size < len(result["code_changes"])
        payload["content_truncated"] = truncated
        payload.pop("follow_up", None)
        if collection == "data":
            page = {**result["page"], "returned_points": size}
            page["truncated"] = page["truncated"] or truncated
            page["first_ms"] = visible[0]["timestamp_ms"] if visible else None
            page["last_ms"] = visible[-1]["timestamp_ms"] if visible else None
            if truncated and not is_latest:
                page["next_cursor"] = make_cursor(result, visible[-1]) if visible else None
            payload["page"] = page
            if page["next_cursor"]:
                payload["follow_up"] = "Pass next_cursor as cursor with the same source, fields and window mode."
            elif page["truncated"]:
                payload["follow_up"] = "Use window mode with explicit bounds or request fewer fields per call."
        elif truncated or result.get("events_truncated") or result.get("code_changes_truncated"):
            payload["follow_up"] = (
                "Request a smaller list of fields." if collection == "metadata"
                else "Query shorter explicit windows to read the remaining buckets or event observations."
            )
        return _dump(payload)

    def fits(content: str) -> bool:
        return len(content.encode("utf-8")) <= MODEL_CONTENT_BYTES

    all_changes = len(result.get("code_changes", []))
    full = candidate(len(items), all_changes)
    if fits(full):
        return full

    def max_visible(upper: int, render: Callable[[int], str]) -> int:
        low, high = 0, upper
        while low < high:
            middle = (low + high + 1) // 2
            if fits(render(middle)):
                low = middle
            else:
                high = middle - 1
        return low

    size = max_visible(len(items), candidate)
    changes = max_visible(all_changes, lambda count: candidate(size, count))
    content = candidate(size, changes)
    if fits(content):
        return content
    return _dump({
        "status": result["status"], "source": result["source"],
        "requested_range": result.get("requested_range"), "content_truncated": True,
        "follow_up": "Request fewer fields or a shorter window.",
    })


class _TelemetryTools:
    def __init__(
        self, source: TelemetrySource, expected_hash: str | None, clock_ms: Callable[[], int]
    ) -> None:
        self._source = source
        self._expected_hash = expected_hash
        self._clock_ms = clock_ms
        self._history_coverage = HistoryCoverageCache()

    def _read(self, args: MetadataArgs | QueryArgs | SummaryArgs | EventArgs) -> Row:
        if args.profile_version != PROFILE:
            raise TelemetryError("version_error", f"Unsupported profile {args.profile_version!r}; expected {PROFILE}")
        cursor = parse_cursor(args) if isinstance(args, QueryArgs) else None
        if isinstance(args, EventArgs):
            fields = (args.code_field,)
        else:
            fields = tuple(dict.fromkeys(args.fields or (*FIELDS, "actual_solution")))
        read_ms = self._clock_ms()

        with self._source.read() as db:
            expected = self._expected_hash
            if expected is None:
                expected = getattr(db, "expected_source_sha256", None)
            metadata, source = load_metadata(db, args, fields, expected)
            result: Row = {
                "status": "ok", "source": source, "metadata": metadata,
                "read_at_ms": read_ms, "read_at_beijing": beijing(read_ms),
            }
            if isinstance(args, MetadataArgs):
                return result
            if cursor and cursor.sha256 != source["source_sha256"]:
                raise TelemetryError("version_error", "Metadata hash changed since the history cursor was issued")

            if isinstance(args, QueryArgs):
                window, data = read_points(db, args, read_ms, metadata, cursor, self._history_coverage)
            else:
                window = query_window(args, read_ms)
                if isinstance(args, SummaryArgs):
                    data = summarize(db, args, window, metadata)
                else:
                    data = find_events(db, args, window, metadata[args.code_field])
            result.update(data)
            result["requested_range"] = time_range(window.start_ms, window.end_ms)
            if not data["coverage"]["points"]:
                result["status"] = "no_coverage"
            if isinstance(args, QueryArgs) and args.mode == "window" and data["page"]["truncated"]:
                result["page"]["next_cursor"] = make_cursor(result, data["data"][-1])
            return result

    def _output(self, args: MetadataArgs | QueryArgs | SummaryArgs | EventArgs) -> ToolOutput:
        try:
            result = self._read(args)
        except TelemetryError as exc:
            error = {
                "status": exc.status, "message": str(exc), "table_name": args.table_name,
                "profile_version": args.profile_version,
            }
            return ToolOutput(_dump(error), details=error, is_error=True)
        return ToolOutput(_model_content(result), details=result)

    async def execute(self, args: Any, context: ToolContext) -> ToolOutput:
        return await asyncio.to_thread(self._output, args)


def register_telemetry_tools(
    registry: ToolRegistry, *,
    source: TelemetrySource | None = None,
    env_file: Path | None = None,
    expected_source_sha256: str | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> None:
    """Register G120 telemetry tools. Connection settings stay on the application side."""
    tools = _TelemetryTools(
        source if source is not None else MySQLTelemetrySource(env_file=env_file),
        expected_source_sha256, clock_ms or (lambda: time.time_ns() // 1_000_000),
    )
    registry.register(
        "get_signal_metadata",
        "Read G120 simulator field definitions, units, conversions, invalid values and model verification by source.",
        MetadataArgs, tools.execute,
    )
    registry.register(
        "query_telemetry",
        "Read original G120 samples in time and row ID order. Use latest for recent points (up to 1000) "
        "or window for a half-open Unix millisecond range. Use the history cursor for further pages.",
        QueryArgs, tools.execute,
    )
    registry.register(
        "summarize_telemetry",
        "Compute canonical-unit numeric statistics and continuous threshold durations in time buckets "
        "within a half-open G120 telemetry window.",
        SummaryArgs, tools.execute,
    )
    registry.register(
        "find_code_events",
        "Find G120 alarm or fault latches, observed clears, code changes and confirmed new triggers in a window. "
        "If filtering, provide the full code including the leading letter and zeros.",
        EventArgs, tools.execute,
    )
