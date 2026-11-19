import operator
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any

from .contracts import MAX_BUCKETS, PERIOD_MS, SummaryArgs, TelemetryError
from .data import Coverage, Window, sample_key, sample_time, time_range, valid_value
from .mysql import MAX_WINDOW_ROWS, Row, TelemetryReader


_COMPARE = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
_UNIT_ALIASES = {"°C": "degC", "℃": "degC"}
_SEGMENT_LIMIT = 8


@dataclass
class _SignalStats:
    count: int = 0
    total: float = 0.0
    first: tuple[float, Row] | None = None
    last: tuple[float, Row] | None = None
    minimum: tuple[float, Row] | None = None
    maximum: tuple[float, Row] | None = None
    duration_ms: int = 0
    longest_ms: int = 0
    segment_start: int | None = None
    segment_end: int | None = None
    segment_count: int = 0
    segments: list[Row] = field(default_factory=list)

    def add(self, value: float, row: Row) -> None:
        point = (value, row)
        self.count += 1
        self.total += value
        if self.first is None:
            self.first = point
        self.last = point
        if self.minimum is None or value < self.minimum[0]:
            self.minimum = point
        if self.maximum is None or value > self.maximum[0]:
            self.maximum = point

    def add_threshold_interval(self, start: int, end: int) -> None:
        if self.segment_end != start:
            self.segment_start = start
            self.segment_count += 1
            if self.segment_count <= _SEGMENT_LIMIT:
                self.segments.append({"start_ms": start, "end_ms": end})
        self.segment_end = end
        self.duration_ms += end - start
        assert self.segment_start is not None
        self.longest_ms = max(self.longest_ms, end - self.segment_start)
        if self.segment_count <= _SEGMENT_LIMIT:
            self.segments[-1]["end_ms"] = end

    def result(self, unit: str, coverage: Coverage, metrics: set[str]) -> Row:
        result: Row = {
            "unit": unit, "valid_points": self.count, "invalid_points": coverage.points - self.count,
        }
        for metric, point in (("min", self.minimum), ("max", self.maximum)):
            if metric in metrics:
                result[metric] = {"value": point[0], **sample_time(point[1])} if point else None
        if "mean" in metrics:
            result["mean"] = self.total / self.count if self.count else None
        if "rate" in metrics:
            result["rate_per_second"] = None
            if self.first and self.last:
                elapsed_ms = sample_key(self.last[1])[0] - sample_key(self.first[1])[0]
                if elapsed_ms > 0:
                    result["rate_per_second"] = (self.last[0] - self.first[0]) * 1000 / elapsed_ms
        return result

    def threshold_result(self) -> Row:
        return {
            "total_ms": self.duration_ms, "longest_continuous_ms": self.longest_ms,
            "segments": self.segments, "segment_count": self.segment_count,
            "segments_truncated": len(self.segments) < self.segment_count,
        }


def summarize(
    db: TelemetryReader, args: SummaryArgs, window: Window, metadata: dict[str, Row]
) -> Row:
    width = args.bucket_seconds * 1000
    count = (window.end_ms - window.start_ms + width - 1) // width
    if count > MAX_BUCKETS:
        raise TelemetryError("invalid_window", f"Increase bucket_seconds or shorten the window; max {MAX_BUCKETS} buckets")
    fields = tuple(metadata)
    units = {name: definition["canonical_unit"] for name, definition in metadata.items()}
    missing_units = [name for name, unit in units.items() if not isinstance(unit, str) or not unit]
    if missing_units:
        raise TelemetryError("version_error", f"Missing canonical units for {', '.join(missing_units)}")
    thresholds = {item.field: item for item in args.thresholds}
    for name, threshold in thresholds.items():
        if _UNIT_ALIASES.get(threshold.unit, threshold.unit) != _UNIT_ALIASES.get(units[name], units[name]):
            raise TelemetryError("invalid_threshold_unit", f"{name} requires canonical unit {units[name]!r}")
    metrics = set(args.metrics)
    if thresholds:
        metrics.add("threshold_duration")

    coverage = Coverage(fields)
    bucket_quality = [Coverage(fields, gap_limit=2) for _ in range(count)]
    bucket_stats = [{name: _SignalStats() for name in fields} for _ in range(count)]
    conversion = {
        name: (float(definition["scale_factor"]), float(definition["offset_value"]))
        for name, definition in metadata.items()
    }
    previous_ms = None
    previous_exceeds = set()
    rows = db.samples(args.table_name, fields, window.available_start, window.available_end)
    for ms, group in groupby(rows, key=lambda row: int(row["timestamp"])):
        index = (ms - window.start_ms) // width
        # A time interval qualifies only when every reading at each endpoint qualifies.
        exceeds = set(thresholds)
        for row in group:
            if coverage.points == MAX_WINDOW_ROWS:
                raise TelemetryError("window_row_limit", "Window has too many rows; use smaller windows")
            validity = {}
            for name, definition in metadata.items():
                stored = row[name]
                valid = valid_value(name, stored, definition)
                validity[name] = valid
                if not valid:
                    exceeds.discard(name)
                    continue
                scale, offset = conversion[name]
                value = float(stored) * scale + offset
                bucket_stats[index][name].add(value, row)
                threshold = thresholds.get(name)
                if threshold and not _COMPARE[threshold.operator](value, threshold.value):
                    exceeds.discard(name)
            bucket_quality[index].add(ms, validity)
            coverage.add(ms, validity)
        if previous_ms is not None and ms - previous_ms == PERIOD_MS:
            old_index = (previous_ms - window.start_ms) // width
            boundary = window.start_ms + index * width
            for name in exceeds & previous_exceeds:
                if index == old_index:
                    bucket_stats[index][name].add_threshold_interval(previous_ms, ms)
                else:
                    bucket_stats[old_index][name].add_threshold_interval(previous_ms, boundary)
                    if boundary < ms:
                        bucket_stats[index][name].add_threshold_interval(boundary, ms)
        previous_ms, previous_exceeds = ms, exceeds

    calculation: dict[str, Any] = {
        "bucket_seconds": args.bucket_seconds,
        "fields": list(fields),
        "metrics": sorted(metrics),
        "thresholds": [{**item.model_dump(), "unit": units[item.field]} for item in args.thresholds],
        "rate_method": "(last valid value - first valid value) / elapsed seconds within the bucket",
        "threshold_method": (
            "Sum intervals between adjacent valid qualifying samples exactly 3000 ms apart. "
            "Clip intervals at bucket boundaries. Both samples must be in the query window."
        ),
    }
    buckets = []
    phase = coverage.first % PERIOD_MS if coverage.first is not None else None
    for index, stats in enumerate(bucket_stats):
        start = window.start_ms + index * width
        end = min(start + width, window.end_ms)
        bucket_window = Window(start, end, window.read_ms)
        quality = bucket_quality[index]
        signals = {}
        for name, signal in stats.items():
            signals[name] = signal.result(units[name], quality, metrics)
            if name in thresholds:
                signals[name]["threshold_duration"] = signal.threshold_result()
        buckets.append({
            **time_range(start, end),
            "ref": f"{args.table_name}/{args.profile_version}/bucket/{start}/{end}",
            "coverage": quality.result(bucket_window, phase=phase),
            "signals": signals,
        })
    return {
        "coverage": coverage.result(window), "buckets": buckets,
        "bucket_count": count, "calculation": calculation, "values_are": "canonical",
    }
