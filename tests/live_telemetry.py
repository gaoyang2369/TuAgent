"""Run all telemetry tools through Agent.run against the configured read-only MySQL source."""

import asyncio
import json

from fakes import ScriptedModel
from tuagent import (
    Agent, AssistantMessage, ModelReply, ModelRequest, ModelToolResult, ToolCall, ToolExecutor,
    ToolRegistry, ToolResultMessage,
)
from tuagent.telemetry import TELEMETRY_ANALYSIS_PROMPT, register_telemetry_tools
from tuagent.telemetry.contracts import PERIOD_MS


TABLES = (
    "sim_normal", "sim_f07900_blocked", "sim_f30002_overvoltage", "sim_f07011_overtemp",
)


def _overview(receipt: ToolResultMessage) -> dict:
    data = receipt.details or {"status": receipt.error_code, "message": receipt.content}
    report = {"call_id": receipt.tool_call_id, "status": data["status"]}
    if receipt.is_error:
        return {**report, "message": data["message"]}
    report.update(source=data["source"], read_at_beijing=data["read_at_beijing"])
    if "coverage" not in data:
        report["metadata_definitions"] = len(data["metadata"])
        return report
    quality = data["coverage"]
    report["requested_range"] = data["requested_range"]
    report["coverage"] = {
        key: quality[key]
        for key in (
            "status", "points", "expected_points", "first_beijing", "last_beijing",
            "valid_points", "invalid_points", "gap_count", "missing_available_points",
            "gaps", "unavailable_ranges",
        )
    }
    if "data" in data:
        report["returned_points"] = data["page"]["returned_points"]
    if "buckets" in data:
        report["returned_buckets"] = data["bucket_count"]
        peaks = [bucket["signals"]["motor_temp"]["max"] for bucket in data["buckets"]]
        peak = max((point for point in peaks if point is not None), key=lambda point: point["value"], default=None)
        report["motor_temp_peak"] = (
            {**peak, "unit": data["metadata"]["motor_temp"]["canonical_unit"]} if peak else None
        )
    if "events" in data:
        report.update(
            returned_events=len(data["events"]), new_trigger_count=data["new_trigger_count"],
            events=data["events"],
        )
    return report


def _call(call_id: str, name: str, table: str, **args) -> ToolCall:
    return ToolCall(call_id, name, json.dumps({"table_name": table, **args}))


async def _audit_history(executor: ToolExecutor, summary: dict) -> dict:
    end = summary["coverage"]["last_ms"]
    bounds = {"start_ms": end - 10_800_000, "end_ms": end}
    keys = []
    page_sizes = []
    cursor = None
    while True:
        page = await executor.execute(_call(
            "history", "query_telemetry", TABLES[-1], mode="window", fields=["motor_temp", "air_intake_temp"],
            limit=1000, cursor=cursor, **bounds,
        ))
        if page.is_error or page.details["status"] != "ok":
            return {"passed": False, "result": _overview(page)}
        data = page.details
        keys.extend((item["timestamp_ms"], item["id"]) for item in data["data"])
        page_sizes.append(data["page"]["returned_points"])
        cursor = data["page"]["next_cursor"]
        if not cursor:
            break

    return {
        "passed": len(keys) == data["coverage"]["points"] and keys == sorted(set(keys)),
        "requested_range": data["requested_range"], "page_sizes": page_sizes,
        "received_points": len(keys), "expected_points": data["coverage"]["expected_points"],
        "gap_count": data["coverage"]["gap_count"], "invalid_points": data["coverage"]["invalid_points"],
    }


async def _audit_latch(executor: ToolExecutor, events: dict) -> dict:
    event = next((item for item in events["events"] if item["trigger"] and item["clear"]), None)
    if event is None:
        return {"passed": False, "status": "not_verified", "reason": "no_observed_trigger_and_clear_in_window"}
    start = event["trigger"]["timestamp_ms"]
    end = event["clear"]["timestamp_ms"] + 1
    raw = await executor.execute(_call(
        "raw_latch", "query_telemetry", TABLES[1], mode="window", fields=["fault_code"],
        start_ms=start - PERIOD_MS, end_ms=end,
    ))
    continued = await executor.execute(_call(
        "continued_latch", "find_code_events", TABLES[1], target_code="F07900",
        start_ms=start + PERIOD_MS, end_ms=end,
    ))
    if raw.is_error or continued.is_error:
        return {"passed": False, "results": [_overview(raw), _overview(continued)]}
    code_rows = sum(item["values"]["fault_code"] == "F07900" for item in raw.details["data"])
    segments = continued.details["events"]
    continued_at_start = bool(segments and segments[0]["crosses_window_start"])
    return {
        "passed": code_rows > 1 and continued_at_start and continued.details["new_trigger_count"] == 0,
        "event_ref": event["ref"], "latched_rows": code_rows,
        "continuation": continued_at_start,
        "continuation_new_trigger_count": continued.details["new_trigger_count"],
    }


async def main() -> bool:
    fields = ["timestamp", "status", "current_actual", "motor_temp", "air_intake_temp", "fault_code"]
    calls = [
        _call("metadata", "get_signal_metadata", TABLES[-1], fields=fields),
        *(_call(table, "query_telemetry", table, fields=fields, limit=10) for table in TABLES),
        _call(
            "overtemp_3h", "summarize_telemetry", TABLES[-1], last_seconds=10800,
            bucket_seconds=60, fields=["motor_temp", "air_intake_temp", "current_actual", "motor_load_rate"],
        ),
        _call(
            "blocked_1h", "find_code_events", TABLES[1], last_seconds=3600,
            code_field="fault_code", target_code="F07900",
        ),
    ]
    observed = []

    def final_reply(request: ModelRequest) -> ModelReply:
        observed.extend(message for message in request.messages if isinstance(message, ModelToolResult))
        return ModelReply(AssistantMessage("Telemetry tool results received."))

    registry = ToolRegistry()
    register_telemetry_tools(registry)
    model = ScriptedModel(
        ModelReply(AssistantMessage(tool_calls=tuple(calls)), "tool_calls"), final_reply,
    )
    result = await Agent(
        model, tools=registry, system_prompt=TELEMETRY_ANALYSIS_PROMPT, max_turns=2,
    ).run("Read recent G120 simulator samples, motor temperature history and blocked-drive events.")
    receipts = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    round_trip = (
        result.status == "completed"
        and [receipt.tool_call_id for receipt in receipts] == [request.id for request in calls]
        and observed == [ModelToolResult(item.tool_call_id, item.content, item.is_error) for item in receipts]
    )
    healthy = round_trip and all(not item.is_error and item.details["status"] == "ok" for item in receipts)
    audits = {}
    if healthy:
        executor = ToolExecutor(registry)
        audits["historical_pagination"] = await _audit_history(executor, receipts[-2].details)
        audits["latch_continuation"] = await _audit_latch(executor, receipts[-1].details)
    print(json.dumps({
        "agent_round_trip": round_trip,
        "results": [_overview(item) for item in receipts],
        "audits": audits,
    }, ensure_ascii=True, indent=2))
    return healthy and all(report["passed"] for report in audits.values())


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
