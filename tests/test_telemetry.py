import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from fakes import ScriptedModel
from tuagent import (
    Agent, AssistantMessage, ModelReply, ModelToolResult, ToolCall, ToolExecutor, ToolRegistry,
)
from tuagent.telemetry import MySQLConfig, TELEMETRY_ANALYSIS_PROMPT, register_telemetry_tools
from tuagent.telemetry.contracts import PERIOD_MS, PROFILE, RETENTION_MS, TelemetryError
from tuagent.telemetry.mysql import FIELDS, MySQLTelemetrySource
from tuagent.telemetry.tools import MODEL_CONTENT_BYTES


BASE = 1_784_001_600_000
OVERHEAT = "sim_f07011_overtemp"
BLOCKED = "sim_f07900_blocked"
OVERVOLTAGE = "sim_f30002_overvoltage"
HASH = "a" * 64
TOOL_NAMES = (
    "get_signal_metadata", "query_telemetry", "summarize_telemetry", "find_code_events",
)


def sample(index, **values):
    return {
        "id": index + 1, "timestamp": str(BASE + index * PERIOD_MS),
        "motor_temp": 20 + index / 30, "air_intake_temp": 25,
        "speed_actual": 0, "current_actual": 0, "dc_voltage": 557,
        "fault_code": "0", "alarm_code": "0", **values,
    }


class MemorySource:
    def __init__(self):
        self.data = {}
        self.definitions = {}
        for name in (*FIELDS, "actual_solution"):
            unit = {
                "motor_temp": "degC", "air_intake_temp": "degC", "current_actual": "A",
                "dc_voltage": "V", "speed_actual": "rpm",
            }.get(name)
            self.definitions[name] = {
                "field_name": name, "description": name, "quantity_kind": "test signal",
                "storage_type": "FLOAT" if unit else "VARCHAR", "is_nullable": 1,
                "stored_unit": unit, "display_unit": unit, "canonical_unit": unit,
                "scale_factor": Decimal("1"), "offset_value": Decimal("0"),
                "value_conventions": '{"kind":"mock"}',
                "invalid_values": "[null,-200]" if name == "air_intake_temp" else "[null]",
                "definition_source": "test", "scope_note": "simulator only",
                "verification_status": "simulation_assumption", "source_sha256": HASH,
            }
        self.read_count = 0

    @contextmanager
    def read(self):
        self.read_count += 1
        yield self

    def metadata(self, table, profile):
        assert profile == PROFILE
        return list(self.definitions.values())

    def samples(self, table, fields, start_ms, end_ms, *, limit=None, newest_first=False):
        rows = [
            row for row in self.data.get(table, []) if start_ms <= int(row["timestamp"]) < end_ms
        ]
        rows.sort(key=lambda row: (int(row["timestamp"]), row["id"]), reverse=newest_first)
        return iter({
            name: row[name] for name in dict.fromkeys(("timestamp", "id", *fields))
        } for row in rows[:limit])


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.source = MemorySource()
        self.now = BASE + 10_800_000
        self.registry = ToolRegistry()
        register_telemetry_tools(self.registry, source=self.source, clock_ms=lambda: self.now)
        self.executor = ToolExecutor(self.registry)

    async def tool(self, name, table=OVERHEAT, **kwargs):
        return await self.executor.execute(ToolCall("test", name, json.dumps({"table_name": table, **kwargs})))

    async def test_three_hour_window_pages_and_minute_summary(self):
        self.source.data[OVERHEAT] = [sample(i) for i in range(3600)]
        request = {"mode": "window", "fields": ["motor_temp", "current_actual"], "limit": 1000}
        first = await self.tool("query_telemetry", **request, last_seconds=10_800)
        self.assertFalse(first.is_error, first.content)
        original_window = first.details["requested_range"]
        self.assertEqual((original_window["start_ms"], original_window["end_ms"]), (BASE, self.now))
        self.assertEqual(first.details["coverage"]["points"], 3600)
        self.assertEqual(first.details["coverage"]["expected_points"], 3600)
        self.assertEqual(first.details["coverage"]["status"], "complete")
        self.assertEqual(first.details["page"]["returned_points"], 1000)
        model_page = json.loads(first.content)
        self.assertTrue(model_page["content_truncated"])
        self.assertLessEqual(len(first.content.encode("utf-8")), MODEL_CONTENT_BYTES)
        self.now += 15_000
        model_next = await self.tool("query_telemetry", **request, cursor=model_page["page"]["next_cursor"])
        self.assertEqual(model_next.details["data"][0]["id"], model_page["data"][-1]["id"] + 1)
        cursor = first.details["page"]["next_cursor"]
        ids = [point["id"] for point in first.details["data"]]
        page_sizes = [len(ids)]
        while cursor:
            page = await self.tool("query_telemetry", **request, cursor=cursor)
            self.assertFalse(page.is_error, page.content)
            self.assertEqual(page.details["requested_range"], original_window)
            self.assertEqual(page.details["coverage"]["points"], 3600)
            ids.extend(point["id"] for point in page.details["data"])
            page_sizes.append(len(page.details["data"]))
            cursor = page.details["page"]["next_cursor"]
        self.assertEqual(page_sizes, [1000, 1000, 1000, 600])
        self.assertEqual(ids, list(range(1, 3601)))
        summary = await self.tool(
            "summarize_telemetry", start_ms=BASE, end_ms=original_window["end_ms"],
            fields=["motor_temp"], bucket_seconds=60,
        )
        self.assertFalse(summary.is_error, summary.content)
        self.assertEqual(summary.details["bucket_count"], 180)
        self.assertTrue(all(item["coverage"]["points"] == 20 for item in summary.details["buckets"]))
        self.assertEqual(summary.details["buckets"][0]["signals"]["motor_temp"]["unit"], "degC")
        self.assertLessEqual(len(summary.content.encode("utf-8")), MODEL_CONTENT_BYTES)

    async def test_raw_peak_and_stable_order_with_duplicate_timestamps(self):
        self.source.data[OVERVOLTAGE] = [
            sample(3, dc_voltage=800), sample(0), sample(2, dc_voltage=721),
            sample(2, id=25, dc_voltage=721), sample(1, dc_voltage=666),
        ]
        result = await self.tool("query_telemetry", OVERVOLTAGE, fields=["dc_voltage", "timestamp"])
        points = result.details["data"]
        self.assertEqual([point["id"] for point in points], [1, 2, 3, 25, 4])
        self.assertEqual(result.details["values_are"], "stored")
        peak = max(points, key=lambda point: point["values"]["dc_voltage"])
        self.assertEqual(peak["values"]["dc_voltage"], 800)
        self.assertEqual(peak["timestamp_ms"] - points[0]["timestamp_ms"], 9000)
        self.assertEqual(peak["values"]["timestamp"], str(BASE + 9000))
        self.assertTrue(peak["time_beijing"].endswith("+08:00"))
        self.assertIn(str(BASE + 9000), peak["ref"])
        query = {
            "mode": "window", "fields": ["dc_voltage"], "limit": 3,
            "start_ms": BASE, "end_ms": BASE + 12000,
        }
        first = (await self.tool("query_telemetry", OVERVOLTAGE, **query)).details
        second = (await self.tool("query_telemetry", OVERVOLTAGE, **query, cursor=first["page"]["next_cursor"])).details
        self.assertEqual([point["id"] for point in second["data"]], [25, 4])

    async def test_history_cursor_rejects_changed_window_and_restarts_with_backfill(self):
        self.source.data[OVERHEAT] = [sample(i) for i in (0, 2, 3, 4)]
        query = dict(mode="window", fields=["motor_temp"], start_ms=BASE, end_ms=BASE + 15000, limit=2)
        first = await self.tool("query_telemetry", **query)
        cursor = json.loads(first.content)["page"]["next_cursor"]
        # A new row outside the fixed range has no effect on this cursor.
        self.source.data[OVERHEAT].append(sample(5))
        second = await self.tool("query_telemetry", **query, cursor=cursor)
        self.assertEqual([row["id"] for row in second.details["data"]], [4, 5])
        self.source.data[OVERHEAT].append(sample(1, id=99))
        stale = await self.tool("query_telemetry", **query, cursor=cursor)
        self.assertTrue(stale.is_error)
        self.assertEqual(stale.details["status"], "stale_cursor")
        # A source with the MySQL keyset fast path must reject the same revision change.
        with patch.object(self.source, "history_page", create=True, return_value=((5, 99), [])) as fetch:
            stale = await self.tool("query_telemetry", **query, cursor=cursor)
            self.assertEqual(stale.details["status"], "stale_cursor")
            self.assertEqual(fetch.call_args.kwargs["after"], (BASE + 6000, 3))
        fresh = await self.tool("query_telemetry", **{**query, "limit": 10})
        self.assertEqual([row["id"] for row in fresh.details["data"]], [1, 99, 3, 4, 5])
        self.assertEqual(fresh.details["coverage"]["status"], "complete")
        before_purge = await self.tool("query_telemetry", **query)
        self.source.data[OVERHEAT].pop(0)
        expired = await self.tool("query_telemetry", **query, cursor=before_purge.details["page"]["next_cursor"])
        self.assertEqual(expired.details["status"], "stale_cursor")

    async def test_metadata_default_includes_reserved_definition(self):
        result = await self.tool("get_signal_metadata")
        self.assertEqual(set(result.details["metadata"]), {*FIELDS, "actual_solution"})
        self.assertIn("actual_solution", json.loads(result.content)["field_names"])

    async def test_live_latch_check_reports_missing_evidence(self):
        from live_telemetry import _audit_latch

        result = await _audit_latch(self.executor, {"events": []})
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "not_verified")

    async def test_duplicate_conflicting_threshold_and_nonfinite_sample(self):
        self.source.definitions["motor_temp"]["canonical_unit"] = "°C"
        self.source.data[OVERHEAT] = [sample(i, motor_temp=120) for i in range(4)]
        self.source.data[OVERHEAT].append(sample(1, id=33, motor_temp=float("inf")))
        raw = await self.tool("query_telemetry", fields=["motor_temp"])
        self.assertFalse(raw.is_error, raw.content)
        point = next(point for point in json.loads(raw.content)["data"] if point["id"] == 33)
        self.assertFalse(point["valid"]["motor_temp"])
        stats = await self.tool(
            "summarize_telemetry", start_ms=BASE, end_ms=BASE + 12000,
            fields=["motor_temp"], bucket_seconds=12,
            thresholds=[dict(field="motor_temp", operator=">", value=100, unit="degC", source="fixture")],
        )
        self.assertFalse(stats.is_error, stats.content)
        signal = stats.details["buckets"][0]["signals"]["motor_temp"]
        self.assertEqual((signal["valid_points"], signal["invalid_points"]), (4, 1))
        self.assertEqual(signal["threshold_duration"]["total_ms"], 3000)

    async def test_duplicate_codes_and_clear_on_left_edge(self):
        self.source.data[BLOCKED] = [
            sample(0), sample(1, fault_code="F07900"), sample(1, id=20, fault_code="F07900"),
            sample(2, fault_code="F07900"), sample(3),
        ]
        async def events(start):
            return (await self.tool("find_code_events", BLOCKED, start_ms=start, end_ms=BASE + 12000)).details

        result = await events(BASE)
        self.assertEqual((result["event_count"], result["new_trigger_count"]), (1, 1))
        self.assertEqual(result["events"][0]["clear"]["timestamp_ms"], BASE + 9000)
        self.assertEqual(result["events"][0]["confirmed_duration_in_window_ms"], 3000)
        edge = await events(BASE + 9000)
        self.assertEqual((edge["event_count"], edge["new_trigger_count"]), (1, 0))
        self.assertEqual(edge["events"][0]["clear"]["timestamp_ms"], BASE + 9000)
        self.assertEqual(edge["events"][0]["confirmed_duration_in_window_ms"], 0)
        self.source.data[BLOCKED][2]["fault_code"] = "0"
        conflict = await events(BASE)
        self.assertEqual((conflict["conflicting_code_timestamps"], conflict["new_trigger_count"]), (1, 0))
        with_conflicting_prior = await events(BASE + 6000)
        self.assertEqual(with_conflicting_prior["events"][0]["start_boundary"], "unknown_invalid_predecessor")
        self.assertEqual(with_conflicting_prior["new_trigger_count"], 0)

    async def test_valid_zero_invalid_temperature_and_threshold_breaks(self):
        self.source.data[OVERHEAT] = [
            sample(i, air_intake_temp=temp)
            for i, temp in [(0, -200), (1, None), (2, 0), (3, 20), (4, 30), (6, 40), (7, 50), (8, None), (9, 60)]
        ]
        self.source.definitions["air_intake_temp"]["scale_factor"] = Decimal(2)
        self.source.definitions["air_intake_temp"]["offset_value"] = Decimal(1)
        window = {"start_ms": BASE, "end_ms": BASE + 30_000}
        fields = ["air_intake_temp", "speed_actual", "current_actual"]
        raw = await self.tool("query_telemetry", **window, fields=fields, mode="window")
        points = raw.details["data"]
        self.assertEqual([item["values"]["air_intake_temp"] for item in points[:3]], [-200, None, 0])
        self.assertEqual([item["valid"]["air_intake_temp"] for item in points[:3]], [False, False, True])
        self.assertTrue(points[0]["valid"]["current_actual"])
        self.assertTrue(points[0]["valid"]["speed_actual"])
        self.assertEqual(raw.details["coverage"]["missing_available_points"], 1)
        self.assertEqual(raw.details["coverage"]["gaps"][0]["start_ms"], BASE + 15_000)
        threshold = {"field": "air_intake_temp", "operator": ">", "value": 30, "unit": "°C", "source": "test"}
        stats = await self.tool(
            "summarize_telemetry", **window, fields=fields, bucket_seconds=30, thresholds=[threshold],
        )
        self.assertFalse(stats.is_error, stats.content)
        signals = stats.details["buckets"][0]["signals"]
        air = signals["air_intake_temp"]
        self.assertEqual((air["invalid_points"], air["valid_points"]), (3, 6))
        self.assertEqual((air["min"]["value"], air["max"]["value"]), (1, 121))
        self.assertEqual(air["max"]["timestamp_ms"], BASE + 27_000)
        self.assertEqual(air["mean"], (1 + 41 + 61 + 81 + 101 + 121) / 6)
        self.assertAlmostEqual(air["rate_per_second"], 120 / 21)
        duration = air["threshold_duration"]
        self.assertEqual((duration["total_ms"], duration["longest_continuous_ms"]), (6000, 3000))
        self.assertEqual(duration["segments"], [
            {"start_ms": BASE + 9000, "end_ms": BASE + 12000},
            {"start_ms": BASE + 18000, "end_ms": BASE + 21000},
        ])
        self.assertEqual(signals["current_actual"]["mean"], 0)
        self.assertEqual(stats.details["calculation"]["thresholds"][0]["unit"], "degC")
        split = await self.tool(
            "summarize_telemetry", **window, fields=fields, bucket_seconds=10, thresholds=[threshold],
        )
        self.assertEqual(
            [item["signals"]["air_intake_temp"]["threshold_duration"]["total_ms"] for item in split.details["buckets"]],
            [1000, 4000, 1000],
        )
        bad_unit = await self.tool(
            "summarize_telemetry", **window, fields=fields, bucket_seconds=30,
            thresholds=[{**threshold, "unit": "V"}],
        )
        self.assertEqual(json.loads(bad_unit.content)["status"], "invalid_threshold_unit")

    async def test_event_latch_continuation_gap_and_other_codes(self):
        self.source.data[BLOCKED] = [
            sample(i, fault_code=code, alarm_code=alarm) for i, code, alarm in [
                (0, "0", "0"), (1, "F07900", "0"), (2, "F07900", "A07910"),
                (3, "0", "A07910"), (4, "F07900", "0"), (6, "F07900", "0"),
                (7, "F07011", "A07910"), (8, "F07011", "A07910"),
            ]
        ]
        events = await self.tool(
            "find_code_events", BLOCKED, target_code="F07900", start_ms=BASE, end_ms=BASE + 12000,
        )
        self.assertEqual(events.details["new_trigger_count"], 1)
        event, = events.details["events"]
        self.assertEqual(event["trigger"]["timestamp_ms"], BASE + 3000)
        self.assertEqual(event["clear"]["timestamp_ms"], BASE + 9000)
        self.assertEqual(event["observed_trigger_to_clear_ms"], 6000)
        self.assertEqual(event["confirmed_duration_in_window_ms"], 3000)
        continuation = await self.tool(
            "find_code_events", BLOCKED, start_ms=BASE + 6000, end_ms=BASE + 12000,
        )
        self.assertEqual(continuation.details["new_trigger_count"], 0)
        self.assertTrue(continuation.details["events"][0]["crosses_window_start"])
        self.assertIsNone(continuation.details["events"][0]["trigger"])
        after_gap = await self.tool(
            "find_code_events", BLOCKED, start_ms=BASE + 12000, end_ms=BASE + 27000,
        )
        gap, unknown, open_event = after_gap.details["events"]
        self.assertEqual(after_gap.details["new_trigger_count"], 1)
        self.assertEqual(gap["end_boundary"], "sampling_gap")
        self.assertEqual(unknown["start_boundary"], "unknown_sampling_gap")
        self.assertEqual(unknown["end_boundary"], "code_change")
        self.assertIsNone(unknown["clear"])
        self.assertTrue(open_event["crosses_window_end_unknown"])
        self.assertEqual(open_event["start_boundary"], "code_change")
        self.assertEqual(after_gap.details["code_changes"][0]["to"], "F07011")
        alarms = await self.tool(
            "find_code_events", BLOCKED, code_field="alarm_code", target_code="A07910",
            start_ms=BASE, end_ms=BASE + 27000,
        )
        self.assertEqual(alarms.details["new_trigger_count"], 2)
        self.assertTrue(all(event["code"] == "A07910" for event in alarms.details["events"]))
        self.source.data[BLOCKED] = [sample(1, fault_code="F07900"), sample(2, fault_code="F07900")]
        unknown_start = await self.tool(
            "find_code_events", BLOCKED, start_ms=BASE + 3000, end_ms=BASE + 12000,
        )
        self.assertEqual(unknown_start.details["new_trigger_count"], 0)
        self.assertEqual(unknown_start.details["events"][0]["start_boundary"], "unknown_missing_predecessor")
        self.assertEqual(unknown_start.details["events"][0]["end_boundary"], "sampling_gap_at_end")

    async def test_invalid_source_window_field_and_version_are_distinct(self):
        malformed_args = [
            ("query_telemetry", {"table": "real_data", "fields": ["motor_temp"]}),
            ("query_telemetry", {"fields": ["actual_solution"]}),
            ("query_telemetry", {"fields": ["unknown_field"]}),
            ("query_telemetry", {"fields": ["motor_temp"], "limit": 1001}),
            ("query_telemetry", {"fields": ["motor_temp"], "mode": "window", "last_seconds": 604801}),
        ]
        for name, args in malformed_args:
            with self.subTest(args=args):
                receipt = await self.tool(name, **args)
                self.assertEqual(receipt.error_code, "invalid_arguments")
        wrong = await self.tool("get_signal_metadata", fields=["motor_temp"], profile_version="v2")
        self.assertEqual(json.loads(wrong.content)["status"], "version_error")
        self.assertEqual(self.source.read_count, 0)
        reserved = await self.tool("get_signal_metadata", fields=["actual_solution"])
        self.assertIn("actual_solution", reserved.details["metadata"])
        self.assertEqual(reserved.details["source"]["model_verification"], "dictionary_consistency_only")
        invalid_cursor = await self.tool("query_telemetry", fields=["motor_temp"], mode="window", cursor="bad")
        self.assertEqual(json.loads(invalid_cursor.content)["status"], "invalid_cursor")
        self.source.definitions["inverter_temp"]["source_sha256"] = "b" * 64
        bad_hash = await self.tool("get_signal_metadata", fields=["motor_temp"])
        self.assertEqual(json.loads(bad_hash.content)["status"], "version_error")
        self.assertIn("source_sha256", bad_hash.content)
        self.source.definitions.pop("inverter_temp")
        missing = await self.tool("get_signal_metadata", fields=["inverter_temp"])
        self.assertIn("inverter_temp", json.loads(missing.content)["message"])
        registry = ToolRegistry()
        register_telemetry_tools(registry, source=self.source, expected_source_sha256="b" * 64)
        actual = await ToolExecutor(registry).execute(ToolCall(
            "external", "get_signal_metadata", json.dumps({"table_name": OVERHEAT, "fields": ["motor_temp"]}),
        ))
        self.assertIn("deployment expectation", actual.content)

    async def test_retention_future_and_empty_coverage(self):
        cutoff = self.now - RETENTION_MS
        result = await self.tool(
            "query_telemetry", mode="window", fields=["motor_temp"],
            start_ms=cutoff - 3000, end_ms=cutoff + 6000,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.details["status"], "no_coverage")
        quality = result.details["coverage"]
        self.assertEqual(quality["unavailable_ranges"][0]["reason"], "outside_retention")
        self.assertEqual((quality["expected_points"], quality["expected_available_points"]), (3, 2))
        self.assertEqual(quality["gaps"][0]["kind"], "empty")
        future = await self.tool(
            "query_telemetry", mode="window", fields=["motor_temp"], start_ms=self.now, end_ms=self.now + 3000,
        )
        self.assertEqual(future.details["coverage"]["unavailable_ranges"][0]["reason"], "after_read_time")
        too_many_buckets = await self.tool(
            "summarize_telemetry", fields=["motor_temp"], bucket_seconds=3, last_seconds=10_800,
        )
        self.assertEqual(json.loads(too_many_buckets.content)["status"], "invalid_window")

    async def test_agent_round_trip_with_registered_tools_and_analysis_prompt(self):
        self.source.data[OVERHEAT] = [sample(0, motor_temp=42)]
        call = ToolCall("temperature", "query_telemetry", json.dumps({
            "table_name": OVERHEAT, "fields": ["motor_temp"], "limit": 1,
        }))

        def first(request):
            self.assertEqual(tuple(tool.name for tool in request.tools), TOOL_NAMES)
            self.assertEqual(request.system_prompt, TELEMETRY_ANALYSIS_PROMPT)
            self.assertIn("北京时间", request.system_prompt)
            return ModelReply(AssistantMessage(tool_calls=(call,)), "tool_calls")

        def second(request):
            receipt = request.messages[-1]
            self.assertIsInstance(receipt, ModelToolResult)
            self.assertEqual(receipt.tool_call_id, "temperature")
            self.assertFalse(receipt.is_error)
            evidence = json.loads(receipt.content)
            self.assertEqual(evidence["metadata"]["motor_temp"]["stored_unit"], "degC")
            self.assertEqual(evidence["data"][0]["values"]["motor_temp"], 42)
            self.assertEqual(evidence["coverage"]["valid_points"]["motor_temp"], 1)
            return ModelReply(AssistantMessage("42 degC"))

        model = ScriptedModel(first, second)
        result = await Agent(model, tools=self.registry, system_prompt=TELEMETRY_ANALYSIS_PROMPT).run("最近电机温度？")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "42 degC")
        self.assertEqual(len(model.requests), 2)


class MySQLConfigTests(unittest.TestCase):
    def test_only_connection_establishment_receives_one_transient_retry(self):
        class DriverError(Exception):
            pass

        config = MySQLConfig("db", "reader", "test-password", "dcma", 3306)
        source = MySQLTelemetrySource(config)
        driver = MagicMock(MySQLError=DriverError)
        with patch.dict("sys.modules", {"pymysql": driver}), patch("tuagent.telemetry.mysql.time.sleep"):
            driver.connect.side_effect = [DriverError(2003), MagicMock()]
            with source.read():
                pass
            self.assertEqual(driver.connect.call_count, 2)
            for code, attempts in ((1045, 1), (2003, 2)):
                with self.subTest(code=code):
                    driver.connect.reset_mock()
                    driver.connect.side_effect = DriverError(code, config.password)
                    with self.assertRaises(TelemetryError) as failure, source.read():
                        self.fail("A failed connection must not start a transaction")
                    self.assertEqual(driver.connect.call_count, attempts)
                    self.assertNotIn(config.password, str(failure.exception))
            driver.connect.reset_mock()
            driver.connect.side_effect = [MagicMock()]
            with self.assertRaises(TelemetryError), source.read():
                raise DriverError(2013)
            self.assertEqual(driver.connect.call_count, 1)

    def test_process_environment_precedes_repository_file_and_secrets_are_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.local"
            path.write_text("HOST=file-host\nMYSQL_USER=user\nMYSQL_PW='secret=with=equals'\nDB_NAME=dcma\nPORT=3306\n")
            with patch.dict(os.environ, {"HOST": "override"}, clear=True):
                config = MySQLConfig.from_env(path)
                self.assertEqual(config.host, "override")
                self.assertEqual(config.password, "secret=with=equals")
                self.assertNotIn(config.password, repr(config))
            with patch.dict(os.environ, {"PORT": "secret=with=equals"}, clear=True):
                with self.assertRaisesRegex(TelemetryError, "PORT") as failure:
                    MySQLConfig.from_env(path)
                self.assertNotIn(config.password, str(failure.exception))
