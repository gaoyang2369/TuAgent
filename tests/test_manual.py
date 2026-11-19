import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fakes import ScriptedModel
from tuagent import (
    Agent,
    AssistantMessage,
    ModelReply,
    ModelToolResult,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
)
from tuagent.manual import register_manual_tools
from tuagent.manual.contracts import ManualEdition
from tuagent.manual.importer import import_manual
from tuagent.manual.index import read_index, write_index
from tuagent.manual.parser import ManualImportError, parse_pages
from tuagent.telemetry import register_telemetry_tools


FIXTURE_PAGES = (
    (
        10,
        """4.2 List of faults and alarms
F00007   Synthetic blocked motor fixture
Cause    The sample motor reached the torque limit.
         See also: A00007 for the sample alarm.
Remedy   Check whether the fixture rotates.

A00007 (N)   Synthetic motor overtemperature alarm
Cause    The fixture temperature crossed the alarm threshold.
Remedy   Check fixture ventilation.

F30002   Synthetic DC link overvoltage fixture
Reaction:    OFF2
List Manual, test edition 10
""",
    ),
    (
        11,
        """4.2 List of faults and alarms
Cause    The fixture motor regenerates energy during rapid deceleration.
         Continued explanation begins on the next page.
""",
    ),
    (
        12,
        """4.2 List of faults and alarms
         Line supply voltage can be high in this fixture.
Remedy   Check the sample braking resistor.
         See also: F30003 shows a different fixture.

F30003   Synthetic DC link undervoltage fixture
Cause    The supply voltage is below the synthetic limit.
Remedy   Check the test supply.
""",
    ),
)


def descriptor(edition="fixture-1", firmware="test firmware", **changes):
    attrs = {
        "document_number": "TUAGENT-SYNTHETIC",
        "edition": edition,
        "language": "en",
        "device_series": "G120",
        "control_units": ["TU-CONTROL_A"],
        "firmware_versions": [firmware],
        "source_url": "https://example.invalid/fixture.pdf#old-fragment",
    }
    return ManualEdition.model_validate({**attrs, **changes})


class ManualImportTests(unittest.TestCase):
    def test_paragraph_and_cross_reference_boundaries(self):
        entries = {entry.code: entry for entry in parse_pages(FIXTURE_PAGES)}
        self.assertEqual(list(entries), ["F00007", "A00007", "F30002", "F30003"])
        fault = entries["F30002"]
        self.assertEqual(fault.page, 10)
        self.assertEqual(
            [p.page for p in fault.passages if p.kind == "cause"], [11, 12]
        )
        self.assertEqual([p.page for p in fault.passages if p.kind == "remedy"], [12])
        self.assertEqual(
            [p.page for p in fault.passages if p.kind == "cross_reference"], [12]
        )
        self.assertNotIn("undervoltage", " ".join(p.text for p in fault.passages))
        self.assertEqual(entries["A00007"].flags, ("N",))
        with self.assertRaisesRegex(ManualImportError, "No fault or alarm entries"):
            parse_pages([(1, "Contents\n   F00007    Synthetic title . . . 10")])
        with self.assertRaisesRegex(ManualImportError, "no Cause/Remedy body"):
            parse_pages([FIXTURE_PAGES[0]])

    def test_reimport_identity_sha_and_transaction_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "fixture.pdf"
            path.write_bytes(b"explicit synthetic PDF fixture; parser mocked")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            index_dir = root / "index with spaces"
            entries = parse_pages(FIXTURE_PAGES)
            with patch(
                "tuagent.manual.importer._extract_pdf", return_value=(10, 12, entries)
            ) as extract:
                with self.assertRaisesRegex(ManualImportError, "SHA-256 mismatch"):
                    import_manual(
                        path, index_dir, descriptor(), expected_sha256="0" * 64
                    )
                extract.assert_not_called()
                first = import_manual(
                    path, index_dir, descriptor(), expected_sha256=digest
                )
                self.assertEqual(first["status"], "imported")
                self.assertEqual(first["entries"], 4)
                self.assertEqual(
                    import_manual(path, index_dir, descriptor())["status"], "unchanged"
                )
                path.write_bytes(b"another synthetic PDF")
                with self.assertRaisesRegex(ManualImportError, "different PDF"):
                    import_manual(path, index_dir, descriptor())
                with self.assertRaises(sqlite3.IntegrityError):
                    write_index(
                        index_dir,
                        descriptor("fixture-2"),
                        digest,
                        10,
                        12,
                        [*entries, entries[0]],
                    )
                with read_index(index_dir) as db:
                    self.assertEqual(
                        db.execute("SELECT COUNT(*) FROM manuals").fetchone()[0], 1
                    )
                    self.assertEqual(
                        db.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 4
                    )
                    with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                        db.execute("DELETE FROM manuals")
                with patch(
                    "tuagent.manual.index._publish",
                    side_effect=OSError("publish interrupted"),
                ):
                    with self.assertRaisesRegex(OSError, "publish interrupted"):
                        import_manual(path, index_dir, descriptor("fixture-2"))
                with read_index(index_dir) as db:
                    self.assertEqual(
                        db.execute("SELECT COUNT(*) FROM manuals").fetchone()[0], 1
                    )
                second = import_manual(path, index_dir, descriptor("fixture-2"))
                self.assertEqual(second["status"], "imported")
                with read_index(index_dir) as db:
                    self.assertEqual(
                        db.execute("SELECT COUNT(*) FROM manuals").fetchone()[0], 2
                    )


class ManualToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index_dir = Path(self.tmp.name) / "local index"
        self.registry = ToolRegistry()
        register_manual_tools(self.registry, index_dir=self.index_dir)
        self.executor = ToolExecutor(self.registry)

    def install(self, edition=None):
        write_index(
            self.index_dir,
            edition or descriptor(),
            "a" * 64,
            10,
            12,
            parse_pages(FIXTURE_PAGES),
        )

    async def call(self, name, executor=None, **args):
        receipt = await (executor or self.executor).execute(
            ToolCall(
                "manual-test",
                name,
                json.dumps({"device_series": "G120", **args}),
            )
        )
        self.assertFalse(receipt.is_error, receipt.content)
        return json.loads(receipt.content)

    async def test_exact_code_and_empty_index(self):
        empty = await self.call("lookup_drive_code", code="f00007")
        self.assertEqual(empty["status"], "empty_index")
        self.assertFalse(self.index_dir.exists())
        self.install()
        fault = await self.call(
            "lookup_drive_code", code=" f00007 ", device_series=" G120_SIM "
        )
        self.assertEqual((fault["code"], fault["code_type"]), ("F00007", "fault"))
        self.assertEqual(
            fault["reference"]["url"], "https://example.invalid/fixture.pdf#page=10"
        )
        self.assertEqual(fault["applicability"]["status"], "candidate_only")
        alarm = await self.call("lookup_drive_code", code="a00007")
        self.assertEqual(alarm["code_type"], "alarm")
        self.assertIn("overtemperature alarm", alarm["meaning"])
        unknown = await self.call("lookup_drive_code", code="A00009")
        self.assertEqual(unknown["status"], "not_found")
        for bad in ("00007", "F7", "F0000７", "F00070F", "C00007"):
            receipt = await self.executor.execute(
                ToolCall(
                    "bad",
                    "lookup_drive_code",
                    json.dumps({"device_series": "G120", "code": bad}),
                )
            )
            self.assertEqual(receipt.error_code, "invalid_arguments")
        sim = await self.call(
            "lookup_drive_code", code="F30002", device_series="G120_SIM"
        )
        self.assertEqual(sim["simulation"]["source_type"], "simulation_design")
        self.assertEqual(sim["reference"]["source_type"], "manual")

    async def test_edition_conflict_and_filters(self):
        self.install()
        self.install(
            descriptor(
                "fixture-2", "different firmware", control_units=["TU-CONTROL_B"]
            )
        )
        ambiguous = await self.call("lookup_drive_code", code="F30002")
        self.assertEqual(ambiguous["status"], "ambiguous_version")
        self.assertEqual(
            {row["reference"]["edition"] for row in ambiguous["candidates"]},
            {"fixture-1", "fixture-2"},
        )
        entry = await self.call(
            "lookup_drive_code",
            code="F30002",
            control_unit="tu-control b",
            firmware="DIFFERENT FIRMWARE",
        )
        self.assertEqual(entry["reference"]["edition"], "fixture-2")
        self.assertEqual(entry["applicability"]["status"], "matches_supplied_scope")
        old = await self.call("lookup_drive_code", code="F30002", edition="fixture-1")
        self.assertEqual(old["reference"]["edition"], "fixture-1")
        wrong = await self.call(
            "lookup_drive_code", code="F30002", firmware="unknown firmware"
        )
        self.assertEqual(wrong["status"], "version_not_applicable")
        wrong_family = await self.call(
            "lookup_drive_code", code="F30002", device_series="V20"
        )
        self.assertEqual(wrong_family["status"], "unsupported_series")

    async def test_bilingual_retrieval_and_keyword_fallback(self):
        self.install()
        lookup = await self.call("lookup_drive_code", code="F30002")
        self.assertEqual(lookup["reference"]["pdf_page"], 10)
        self.assertEqual(lookup["passage_pages"]["cause"], [11, 12])
        result = await self.call(
            "search_manual",
            query="减速时母线电压升高，手册列了哪些可能原因",
            section="4.2",
        )
        self.assertEqual(result["retrieval_mode"], "hybrid_concepts")
        hit = result["hits"][0]
        self.assertEqual(
            (hit["code"], hit["kind"], hit["reference"]["pdf_page"]),
            ("F30002", "cause", 11),
        )
        self.assertIn("regenerates energy", hit["excerpt"])
        self.assertNotIn("undervoltage", hit["excerpt"])
        check = await self.call("search_manual", query="核查", code="F30002")
        self.assertEqual([hit["reference"]["pdf_page"] for hit in check["hits"]], [12])
        for query in ("冻豆腐炖什么", "Tomato soup recipe"):
            irrelevant = await self.call("search_manual", query=query)
            self.assertEqual(irrelevant["status"], "no_results")
        other_section = await self.call(
            "search_manual", query="overvoltage", section="2"
        )
        self.assertEqual(other_section["status"], "no_results")
        lexical = ToolRegistry()
        register_manual_tools(lexical, index_dir=self.index_dir, semantic=False)
        fallback = await self.call(
            "search_manual", ToolExecutor(lexical), query="motor regenerates energy"
        )
        self.assertEqual(fallback["retrieval_mode"], "keyword_only")
        self.assertEqual(fallback["hits"][0]["code"], "F30002")

    async def test_agent_receives_scopes_and_citations_for_both_tools(self):
        self.install()
        register_telemetry_tools(self.registry)
        calls = (
            ToolCall(
                "code",
                "lookup_drive_code",
                '{"device_series":"G120_SIM","code":"F30002"}',
            ),
            ToolCall(
                "docs",
                "search_manual",
                '{"device_series":"G120_SIM","code":"F30002","query":"原因"}',
            ),
        )

        def acknowledge(request):
            self.assertEqual(len(request.tools), 6)
            messages = request.messages[-2:]
            self.assertTrue(
                all(
                    isinstance(message, ModelToolResult) and not message.is_error
                    for message in messages
                )
            )
            self.assertEqual(
                [message.tool_call_id for message in messages], ["code", "docs"]
            )
            lookup, search = (json.loads(message.content) for message in messages)
            self.assertIn("overvoltage", lookup["meaning"])
            self.assertEqual(lookup["applicability"]["status"], "candidate_only")
            self.assertEqual(lookup["reference"]["pdf_page"], 10)
            self.assertTrue(search["hits"][0]["reference"]["url"].endswith("#page=11"))
            self.assertIn("regenerates energy", search["hits"][0]["excerpt"])
            return ModelReply(AssistantMessage("Fixture citations received."))

        model = ScriptedModel(
            ModelReply(AssistantMessage(tool_calls=calls), "tool_calls"), acknowledge
        )
        result = await Agent(model, tools=self.registry).run(
            "Check F30002 in the synthetic manual."
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "Fixture citations received.")
