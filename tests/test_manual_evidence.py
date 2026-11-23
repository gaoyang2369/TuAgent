import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from fakes import ScriptedModel
from test_manual import descriptor
from test_manual_vectors import FakeEmbedder
from tuagent import (
    Agent,
    AssistantMessage,
    ModelReply,
    ModelToolResult,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
)
from tuagent.manual import EvidenceOptions, register_manual_tools
from tuagent.manual.evidence import (
    EvidenceServiceError,
    QuestionEvidence,
    select_evidence,
)
from tuagent.manual.index import write_index
from tuagent.manual.parser import Entry, Passage
from tuagent.manual.vectors import build_vectors


G120 = {
    "device_series": "G120",
    "control_unit": "TU-CONTROL_A",
    "firmware": "test firmware",
    "limit": 5,
}


def fixture(code, title, page, *passages):
    return Entry(code, title, page, passages=[Passage("title", page, title), *passages])


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.index = Path(directory.name)
        entries = [
            fixture(
                "F30001",
                "Power unit: Overcurrent",
                904,
                Passage(
                    "cause",
                    904,
                    "The power unit has detected an overcurrent condition. "
                    "- The motor is incorrectly parameterized. "
                    "- power cables exceed the maximum permissible length.",
                ),
            ),
            fixture(
                "F30002",
                "Power unit: DC link voltage overvoltage",
                904,
                Passage(
                    "cause",
                    905,
                    "The power unit has detected overvoltage in the DC link. "
                    "- motor regenerates too much energy. - line supply voltage too high.",
                ),
            ),
            fixture(
                "F07900",
                "Drive: Motor blocked",
                885,
                Passage(
                    "cross_reference",
                    885,
                    "See also: p2175 (Motor blocked speed threshold), p2177 (Motor blocked delay time)",
                ),
            ),
        ]
        write_index(self.index, descriptor(), "a" * 64, 885, 905, entries)

    def registry(self, mode, **kwargs):
        registry = ToolRegistry()
        register_manual_tools(registry, index_dir=self.index, mode=mode, **kwargs)
        return registry

    def call(self, query, **scope):
        return ToolCall(
            "manual", "search_manual", json.dumps({**G120, "query": query, **scope})
        )

    async def test_local_repairs_keep_original_code_and_scope(self):
        executor = ToolExecutor(self.registry("evidence_rules"))
        cases = (
            ("F30002 有哪些可能的原因？", "F30002", 905, "regenerates too much energy"),
            (
                "刹车时直流侧电压往上窜，手册列了哪些原因？",
                "F30002",
                905,
                "regenerates too much energy",
            ),
            ("p2177 表示什么？", "F07900", 885, "Motor blocked delay time"),
        )
        for question, code, page, phrase in cases:
            with self.subTest(question=question):
                result = await executor.execute(self.call(question))
                self.assertFalse(result.is_error, result.content)
                hits = json.loads(result.content)["hits"]
                self.assertTrue(
                    any(
                        h["code"] == code
                        and h["reference"]["pdf_page"] == page
                        and phrase in h["excerpt"]
                        for h in hits
                    ),
                    result.content,
                )
        for attribute in (
            "费用是多少元？",
            "寿命有几年？",
            "能查这台的序列号吗？",
            "昨天出现了几次？",
        ):
            result = await executor.execute(self.call("F30002 " + attribute))
            self.assertEqual(json.loads(result.content)["status"], "no_results")
        comparison = await executor.execute(
            self.call("比较 F30001 和 F30002 的上门维修价格")
        )
        self.assertEqual(json.loads(comparison.content)["status"], "no_results")
        unchanged = await ToolExecutor(self.registry("hybrid_concepts")).execute(
            self.call("p2177 表示什么？")
        )
        self.assertEqual(json.loads(unchanged.content)["status"], "no_results")

    async def test_agent_model_sees_complete_source_item(self):
        seen = []

        def inspect(request):
            seen.extend(
                json.loads(m.content)
                for m in request.messages
                if isinstance(m, ModelToolResult)
            )
            return ModelReply(AssistantMessage("Source seen."))

        call = self.call("Can overly long power cables cause an overcurrent trip?")
        model = ScriptedModel(
            ModelReply(AssistantMessage(tool_calls=(call,)), "tool_calls"), inspect
        )
        agent = Agent(model, tools=self.registry("evidence_rules"), max_turns=2)
        result = await agent.run("Find complete supporting evidence.")
        self.assertEqual(result.status, "completed")
        hit = seen[0]["hits"][0]
        self.assertEqual(
            hit["excerpt"], "power cables exceed the maximum permissible length."
        )
        self.assertEqual(hit["reference"]["pdf_page"], 904)
        self.assertEqual(hit["applicability"]["status"], "matches_supplied_scope")

    def test_numeric_item_and_rewrite_preservation(self):
        excerpt, _ = select_evidence(
            QuestionEvidence("r2124 的 12 表示什么？"),
            "Motor overtemperature",
            "A07910",
            "diagnostic_value",
            "Alarm value (r2124, interpret decimal): 11: No output current reduction. 12: Output current reduction active.",
            scoped=True,
            min_anchors=1,
        )
        self.assertEqual(excerpt, "12: Output current reduction active.")
        question = QuestionEvidence("半导体光纤断线有哪些原因？")
        fixed = question.fixed_rewrite()
        self.assertTrue(question.validate_rewrite(fixed))
        self.assertFalse(
            question.validate_rewrite(
                "What causes a short-circuit in the semiconductor fiber-optic cable?"
            )
        )
        self.assertFalse(question.validate_rewrite(fixed.replace("?", " F30002?")))
        self.assertFalse(
            question.validate_rewrite(fixed.replace("causes", "does not cause"))
        )
        for query in (
            "没有光纤断线会怎样？",
            "比较电机热度和散热片热度",
            "温度探头等 50 ms 后会怎样？",
        ):
            self.assertIsNone(QuestionEvidence(query).fixed_rewrite(), query)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"), "manual-vector extra unavailable"
    )
    async def test_fusion_stages_and_strict_service_failure(self):
        class Reranker:
            def __init__(self):
                self.calls = []

            def score(self, question, documents):
                self.calls.append((question, documents))
                return list(range(len(documents)))

        class Unavailable:
            def score(self, question, documents):
                raise EvidenceServiceError("rerank_timeout")

        fake = FakeEmbedder()
        build_vectors(self.index, fake)
        reranker = Reranker()
        query = self.call("刹车时直流侧电压往上窜，手册列了哪些原因？")
        allowed = ToolExecutor(
            self.registry(
                "evidence_rrf",
                embedder=fake,
                evidence=EvidenceOptions(reranker=reranker),
            )
        )
        reply = await allowed.execute(query)
        self.assertFalse(reply.is_error, reply.content)
        trace = reply.details["ranking_diagnostics"]
        self.assertTrue(trace["vector"][0])
        self.assertEqual(len(reranker.calls), 1)
        self.assertEqual(reranker.calls[0][0], json.loads(query.arguments)["query"])
        fused = [h["passage_id"] for h in trace["fused"][0]]
        self.assertEqual(len(fused), len(set(fused)))
        self.assertEqual(set(fused), {h["passage_id"] for h in trace["reranked"][0]})
        await allowed.execute(self.call("母线过压怎么查？", edition="01/1900"))
        self.assertEqual(len(reranker.calls), 1)
        for fallback in (False, True):
            unavailable = ToolExecutor(
                self.registry(
                    "evidence_rrf",
                    embedder=fake,
                    evidence=EvidenceOptions(reranker=Unavailable(), fallback=fallback),
                )
            )
            result = await unavailable.execute(query)
            self.assertEqual(result.is_error, not fallback)
            payload = json.loads(result.content)
            self.assertEqual(
                payload["fallback_reason"], "rerank_timeout" if fallback else None
            )
