import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fakes import ScriptedModel
from test_manual import FIXTURE_PAGES, descriptor
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
from tuagent.manual.index import read_index, read_manuals, write_index
from tuagent.manual.parser import parse_pages
from tuagent.manual.vectors import (
    DOCUMENT_TEMPLATE,
    QUERY_TEMPLATE,
    VECTOR_FILENAME,
    VectorError,
    build_vectors,
    corpus,
)


class FakeEmbedder:
    model = "test-embed:fixed"
    revision = "a" * 64
    failing = False
    query_vector = (1, 0, 0)

    def __init__(self):
        self.calls = []

    def digest(self):
        return self.revision

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.failing:
            raise VectorError("embedding_timeout", "fixture timeout")
        vectors = []
        for text in texts:
            if text.startswith("Instruct:"):
                vectors.append(self.query_vector)
            elif "regenerates energy" in text:
                vectors.append((6, 1, 0))
            elif "Line supply voltage can be high" in text:
                vectors.append((2, 1, 0))
            elif "the fixture temperature" in text.lower():
                vectors.append((0, 1, 0))
            else:
                vectors.append((0, 0, 2))
        return vectors


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "manual-vector extra unavailable"
)
class ManualVectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.embedder = FakeEmbedder()
        self.install()

    def install(self, edition=None):
        entries = parse_pages(FIXTURE_PAGES)
        cause = next(
            p
            for e in entries
            if e.code == "F30002"
            for p in e.passages
            if p.kind == "cause"
        )
        target = next(e for e in entries if e.code == "F30002")
        target.passages.append(cause)
        write_index(self.root, edition or descriptor(), "a" * 64, 10, 12, entries)

    def executor(self, mode, **options):
        registry = ToolRegistry()
        register_manual_tools(
            registry, index_dir=self.root, mode=mode, embedder=self.embedder, **options
        )
        return ToolExecutor(registry)

    async def call(self, mode, query, *, fallback=False, **args):
        receipt = await self.executor(mode, fallback_to_rules=fallback).execute(
            ToolCall(
                "t1",
                "search_manual",
                json.dumps({"device_series": "G120", "query": query, **args}),
            ),
        )
        return json.loads(receipt.content), receipt.is_error

    def test_snapshot_identity_templates_and_interrupted_publish(self):
        first = build_vectors(self.root, self.embedder, batch_size=3)
        self.assertEqual(first["status"], "built")
        self.assertEqual(first["dimension"], 3)
        self.assertEqual(first["matrix_bytes"], first["passages"] * 3 * 4)
        self.assertEqual(first["document_template"], DOCUMENT_TEMPLATE)
        self.assertTrue(
            all(
                text.startswith("Code: ")
                for batch in self.embedder.calls
                for text in batch
            )
        )
        with read_index(self.root) as db:
            snapshot = corpus(db, read_manuals(db))
        same_page = [
            id_
            for id_, (_, row) in zip(snapshot.ids, snapshot.passages, strict=True)
            if (row["code"], row["kind"], row["page"]) == ("F30002", "cause", 11)
        ]
        self.assertEqual(len(set(same_page)), 2)
        path = self.root / VECTOR_FILENAME
        before = path.read_bytes()
        calls = len(self.embedder.calls)
        self.assertEqual(build_vectors(self.root, self.embedder)["status"], "unchanged")
        self.assertEqual(len(self.embedder.calls), calls)
        with patch(
            "tuagent.manual.vectors._publish", side_effect=OSError("interrupted")
        ):
            with self.assertRaisesRegex(OSError, "interrupted"):
                build_vectors(self.root, self.embedder, rebuild=True)
        self.assertEqual(path.read_bytes(), before)
        calls = len(self.embedder.calls)
        resumed = build_vectors(self.root, self.embedder, rebuild=True)
        self.assertEqual(resumed["resumed_passages"], first["passages"])
        self.assertEqual(len(self.embedder.calls), calls)
        self.assertEqual(resumed["matrix_sha256"], first["matrix_sha256"])

    def test_invalid_embeddings_cannot_publish(self):
        cases = (
            [],
            [[0, 0, 0]],
            [["1", "0", "0"]],
            [[float("nan"), 1, 0]],
            [[float("inf"), 1, 0]],
            [[1, 0, 0], [1, 0]],
        )
        for vectors in cases:
            with (
                self.subTest(vectors=vectors),
                patch.object(self.embedder, "embed", return_value=vectors),
            ):
                with self.assertRaisesRegex(VectorError, "Embedding|embedding"):
                    build_vectors(self.root, self.embedder, batch_size=1)
                self.assertFalse((self.root / VECTOR_FILENAME).exists())

    async def test_independent_zero_postings_vector_recall_and_fusion_deduplication(
        self,
    ):
        build_vectors(self.root, self.embedder)
        for mode in ("hybrid_concepts", "vector", "rrf"):
            with self.subTest(mode=mode):
                result, error = await self.call(mode, "刹车时直流侧电压往上窜")
                self.assertFalse(error, result)
                self.assertEqual(result["requested_mode"], mode)
                if mode == "hybrid_concepts":
                    self.assertEqual(result["status"], "no_results")
                else:
                    hit = result["hits"][0]
                    self.assertEqual(
                        (hit["code"], hit["kind"], hit["reference"]["pdf_page"]),
                        ("F30002", "cause", 11),
                    )
                    self.assertEqual(
                        self.embedder.calls[-1],
                        [QUERY_TEMPLATE.format(query="刹车时直流侧电压往上窜")],
                    )
        result, _ = await self.call(
            "rrf", "motor regenerates energy", code="F30002", limit=8
        )
        ids = [hit["passage_id"] for hit in result["hits"]]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertIn("regenerates energy", result["hits"][0]["excerpt"])

    async def test_all_modes_apply_scope_and_explicit_codes_before_embedding(self):
        self.install(descriptor("fixture-2", "alt firmware", language="de"))
        build_vectors(self.root, self.embedder)
        start = len(self.embedder.calls)
        for mode in ("hybrid_concepts", "keyword_only", "vector", "rrf"):
            for query, filters, status in (
                ("F99999 overvoltage causes", {}, "no_results"),
                ("F30003 causes", {"code": "F30002"}, "ambiguous_code"),
                ("F30002 F30003 causes", {}, "ambiguous_code"),
                ("overvoltage", {"section": "8.8"}, "no_results"),
                ("overvoltage", {"device_series": "G999"}, "unsupported_series"),
                ("overvoltage", {"edition": "unknown"}, "version_not_applicable"),
                ("overvoltage", {"language": "zh"}, "version_not_applicable"),
                ("overvoltage", {"document_number": "wrong"}, "version_not_applicable"),
                ("overvoltage", {"control_unit": "wrong"}, "version_not_applicable"),
                ("overvoltage", {"firmware": "wrong"}, "version_not_applicable"),
            ):
                with self.subTest(mode=mode, filters=filters):
                    result, error = await self.call(mode, query, **filters)
                    self.assertFalse(error, result)
                    self.assertEqual(result["status"], status)
                    self.assertFalse(result.get("hits"))
        self.assertEqual(len(self.embedder.calls), start)
        result, _ = await self.call(
            "vector", "F30002 原因", language="de", firmware="alt firmware"
        )
        self.assertTrue(result["hits"])
        self.assertTrue(
            all(
                hit["code"] == "F30002" and hit["reference"]["edition"] == "fixture-2"
                for hit in result["hits"]
            )
        )
        result, _ = await self.call("hybrid_concepts", "F30002 如何排查")
        self.assertEqual(result["hits"][0]["kind"], "remedy")
        result, _ = await self.call("hybrid_concepts", "（F30002）的原因")
        self.assertEqual(result["hits"][0]["kind"], "cause")
        result, _ = await self.call(
            "hybrid_concepts", "比较 F30002 和 F30003 的原因", limit=2
        )
        self.assertEqual({hit["code"] for hit in result["hits"]}, {"F30002", "F30003"})
        result, _ = await self.call(
            "hybrid_concepts", "Compare F30002 vs F30003", limit=2
        )
        self.assertEqual([hit["kind"] for hit in result["hits"]], ["title", "title"])

    async def test_rejection_mismatch_and_opt_in_fallback(self):
        missing, error = await self.call("vector", "母线过压的原因")
        self.assertTrue(error)
        self.assertEqual(missing["vector_error"], "vector_cache_missing")
        build_vectors(self.root, self.embedder)
        self.embedder.query_vector = (-1, -1, -1)
        for mode in ("vector", "rrf"):
            result, error = await self.call(mode, "母线过压和餐厅菜价有什么关联？")
            self.assertFalse(error)
            self.assertEqual(result["status"], "no_results")
        self.embedder.query_vector = (1, 0)
        result, error = await self.call("vector", "母线过压")
        self.assertTrue(error)
        self.assertEqual(result["vector_error"], "embedding_dimension_mismatch")
        self.embedder.failing = True
        result, error = await self.call("rrf", "母线过压原因", fallback=True)
        self.assertFalse(error)
        self.assertEqual(
            (
                result["requested_mode"],
                result["actual_mode"],
                result["fallback_reason"],
            ),
            ("rrf", "hybrid_concepts", "embedding_timeout"),
        )
        self.embedder.failing = False
        self.embedder.revision = "b" * 64
        result, error = await self.call("vector", "母线过压")
        self.assertTrue(error)
        self.assertEqual(result["vector_error"], "embedding_model_mismatch")
        self.install(descriptor("fixture-new"))
        result, error = await self.call("vector", "母线过压")
        self.assertTrue(error)
        self.assertEqual(result["vector_error"], "vector_corpus_mismatch")

    async def test_agent_sees_original_excerpt_page_and_scope(self):
        build_vectors(self.root, self.embedder)
        seen = []

        def inspect(request):
            seen.extend(
                json.loads(m.content)
                for m in request.messages
                if isinstance(m, ModelToolResult)
            )
            return ModelReply(AssistantMessage("Cited."))

        registry = ToolRegistry()
        register_manual_tools(
            registry, index_dir=self.root, mode="rrf", embedder=self.embedder
        )
        call = ToolCall(
            "dense",
            "search_manual",
            json.dumps(
                {
                    "device_series": "G120",
                    "query": "刹车时直流侧电压往上窜",
                    "limit": 1,
                    "control_unit": "TU-CONTROL_A",
                    "firmware": "test firmware",
                }
            ),
        )
        model = ScriptedModel(
            ModelReply(AssistantMessage(tool_calls=(call,)), "tool_calls"), inspect
        )
        result = await Agent(model, tools=registry, max_turns=2).run(
            "Find the evidence."
        )
        self.assertEqual(result.status, "completed")
        (payload,) = seen
        (hit,) = payload["hits"]
        self.assertEqual(payload["actual_mode"], "rrf")
        self.assertIn("motor regenerates energy", hit["excerpt"])
        self.assertEqual(hit["reference"]["pdf_page"], 11)
        self.assertEqual(hit["applicability"]["status"], "matches_supplied_scope")
