"""Snapshot the public v1 responses and pilot success/failure fixtures for revision comparison."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_manual_evidence import EvidenceTests
from test_manual_vectors import FakeEmbedder
from tuagent import ToolCall, ToolExecutor, ToolRegistry
from tuagent.manual import EvidenceOptions, register_manual_tools
from tuagent.manual.vectors import build_vectors


async def snapshot(index_dir):
    dataset = json.loads(
        Path(__file__).with_name("questions.json").read_text(encoding="utf8")
    )
    registry = ToolRegistry()
    register_manual_tools(registry, index_dir=index_dir)
    executor = ToolExecutor(registry)
    rows = {}
    for case in dataset["questions"]:
        tool = case.get("tool", "search_manual")
        args = dict(dataset["scope"])
        if tool == "search_manual":
            args.update(query=case["query"], section=dataset["section"], limit=5)
        args.update(case.get("args", {}))
        reply = await executor.execute(ToolCall(case["id"], tool, json.dumps(args)))
        rows[case["id"]] = reply.content

    fixture = EvidenceTests()
    fixture.setUp()
    try:
        embedder = FakeEmbedder()
        build_vectors(fixture.index, embedder)
        query = fixture.call("刹车时直流侧电压往上窜，手册列了哪些原因？")
        for key, call, failing, fallback, status in [
            ("hit", query, False, False, "ok"),
            (
                "no_answer",
                fixture.call("上门维修费用多少？"),
                False,
                False,
                "no_results",
            ),
            ("failure", query, True, False, "evidence_unavailable"),
            ("fallback", query, True, True, "ok"),
        ]:
            embedder.failing = failing
            registry = fixture.registry(
                "evidence_rrf",
                embedder=embedder,
                evidence=EvidenceOptions(depth=20, min_anchors=2, fallback=fallback),
            )
            reply = await ToolExecutor(registry).execute(call)
            assert json.loads(reply.content)["status"] == status, reply.content
            rows[key] = reply.content
    finally:
        fixture.doCleanups()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path(".manual/index"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(asyncio.run(snapshot(args.index_dir)), ensure_ascii=False),
        encoding="utf8",
    )
