"""Validate the Siemens 02/2023 edition and its citations through Agent.run."""

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from fakes import ScriptedModel
from tuagent import (
    Agent,
    AssistantMessage,
    ModelReply,
    ModelToolResult,
    ToolCall,
    ToolRegistry,
)
from tuagent.manual import register_manual_tools


SHA256 = "d6a1e4b9728de4d577b00dd5a81af761c7f57809afbe2557791d3870054912af"
PAGES = {
    "F07011": (870, 870, "fault"),
    "F07900": (885, 885, "fault"),
    "A07910": (887, 887, "alarm"),
    "F30002": (904, 905, "fault"),
}
SCOPE = {
    "device_series": "G120",
    "edition": "02/2023",
    "control_unit": "CU240E-2",
    "firmware": "4.7 SP14",
}
SOURCE = (
    "https://cache.industry.siemens.com/dl/files/919/109817919/att_1133702/"
    "v1/G120_CU240BE-2_list_man_0223_en-US.pdf"
)


@dataclass(frozen=True)
class PdfEvidence:
    titles: dict[str, str]
    pages: dict[int, str]


def _original(path: Path) -> PdfEvidence:
    from pypdf import PdfReader

    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, (
        "Source PDF hash mismatch"
    )
    pdf = PdfReader(path)
    cover = pdf.pages[2].extract_text()
    assert "A5E33839529" in cover and "4.7 SP14" in cover and "CU240E-2" in cover, (
        "Unexpected cover"
    )
    page_numbers = {
        page for title, cause, _ in PAGES.values() for page in (title, cause)
    }
    texts = {
        page: pdf.pages[page - 1].extract_text(extraction_mode="layout")
        for page in page_numbers
    }
    titles = {}
    for code, (page, _, _) in PAGES.items():
        heading = re.search(
            rf"^{code}(?: \([A-Z, ]+\))?\s{{2,}}([^\r\n]+)", texts[page], re.M
        )
        assert heading is not None, (
            f"{code} original title is absent on PDF page {page}"
        )
        titles[code] = heading[1].strip()
    return PdfEvidence(titles, texts)


def _call(call_id: str, name: str, **args) -> ToolCall:
    return ToolCall(call_id, name, json.dumps({**SCOPE, **args}))


async def _agent(
    index_dir: Path, calls: list[ToolCall], *, semantic: bool = True
) -> dict[str, dict]:
    seen = {}

    def read_results(request):
        assert {tool.name for tool in request.tools} == {
            "lookup_drive_code",
            "search_manual",
        }
        for message in request.messages:
            if isinstance(message, ModelToolResult):
                assert not message.is_error, message.content
                seen[message.tool_call_id] = json.loads(message.content)
        return ModelReply(AssistantMessage("Manual citations received."))

    model = ScriptedModel(
        ModelReply(AssistantMessage(tool_calls=tuple(calls)), "tool_calls"),
        read_results,
    )
    registry = ToolRegistry()
    register_manual_tools(registry, index_dir=index_dir, semantic=semantic)
    result = await Agent(model, tools=registry, max_turns=2).run(
        "Check manual meanings and original evidence."
    )
    assert result.status == "completed" and list(seen) == [call.id for call in calls], (
        "Incomplete Agent.run round trip"
    )
    return seen


def _check_reference(reference: dict, page: int) -> None:
    assert reference["document_number"] == "A5E33839529", reference
    assert reference["sha256"] == SHA256 and reference["edition"] == "02/2023", (
        reference
    )
    assert reference["section"] == "4.2" and reference["pdf_page"] == page, reference
    assert reference["url"] == f"{SOURCE}#page={page}", reference


def _on_original(excerpt: str, original: str) -> bool:
    return " ".join(excerpt.strip("…").split()) in " ".join(original.split())


async def _verify(pdf_path: Path, index_dir: Path) -> dict:
    original = _original(pdf_path)
    calls = [
        *(_call(f"lookup-{code}", "lookup_drive_code", code=code) for code in PAGES),
        *(
            _call(f"cause-{code}", "search_manual", code=code, query="原因", limit=1)
            for code in PAGES
        ),
        _call("remedy-F30002", "search_manual", code="F30002", query="核查", limit=1),
        _call("adjacent", "search_manual", code="F30002", query="undervoltage"),
        _call(
            "chinese",
            "search_manual",
            query="减速时母线电压升高，手册列了哪些可能原因",
            limit=1,
        ),
        _call("unrelated", "search_manual", query="番茄浓汤的食谱和配料"),
        _call(
            "simulator",
            "lookup_drive_code",
            code="F30002",
            device_series="G120_SIM",
            control_unit=None,
            firmware=None,
        ),
    ]
    results = await _agent(index_dir, calls)
    codes = {}
    for code, (title_page, cause_page, code_type) in PAGES.items():
        entry, causes = results[f"lookup-{code}"], results[f"cause-{code}"]
        assert entry["status"] == causes["status"] == "ok", (entry, causes)
        assert (
            entry["meaning"] == original.titles[code]
            and entry["code_type"] == code_type
        ), entry
        assert entry["applicability"]["status"] == "matches_supplied_scope", entry
        assert entry["applicability"]["firmware_versions"] == ["4.7 SP14"], entry
        assert "CU240E-2" in entry["applicability"]["control_units"], entry
        _check_reference(entry["reference"], title_page)
        (cause,) = causes["hits"]
        assert cause["code"] == code and cause["kind"] == "cause", cause
        _check_reference(cause["reference"], cause_page)
        assert _on_original(cause["excerpt"], original.pages[cause_page]), cause
        codes[code] = {
            "type": code_type,
            "title_pdf_page": title_page,
            "cause_pdf_page": cause_page,
        }

    prior_to_next = re.split(r"^F30003\b", original.pages[905], maxsplit=1, flags=re.M)
    assert len(prior_to_next) == 2, "Cannot find F30003 boundary on PDF page 905"
    cause = results["cause-F30002"]["hits"][0]
    (remedy,) = results["remedy-F30002"]["hits"]
    _check_reference(remedy["reference"], 905)
    assert remedy["code"] == "F30002" and remedy["kind"] == "remedy", remedy
    assert "regenerates" in cause["excerpt"].lower(), cause
    assert all(
        _on_original(hit["excerpt"], prior_to_next[0]) for hit in (cause, remedy)
    )
    assert results["lookup-F30002"]["passage_pages"]["cause"] == [905]
    assert results["lookup-F30002"]["passage_pages"]["remedy"] == [905]
    assert results["adjacent"]["status"] == "no_results", results["adjacent"]

    chinese = results["chinese"]
    assert (
        chinese["status"] == "ok" and chinese["retrieval_mode"] == "hybrid_concepts"
    ), chinese
    (chinese_hit,) = chinese["hits"]
    assert chinese_hit["code"] == "F30002" and chinese_hit["kind"] == "cause", (
        chinese_hit
    )
    _check_reference(chinese_hit["reference"], 905)
    assert _on_original(chinese_hit["excerpt"], prior_to_next[0]), chinese_hit
    assert (
        results["unrelated"]["status"] == "no_results"
        and not results["unrelated"]["hits"]
    )
    assert results["simulator"]["applicability"]["status"] == "candidate_only", results[
        "simulator"
    ]
    assert results["simulator"]["simulation"]["source_type"] == "simulation_design"

    fallback = (
        await _agent(
            index_dir,
            [
                _call(
                    "keyword",
                    "search_manual",
                    code="F30002",
                    query="motor regenerates energy",
                    limit=1,
                )
            ],
            semantic=False,
        )
    )["keyword"]
    assert (
        fallback["status"] == "ok" and fallback["retrieval_mode"] == "keyword_only"
    ), fallback
    _check_reference(fallback["hits"][0]["reference"], 905)
    return {
        "status": "passed",
        "sha256": SHA256,
        "agent_round_trip_calls": len(calls) + 1,
        "codes": codes,
        "F30002_remedy_pdf_page": remedy["reference"]["pdf_page"],
        "F30002_stops_before_F30003": True,
        "chinese_top_hit": chinese_hit["code"],
        "chinese_pdf_page": chinese_hit["reference"]["pdf_page"],
        "search_mode": chinese["retrieval_mode"],
        "unrelated_query": results["unrelated"]["status"],
        "fallback": fallback["retrieval_mode"],
        "simulator_applicability": results["simulator"]["applicability"]["status"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                asyncio.run(_verify(args.pdf, args.index_dir)),
                ensure_ascii=True,
                indent=2,
            )
        )
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        raise SystemExit(f"Manual acceptance failed: {exc}") from None
