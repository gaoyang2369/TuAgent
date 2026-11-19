"""Score the actual model-visible content of public manual tool calls."""

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import sqlite3
import statistics
import sys
import time
from contextlib import closing
from pathlib import Path

from tuagent import ToolCall, ToolRegistry
from tuagent.manual import register_manual_tools
from tuagent.tools import ToolExecutor


DATA = Path(__file__).parent / "questions.json"
FROZEN_DATA_SHA = "625d8b6c7e00fa1328ff32983ef8672ce6277ea4dda635c9fe9bb0a638ceb861"
BASELINE_TOOL_SHA = "6bd5bcdf9dba8047fabeb070f637fcda09b27753910105283fcd8de790fffde4"


def normalize(text):
    return " ".join(text.casefold().split())


def verify_labels(pdf_path, dataset):
    from pypdf import PdfReader

    assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == dataset["pdf_sha256"]
    pdf = PdfReader(pdf_path)
    pages = {}
    groups = {}
    ids = set()
    for case in dataset["questions"]:
        assert case["id"] not in ids, case
        ids.add(case["id"])
        assert groups.setdefault(case["group"], case["split"]) == case["split"], case
        for alternative in case.get("evidence", []):
            for code, kind, page, phrase in alternative:
                if page not in pages:
                    pages[page] = normalize(
                        pdf.pages[page - 1].extract_text(extraction_mode="layout")
                    )
                assert normalize(phrase) in pages[page], (
                    case["id"],
                    code,
                    kind,
                    page,
                    phrase,
                )
    return {"verified_cases": len(ids), "verified_pdf_pages": sorted(pages)}


def location(hit):
    return hit["code"], hit["kind"], hit["reference"]["pdf_page"]


def evidence_sources(index_dir, dataset):
    with closing(
        sqlite3.connect(
            f"{(index_dir / 'manuals.sqlite3').resolve().as_uri()}?mode=ro",
            uri=True,
        )
    ) as db:
        rows = db.execute(
            "SELECT p.code, p.kind, p.page, p.text FROM passages p "
            "JOIN manuals m ON m.id=p.manual_id WHERE m.sha256=? AND "
            "p.kind IN ('title', 'cause', 'remedy', 'diagnostic_value', 'cross_reference')",
            (dataset["pdf_sha256"],),
        ).fetchall()
    if len(rows) != dataset["retrievable_passages"]:
        raise ValueError("Index does not match the frozen evaluation corpus")
    by_location = {}
    for code, kind, page, text in rows:
        by_location.setdefault((code, kind, page), []).append(normalize(text))
    for case in dataset["questions"]:
        for group in case.get("evidence", []):
            for code, kind, page, phrase in group:
                if not any(
                    normalize(phrase) in text
                    for text in by_location.get((code, kind, page), ())
                ):
                    raise ValueError(
                        f"{case['id']} label is absent from its indexed code, kind or page"
                    )
    return by_location


def evidence_matches(item, hit, sources, *, require_phrase):
    if location(hit) != tuple(item[:3]):
        return False
    phrase = normalize(item[3])
    excerpt = normalize(hit["excerpt"].strip("…"))
    if require_phrase:
        return phrase in excerpt
    location_sources = sources[location(hit)]
    if len(location_sources) == 1:
        return True
    candidates = [text for text in location_sources if excerpt in text]
    return bool(candidates) and all(phrase in text for text in candidates)


def grade(case, payload, dataset, sources):
    expected = case.get("evidence", [])
    status = payload.get("status")
    hits = payload.get("hits", [])
    lookup = case.get("tool") == "lookup_drive_code"
    if lookup and status == "ok":
        hits = [{**payload, "kind": "title", "excerpt": payload["meaning"]}]

    accepted = [tuple(item[:3]) for group in expected for item in group]
    codes = {code for code, _, _ in accepted}
    page_pairs = {(code, page) for code, _, page in accepted}
    annotated = []
    for hit in hits:
        ref = hit["reference"]
        code, kind, page = location(hit)
        scope_ok = (
            ref["document_number"] == dataset["scope"]["document_number"]
            and ref["edition"] == dataset["scope"]["edition"]
            and ref["language"] == dataset["scope"]["language"]
            and ref["sha256"] == dataset["pdf_sha256"]
            and ref["section"] == dataset["section"]
            and hit["applicability"]["status"] == "matches_supplied_scope"
            and dataset["scope"]["control_unit"]
            in hit["applicability"]["control_units"]
            and dataset["scope"]["firmware"]
            in hit["applicability"]["firmware_versions"]
            and (not case.get("args", {}).get("code") or code == case["args"]["code"])
        )
        annotated.append(
            {
                **hit,
                "assessment": {
                    "scope_ok": scope_ok,
                    "wrong_code": bool(expected and code not in codes),
                    "wrong_page": code in codes and (code, page) not in page_pairs,
                    "expected_passage": any(
                        evidence_matches(item, hit, sources, require_phrase=False)
                        for group in expected
                        for item in group
                    ),
                    "supports_answer": any(
                        evidence_matches(item, hit, sources, require_phrase=True)
                        for group in expected
                        for item in group
                    ),
                },
            }
        )

    def completion_rank(require_phrase):
        ranks = []
        for group in expected:
            rank = next(
                (
                    i
                    for i, hit in enumerate(annotated, 1)
                    if hit["assessment"]["scope_ok"]
                    and any(
                        evidence_matches(
                            item, hit, sources, require_phrase=require_phrase
                        )
                        for item in group
                    )
                ),
                None,
            )
            if rank is None:
                return None
            ranks.append(rank)
        return max(ranks, default=None)

    best = annotated[0]["assessment"] if annotated else {}
    retrieval_rank = completion_rank(False)
    support_rank = completion_rank(True)
    expected_status = case.get("status", "ok")
    empty_scope = (
        case["slice"] == "unknown"
        or expected_status in {"version_not_applicable", "ambiguous_code"}
        or case.get("args", {}).get("section", dataset["section"]) != dataset["section"]
    )
    return {
        "status": status,
        "expected_status": expected_status,
        "status_ok": status == expected_status,
        "retrieval_rank": retrieval_rank,
        "support_rank": support_rank,
        "false_hit": not expected and bool(hits),
        "wrong_code_at_1": best.get("wrong_code", False),
        "wrong_page_at_1": best.get("wrong_page", False),
        "wrong_code_in_5": any(h["assessment"]["wrong_code"] for h in annotated),
        "wrong_page_in_5": any(h["assessment"]["wrong_page"] for h in annotated),
        "scope_failure": any(not h["assessment"]["scope_ok"] for h in annotated)
        or (empty_scope and bool(hits)),
        "excerpt_insufficient": retrieval_rank is not None and support_rank is None,
        "hits": annotated,
    }


def summarize(rows):
    lookup = [row for row in rows if row["slice"] == "lookup"]
    searches = [row for row in rows if row["slice"] != "lookup"]
    answerable = [row for row in searches if row["expected_status"] == "ok"]
    negatives = [row for row in searches if row["expected_status"] != "ok"]
    no_answer = [
        row for row in negatives if row["slice"] in {"domain_no_answer", "unrelated"}
    ]
    chinese = [row for row in answerable if row["slice"] == "zh_no_code"]
    latency = sorted(row["ms"] for row in searches)
    return {
        "lookup_passed": sum(
            r["status_ok"] and (r["expected_status"] != "ok" or r["support_rank"] == 1)
            for r in lookup
        ),
        "lookup_total": len(lookup),
        "answerable": len(answerable),
        "hit_at_1": sum(r["retrieval_rank"] == 1 for r in answerable),
        "hit_at_5": sum(r["retrieval_rank"] is not None for r in answerable),
        "mrr_at_5": round(
            statistics.mean(
                1 / r["retrieval_rank"] if r["retrieval_rank"] else 0
                for r in answerable
            ),
            4,
        )
        if answerable
        else None,
        "support_at_1": sum(r["support_rank"] == 1 for r in answerable),
        "support_at_5": sum(r["support_rank"] is not None for r in answerable),
        "zh_total": len(chinese),
        "zh_hit_at_1": sum(r["retrieval_rank"] == 1 for r in chinese),
        "zh_hit_at_5": sum(r["retrieval_rank"] is not None for r in chinese),
        "zh_support_at_5": sum(r["support_rank"] is not None for r in chinese),
        "negative_total": len(negatives),
        "negative_false_hits": sum(r["false_hit"] for r in negatives),
        "no_answer_total": len(no_answer),
        "no_answer_false_hits": sum(r["false_hit"] for r in no_answer),
        "wrong_code_at_1": sum(r["wrong_code_at_1"] for r in answerable),
        "wrong_page_at_1": sum(r["wrong_page_at_1"] for r in answerable),
        "wrong_code_in_5": sum(r["wrong_code_in_5"] for r in answerable),
        "wrong_page_in_5": sum(r["wrong_page_in_5"] for r in answerable),
        "excerpt_insufficient": sum(r["excerpt_insufficient"] for r in answerable),
        "scope_failures": sum(r["scope_failure"] for r in searches),
        "negative_status_mismatches": sum(not r["status_ok"] for r in negatives),
        "errors": sum(r["is_error"] for r in rows),
        "fallbacks": sum(r["fallback_reason"] is not None for r in searches),
        "latency_samples": len(latency),
        "p95_ms": latency[math.ceil(0.95 * len(latency)) - 1] if latency else None,
    }


async def run(args, dataset, *, embedder=None):
    sources = evidence_sources(args.index_dir, dataset)
    registry = ToolRegistry()
    registration = {}
    if args.mode != "baseline":
        from tuagent.manual.retrieval import RankingOptions

        overrides = {
            key: value
            for key, value in vars(args).items()
            if key in RankingOptions.__dataclass_fields__ and value is not None
        }
        registration = {
            "mode": args.mode,
            "ranking": RankingOptions(**overrides),
            "embedder": embedder,
        }
    register_manual_tools(registry, index_dir=args.index_dir, **registration)
    executor = ToolExecutor(registry)
    rows = []

    async def invoke(case):
        tool = case.get("tool", "search_manual")
        params = dict(dataset["scope"])
        if tool == "search_manual":
            params.update(query=case["query"], limit=5, section=dataset["section"])
        params.update(case.get("args", {}))
        return await executor.execute(ToolCall(case["id"], tool, json.dumps(params)))

    warmup_ms = []
    for i in range(args.warmup):
        start = time.perf_counter()
        warmed = await invoke(
            {"id": f"warmup-{i}", "query": "DC link voltage controller operation"}
        )
        if warmed.is_error:
            raise RuntimeError(f"Warm-up failed: {warmed.content}")
        warmup_ms.append(round((time.perf_counter() - start) * 1000, 3))

    for case in dataset["questions"]:
        if args.case != case["id"] and (args.case or case["split"] != args.split):
            continue
        start = time.perf_counter()
        result = await invoke(case)
        ms = round((time.perf_counter() - start) * 1000, 3)
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            payload = {"status": result.error_code, "message": result.content}
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "slice": case["slice"],
                "query": case.get("query"),
                "ms": ms,
                "is_error": result.is_error,
                "retrieval_mode": payload.get("retrieval_mode"),
                "actual_mode": payload.get(
                    "actual_mode", payload.get("retrieval_mode")
                ),
                "fallback_reason": payload.get("fallback_reason"),
                **({"error": payload.get("message")} if result.is_error else {}),
                **(
                    {"diagnostics": (result.details or {}).get("ranking_diagnostics")}
                    if args.diagnostics
                    else {}
                ),
                **grade(case, payload, dataset, sources),
            }
        )
    if not rows:
        raise ValueError("No matching eval questions")
    return {
        "ranking": vars(registration["ranking"]) if registration else None,
        "warmup_ms": warmup_ms,
        "summary": summarize(rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path(".manual/index"))
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--case")
    parser.add_argument(
        "--mode",
        choices=("baseline", "hybrid_concepts", "vector", "rrf"),
        default="hybrid_concepts",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--vector-weight", type=float)
    parser.add_argument("--vector-min-similarity", type=float)
    parser.add_argument("--rrf-min-similarity", type=float)
    parser.add_argument("--verify-labels", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    from tuagent.manual import tools

    baseline_loaded = "mode" not in inspect.signature(register_manual_tools).parameters
    if (args.mode == "baseline") != baseline_loaded:
        parser.error(
            "Use the archived 79696be src in PYTHONPATH only for --mode baseline"
        )
    tool_sha = hashlib.sha256(Path(tools.__file__).read_bytes()).hexdigest()
    if args.mode == "baseline" and tool_sha != BASELINE_TOOL_SHA:
        parser.error("Archived baseline tool source hash does not match")
    source = DATA.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != FROZEN_DATA_SHA:
        parser.error("Frozen evaluation set hash does not match")
    dataset = json.loads(source)
    manual_sources = Path(tools.__file__).parent
    source_hashes = {
        name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in ("index.py", "retrieval.py", "tools.py", "vectors.py")
        if (path := manual_sources / name).is_file()
    }
    result = {
        "dataset": dataset["version"],
        "dataset_sha256": hashlib.sha256(source).hexdigest(),
        "split": args.split if not args.case else None,
        "mode": args.mode,
        "baseline_commit": dataset["baseline_commit"],
        "tool_source_sha256": tool_sha,
        "retrieval_source_sha256s": source_hashes,
        "python": sys.version.split()[0],
        "pdf_sha256": dataset["pdf_sha256"],
        "scope": dataset["scope"],
        "section": dataset["section"],
        **(
            {"label_check": verify_labels(args.verify_labels, dataset)}
            if args.verify_labels
            else {}
        ),
        **asyncio.run(run(args, dataset)),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
    else:
        print(
            serialized.encode(
                sys.stdout.encoding or "utf-8", errors="backslashreplace"
            ).decode(sys.stdout.encoding or "utf-8")
        )


if __name__ == "__main__":
    main()
