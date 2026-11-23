"""Evaluate the retained manual retrieval modes through search_manual."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import evidence_sources, grade, normalize
from tuagent import ToolCall, ToolRegistry
from tuagent.manual import EvidenceOptions, RankingOptions, register_manual_tools
from tuagent.manual.index import read_index, read_manuals
from tuagent.manual.vectors import OllamaEmbedder, corpus
from tuagent.tools import ToolExecutor


HERE = Path(__file__).parent
DATA = HERE / "questions_v2.json"
PROTOCOL = HERE / "protocol_v2.json"
ARMS = {
    "default": "hybrid_concepts",
    "frozen_rrf": "rrf",
    "rules_items": "evidence_rules",
    "fusion": "evidence_rrf",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def lf_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def label_ids(index_dir, dataset):
    by_location = defaultdict(list)
    with read_index(index_dir) as db:
        documents = corpus(db, read_manuals(db))
        for sid, (manual, row) in zip(documents.ids, documents.passages, strict=True):
            if manual.sha256 == dataset["pdf_sha256"]:
                by_location[(row["code"], row["kind"], row["page"])].append(
                    (sid, normalize(row["text"]))
                )
    labels = {}
    for case in dataset["questions"]:
        groups = []
        for group in case.get("evidence", []):
            ids = {
                sid
                for code, kind, page, phrase in group
                for sid, original in by_location[code, kind, page]
                if normalize(phrase) in original
            }
            if not ids:
                raise ValueError(
                    f"Missing PDF evidence in the frozen corpus for {case['id']}"
                )
            groups.append(ids)
        labels[case["id"]] = groups
    return labels, documents.sha256


def complete(hits, evidence_groups):
    return bool(evidence_groups) and all(hits & group for group in evidence_groups)


class CountedEmbedder:
    def __init__(self, source):
        self.source = source
        self.model = source.model
        self.counts = Counter()
        self.ms = Counter()

    def reset(self):
        self.counts.clear()
        self.ms.clear()

    def _call(self, key, callback, *args):
        self.counts[key] += 1
        start = time.perf_counter()
        try:
            return callback(*args)
        finally:
            self.ms[key] += (time.perf_counter() - start) * 1000

    def digest(self):
        return self._call("tags", self.source.digest)

    def embed(self, texts):
        return self._call("embed", self.source.embed, texts)


def reg_cases():
    original = read_json(HERE / "questions.json")
    translations = read_json(HERE / "v1_translations.json")
    cases = [
        dict(case, split="regression", id=case["id"].upper())
        for case in original["questions"]
    ]
    cases += [
        dict(case, id=case["id"] + "-en", query=translations[case["id"]], slice="en")
        for case in list(cases)
        if case["id"] in translations
    ]
    cases.extend(
        (
            {
                "id": "R01",
                "group": "regression-cause",
                "query": "F30002 有哪些可能的原因？",
                "slice": "code",
                "evidence": [
                    [["F30002", "cause", 905, "motor regenerates too much energy."]]
                ],
            },
            {
                "id": "R02",
                "group": "regression-param",
                "query": "p2177 表示什么？",
                "slice": "parameter",
                "evidence": [
                    [
                        [
                            "F07900",
                            "cross_reference",
                            885,
                            "p2177 (Motor blocked delay time)",
                        ],
                        ["F07900", "cause", 885, "time specified in p2177"],
                    ]
                ],
            },
            {
                "id": "R03",
                "group": "regression-prices",
                "query": "比较 A07910 与 F07011 的上门维修价格",
                "slice": "hard_negative",
                "status": "no_results",
            },
        )
    )
    for case in cases:
        if case["slice"] in {"domain_no_answer", "unrelated"}:
            case["slice"] = "hard_negative"
    return [dict(case, split="regression") for case in cases]


def nearest(times, fraction):
    return (
        round(sorted(times)[math.ceil(fraction * len(times)) - 1], 3) if times else None
    )


def summarize(rows):
    search = [r for r in rows if r["slice"] != "lookup"]
    completed = [r for r in search if not r["is_error"]]
    positive = [r for r in search if r["expected_status"] == "ok"]
    chinese = [r for r in positive if r["slice"] == "zh_no_code"]
    hard = [r for r in search if r["slice"] == "hard_negative"]
    lookups = [r for r in rows if r["slice"] == "lookup"]
    counts = Counter()
    for row in search:
        counts.update(row["requests"])
    return {
        "attempted": len(search),
        "completed": len(completed),
        "lookup_passed": sum(
            r["status_ok"] and (r["expected_status"] != "ok" or r["support_rank"] == 1)
            for r in lookups
        ),
        "lookup_attempted": len(lookups),
        "zh_attempted": len(chinese),
        "zh_rule_hit20": sum(r["candidate_hit20"].get("rules", False) for r in chinese),
        "zh_vector_hit20": sum(
            r["candidate_hit20"].get("vector", False) for r in chinese
        ),
        "zh_union_hit20": sum(r["candidate_hit20"]["union"] for r in chinese),
        "zh_hit1": sum(r["retrieval_rank"] == 1 for r in chinese),
        "zh_hit5": sum(r["retrieval_rank"] is not None for r in chinese),
        "zh_support5": sum(r["support_rank"] is not None for r in chinese),
        "answerable": len(positive),
        "hit1": sum(r["retrieval_rank"] == 1 for r in positive),
        "hit5": sum(r["retrieval_rank"] is not None for r in positive),
        "support5": sum(r["support_rank"] is not None for r in positive),
        "false_refusal": sum(
            r["status"] == "no_results" and not r["is_error"] for r in positive
        ),
        "excerpt_insufficient": sum(r["excerpt_insufficient"] for r in positive),
        "hard_false_accept": sum(r["false_hit"] for r in hard),
        "hard_attempted": len(hard),
        "hard_completed": sum(not r["is_error"] for r in hard),
        "wrong_code1": sum(r["wrong_code_at_1"] for r in positive),
        "wrong_page1": sum(r["wrong_page_at_1"] for r in positive),
        "wrong_code5": sum(r["wrong_code_in_5"] for r in positive),
        "wrong_page5": sum(r["wrong_page_in_5"] for r in positive),
        "scope_failures": sum(r["scope_failure"] for r in search),
        "timeouts": sum(r["timeout"] for r in search),
        "fallbacks": sum(bool(r["fallback"]) for r in search),
        "requests": dict(counts),
        "p50_ms": nearest([r["ms"] for r in search], 0.50),
        "p95_ms": nearest([r["ms"] for r in search], 0.95),
    }


async def run_arm(arm, settings, index, dataset, cases, ids, sources):
    mode = ARMS[arm]
    evidence = EvidenceOptions()
    counted = None
    if mode in {"rrf", "evidence_rrf"}:
        counted = CountedEmbedder(
            OllamaEmbedder(
                settings["ollama_url"],
                settings["embedding_model"],
                settings["embedding_timeout"],
                threads=settings["embedding_threads"],
            )
        )
    registry = ToolRegistry()
    register_manual_tools(
        registry,
        index_dir=index,
        mode=mode,
        evidence=evidence,
        embedder=counted,
        ranking=RankingOptions(),
        trace_candidates=True,
    )
    executor = ToolExecutor(registry)

    async def invoke(case):
        if counted:
            counted.reset()
        tool = case.get("tool", "search_manual")
        args = dict(dataset["scope"])
        if tool == "search_manual":
            args.update(query=case["query"], section=dataset["section"], limit=5)
        args.update(case.get("args", {}))
        start = time.perf_counter()
        result = await executor.execute(
            ToolCall(case["id"], tool, json.dumps(args, ensure_ascii=False))
        )
        ms = round((time.perf_counter() - start) * 1000, 3)
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            payload = {"status": result.error_code, "message": result.content}
        return result, payload, ms

    warmup, warm_payload, warm_ms = await invoke(
        {"id": "warm", "query": "减速时直流母线过压有哪些原因？"}
    )
    rows = []
    for case in cases:
        result, payload, ms = await invoke(case)
        trace = (result.details or {}).get("ranking_diagnostics", {})
        routes = {
            stage: {
                h["passage_id"] for group in trace.get(stage, []) for h in group[:20]
            }
            for stage in ("rules", "vector")
        }
        routes["union"] = set().union(*routes.values())
        required = ids[case["id"]]
        scored = grade(case, payload, dataset, sources)
        if case.get("tool") != "lookup_drive_code":
            scored["retrieval_rank"] = next(
                (
                    n
                    for n in range(1, min(5, len(scored["hits"])) + 1)
                    if complete(
                        {
                            h.get("passage_id")
                            for h in scored["hits"][:n]
                            if h["assessment"]["scope_ok"]
                        },
                        required,
                    )
                ),
                None,
            )
            scored["excerpt_insufficient"] = (
                scored["retrieval_rank"] is not None and scored["support_rank"] is None
            )
        failures = [
            *trace.get("failures", {}).values(),
            payload.get("vector_error", ""),
        ]
        requests = dict(counted.counts) if counted else {}
        rows.append(
            {
                "id": case["id"],
                "query": case.get("query"),
                "group": case["group"],
                "slice": case["slice"],
                "ms": ms,
                "is_error": result.is_error,
                "timeout": any("timeout" in code for code in failures),
                "fallback": payload.get("fallback_reason"),
                "requests": requests,
                "http_ms": {
                    k: round(v, 3) for k, v in (counted.ms if counted else {}).items()
                },
                "candidate_hit20": {
                    route: complete(found, required) for route, found in routes.items()
                },
                "trace": trace,
                **scored,
            }
        )
    return {
        "warmup": {
            "ms": warm_ms,
            "error": warmup.is_error,
            "status": warm_payload.get("status"),
        },
        "summary": summarize(rows),
        "cases": rows,
    }


def settings_from_args(args):
    return {
        "depth": 20,
        "min_anchors": 2,
        "ollama_url": os.environ.get("OLLAMA_BASE_URL", "http://10.108.13.254:11434"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:8b"),
        "embedding_timeout": 10.0,
        "embedding_threads": args.embedding_threads,
    }


async def run(args, dataset, cases, ids, sources, arms, settings):
    reports = {}
    for arm in arms:
        report = await run_arm(
            arm, settings, args.index_dir, dataset, cases, ids, sources
        )
        reports[arm] = report
        print(json.dumps({"arm": arm, **report["summary"]}), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("dev", "regression", "public-v2", "new"), default="dev"
    )
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS))
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--embedding-threads", type=int, default=24)
    parser.add_argument("--index-dir", type=Path, default=Path(".manual/index"))
    parser.add_argument(
        "--pdf", type=Path, default=Path(".manual/g120-cu240be-2-0223-en.pdf")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    dataset = read_json(DATA)
    if (
        lf_sha(DATA) != protocol["dataset_sha256_lf"]
        or file_sha(args.pdf) != protocol["pdf_sha256"]
    ):
        parser.error("Frozen dataset or PDF checksum mismatch")
    if args.split == "regression":
        dataset["questions"] = reg_cases()
    if args.split == "new":
        dataset["questions"] = read_json(HERE / "section12_questions.json")
    selected = "blind" if args.split == "public-v2" else args.split
    cases = [
        c
        for c in dataset["questions"]
        if c["split"] == selected and (not args.case or c["id"] in args.case)
    ]
    if not cases:
        parser.error("No requested cases")
    ids, corpus_sha = label_ids(args.index_dir, dataset)
    if corpus_sha != protocol["corpus_sha256"]:
        parser.error("The SQLite corpus is outside the frozen benchmark")
    sources = evidence_sources(args.index_dir, dataset)
    settings = settings_from_args(args)
    arms = args.arms or ["rules_items"]
    if args.out.exists():
        parser.error("Choose a new output path")
    report = {
        "split": args.split,
        "dataset_sha256": lf_sha(
            HERE / "section12_questions.json" if args.split == "new" else DATA
        ),
        "settings": settings,
        "scope": dataset["scope"],
        "arms": asyncio.run(run(args, dataset, cases, ids, sources, arms, settings)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
