import asyncio
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..tools import ToolContext, ToolOutput, ToolRegistry
from .contracts import LookupArgs, ManualScope, SearchArgs, clean
from .evidence import (
    EvidenceOptions,
    EvidenceRulePlan,
    EvidenceServiceError,
    QuestionEvidence,
    select_evidence,
)
from .index import (
    IndexedManual,
    ManualIndexError,
    candidates,
    entry_locations,
    passage_key,
    read_entry,
    read_index,
    read_manuals,
)
from .parser import SECTION, SECTION_TITLE
from .retrieval import (
    RankingOptions,
    RetrievalMode,
    SearchPlan,
    concepts,
    keywords,
    search_codes,
)
from .vectors import Embedder, VectorError, VectorSearch, corpus


_SIMULATOR = {
    "F07011": "模拟电机超温故障。",
    "F07900": "模拟电机堵转故障。",
    "A07910": "模拟电机超温报警。",
    "F30002": "模拟直流母线过压故障。",
}
_MAX_CONTENT_BYTES = 16 * 1024


def _series(scope: ManualScope) -> str:
    normalized = clean(scope.device_series).upper()
    return "G120" if normalized in ("G120_SIM", "SINAMICS G120") else normalized


def _control(value: str) -> str:
    return clean(value).upper().replace(" ", "_")


def _matches(manual: IndexedManual, scope: ManualScope) -> bool:
    edition = manual.edition
    return (
        (
            scope.edition is None
            or scope.edition.casefold() == edition.edition.casefold()
        )
        and (
            scope.language is None
            or scope.language.casefold() == edition.language.casefold()
        )
        and (
            scope.document_number is None
            or scope.document_number.casefold() == edition.document_number.casefold()
        )
        and (
            scope.control_unit is None
            or _control(scope.control_unit)
            in {_control(cu) for cu in edition.control_units}
        )
        and (
            scope.firmware is None
            or clean(scope.firmware).casefold()
            in {clean(firmware).casefold() for firmware in edition.firmware_versions}
        )
    )


def _applicability(manual: IndexedManual, scope: ManualScope) -> dict:
    details = manual.edition.model_dump(
        mode="json", exclude={"source_url", "document_number", "edition", "language"}
    )
    if scope.device_series.upper() == "G120_SIM":
        return {
            **details,
            "status": "candidate_only",
            "note": "G120_SIM has no confirmed control unit or firmware; this is a candidate manual.",
        }
    status = (
        "matches_supplied_scope"
        if scope.control_unit and scope.firmware
        else "document_scope_only"
    )
    return {**details, "status": status}


def _citation(manual: IndexedManual, page: int) -> dict:
    edition = manual.edition
    url = urlunsplit(urlsplit(edition.source_url)._replace(fragment=f"page={page}"))
    return {
        "source_type": "manual",
        "document_number": edition.document_number,
        "edition": edition.edition,
        "language": edition.language,
        "sha256": manual.sha256,
        "section": SECTION,
        "section_title": SECTION_TITLE,
        "pdf_page": page,
        "url": url,
    }


def _lookup_base(args: LookupArgs) -> dict:
    base: dict = {
        "code": args.code,
        "code_type": "fault" if args.code.startswith("F") else "alarm",
    }
    if args.device_series.upper() == "G120_SIM" and args.code in _SIMULATOR:
        base["simulation"] = {
            "source_type": "simulation_design",
            "scope": "G120_SIM / g120_sim_v1",
            "meaning": _SIMULATOR[args.code],
            "reference": "docs/fault-diagnosis-agent-design.md §6.2",
        }
    return base


def _lookup(
    db: sqlite3.Connection, manuals: list[IndexedManual], args: LookupArgs
) -> dict:
    base = _lookup_base(args)
    matches = []
    for manual in manuals:
        row = read_entry(db, manual.id, args.code)
        if row is not None:
            matches.append((manual, row))
    if not matches:
        return {
            **base,
            "status": "not_found",
            "message": "Exact code has no entry in the applicable editions.",
        }
    if len(matches) > 1:
        return {
            **base,
            "status": "ambiguous_version",
            "candidate_count": len(matches),
            "message": "Multiple manual editions or documents contain this code. Specify the edition or document.",
            "candidates": [
                {
                    "meaning": row["title"],
                    "reference": _citation(manual, row["page"]),
                    "applicability": _applicability(manual, args),
                }
                for manual, row in matches[:4]
            ],
        }
    manual, row = matches[0]
    pages: dict[str, list[int]] = {}
    for location in entry_locations(db, manual.id, args.code):
        pages.setdefault(location["kind"], []).append(location["page"])
    return {
        **base,
        "status": "ok",
        "meaning": row["title"],
        "message_flags": json.loads(row["flags"]),
        "applicability": _applicability(manual, args),
        "reference": _citation(manual, row["page"]),
        "passage_pages": pages,
    }


def _excerpt(text: str, plan: SearchPlan) -> str:
    words = list(re.finditer(r"\S+", text))
    if len(text) <= 280 and len(words) <= 24:
        return text
    best_score = -1
    start, end = 0, min(280, len(text))
    for index in range(0, len(words), 6):
        left = words[index].start()
        last = min(index + 23, len(words) - 1)
        right = words[last].end()
        while last > index and right - left > 280:
            last -= 1
            right = words[last].end()
        excerpt = text[left : min(right, left + 280)]
        detected = concepts(excerpt) if plan.concepts else set()
        score = 3 * len(detected & plan.concepts) + 2 * len(detected & plan.related)
        score += len(keywords(excerpt) & plan.keywords)
        if score > best_score:
            start, end, best_score = left, min(right, left + 280), score
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


Ranked = list[tuple[float, IndexedManual, sqlite3.Row]]


def _rule_rank(
    db: sqlite3.Connection,
    manuals: list[IndexedManual],
    code: str | None,
    plan: SearchPlan,
) -> Ranked:
    ranked: Ranked = []
    for manual in manuals:
        for row in candidates(db, manual.id, code, plan):
            score = plan.rank(
                row["code"],
                row["title"],
                row["kind"],
                row["text"],
                code_filter=code is not None,
            )
            if score > 0:
                ranked.append((score, manual, row))
    ranked.sort(key=lambda item: (-item[0], item[1].id, item[2]["id"]))
    return ranked


def _rrf(rules: Ranked, vectors: Ranked, options: RankingOptions) -> Ranked:
    scores: dict[str, float] = defaultdict(float)
    passages = {}
    for list_, weight in ((rules, 1.0), (vectors, options.vector_weight)):
        for rank, (_, manual, row) in enumerate(list_[: options.depth], 1):
            key = passage_key(manual, row)
            scores[key] += weight / (options.rrf_k + rank)
            passages[key] = (manual, row)
    order = sorted(
        scores,
        key=lambda key: (-scores[key], passages[key][0].id, passages[key][1]["id"]),
    )
    return [(scores[key], *passages[key]) for key in order]


def _hybrid_rank(
    rules: Ranked, vectors: Ranked, options: RankingOptions, scoped_code: bool
) -> Ranked:
    threshold = options.rrf_min_similarity
    eligible = [item for item in vectors if item[0] >= threshold]
    if not eligible and rules and not scoped_code:
        rule_leaders = {passage_key(manual, row) for _, manual, row in rules[:3]}
        eligible = [
            item
            for item in vectors[:3]
            if item[0] >= threshold - 0.05
            and passage_key(item[1], item[2]) in rule_leaders
        ]
    if eligible or (scoped_code and rules):
        return _rrf(rules, eligible, options)
    return []


def _group_vectors(vectors: Ranked, codes: tuple[str, ...]) -> list[Ranked]:
    if not codes:
        return [vectors]
    grouped: dict[str, Ranked] = defaultdict(list)
    for item in vectors:
        grouped[item[2]["code"]].append(item)
    return [grouped[code] for code in codes]


def _no_results(
    base: dict, message: str = "Insufficient relevant evidence in the scoped manuals."
) -> dict:
    return {**base, "status": "no_results", "message": message, "hits": []}


def _hit(item, args: SearchArgs, plan: SearchPlan) -> dict:
    _, manual, row = item
    return {
        "passage_id": passage_key(manual, row),
        "code": row["code"],
        "title": row["title"],
        "kind": row["kind"],
        "excerpt": _excerpt(row["text"], plan),
        "reference": _citation(manual, row["page"]),
        "applicability": _applicability(manual, args),
    }


def _trace_rank(items: Ranked, depth: int) -> list[dict]:
    return [
        {
            "passage_id": passage_key(manual, row),
            "code": row["code"],
            "kind": row["kind"],
            "pdf_page": row["page"],
            "rank": rank,
            "score": round(score, 8),
        }
        for rank, (score, manual, row) in enumerate(items[:depth], 1)
    ]


def _output(
    result: dict, *, is_error: bool = False, diagnostics: dict | None = None
) -> ToolOutput:
    payload = dict(result)
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for key in ("hits", "candidates"):
        if key not in payload:
            continue
        while len(content.encode()) > _MAX_CONTENT_BYTES and payload[key]:
            payload[key] = payload[key][:-1]
            payload["content_truncated"] = True
            content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(content.encode()) > _MAX_CONTENT_BYTES:
        # Preserve the entry heading, scope status and primary reference.
        payload.pop("passage_pages", None)
        if "meaning" in payload:
            payload["meaning"] = payload["meaning"][:320]
        if "applicability" in payload:
            scope = payload["applicability"]
            payload["applicability"] = {
                **scope,
                "control_units": scope["control_units"][:8],
                "firmware_versions": scope["firmware_versions"][:8],
            }
        payload["content_truncated"] = True
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    details = (
        {**result, "ranking_diagnostics": diagnostics}
        if diagnostics is not None
        else result
    )
    return ToolOutput(content, details=details, is_error=is_error)


class _ManualTools:
    def __init__(
        self,
        index_dir: Path,
        mode: RetrievalMode,
        embedder: Embedder | None,
        fallback_to_rules: bool,
        ranking: RankingOptions,
        evidence: EvidenceOptions,
        trace_candidates: bool,
    ) -> None:
        self.index_dir = index_dir
        self.mode = mode
        self.fallback_to_rules = fallback_to_rules
        self.ranking = ranking
        self.evidence = evidence
        self.trace_candidates = trace_candidates
        self.vectors = VectorSearch(index_dir, embedder)

    def _mode(
        self, actual: RetrievalMode | None = None, fallback_reason: str | None = None
    ) -> dict:
        return {
            "retrieval_mode": actual or self.mode,
            "requested_mode": self.mode,
            "actual_mode": actual,
            "fallback_reason": fallback_reason,
        }

    def _search(
        self,
        db: sqlite3.Connection,
        manuals: list[IndexedManual],
        applicable: list[IndexedManual],
        args: SearchArgs,
    ) -> ToolOutput:
        if args.section and args.section.casefold() not in {
            SECTION,
            SECTION_TITLE.casefold(),
            f"{SECTION} {SECTION_TITLE}".casefold(),
        }:
            return _output(
                _no_results(
                    self._mode(), "Insufficient evidence in the requested section."
                )
            )
        routed = search_codes(args.query, args.code, args.limit)
        if routed is None:
            return _output(
                {
                    **self._mode(),
                    "status": "ambiguous_code",
                    "hits": [],
                    "message": "Resolve conflicting codes, or request an explicit comparison with space for each code.",
                }
            )
        codes, query = routed
        if any(
            not any(read_entry(db, manual.id, code) for manual in applicable)
            for code in codes
        ):
            return _output(
                _no_results(
                    self._mode(), "An explicit code is absent from the scoped manuals."
                )
            )
        if self.mode in {"evidence_rules", "evidence_rrf"}:
            return self._search_evidence(db, manuals, applicable, args, codes, query)
        plan = SearchPlan(
            query,
            self.mode != "keyword_only",
            code_filter=bool(codes),
            comparison=len(codes) > 1,
        )
        base = self._mode(self.mode)
        rules = (
            [_rule_rank(db, applicable, code, plan) for code in (codes or (None,))]
            if self.mode != "vector"
            else []
        )
        ranked = rules
        diagnostics = (
            {"rules": [_trace_rank(group, 20) for group in rules]}
            if self.trace_candidates
            else None
        )
        if self.mode in {"vector", "rrf"}:
            try:
                vectors = self.vectors.rank(
                    corpus(db, manuals), {m.id for m in applicable}, codes, args.query
                )
            except VectorError as exc:
                if not self.fallback_to_rules:
                    return _output(
                        {
                            **self._mode(),
                            "status": "vector_unavailable",
                            "hits": [],
                            "vector_error": exc.reason,
                            "message": str(exc),
                        },
                        is_error=True,
                        diagnostics=diagnostics,
                    )
                base = {
                    **self._mode("hybrid_concepts", exc.reason),
                    "fallback_message": str(exc),
                }
                ranked = rules or [
                    _rule_rank(db, applicable, code, plan)
                    for code in (codes or (None,))
                ]
            else:
                groups = _group_vectors(vectors, codes)
                diagnostics = {
                    **(diagnostics or {}),
                    "vector_candidates": [
                        {
                            **_hit(item, args, plan),
                            "similarity": round(item[0], 6),
                        }
                        for group in groups
                        for item in group[:5]
                    ],
                }
                if self.trace_candidates:
                    diagnostics["vector"] = [_trace_rank(group, 20) for group in groups]
                if self.mode == "vector":
                    ranked = [
                        [
                            item
                            for item in group
                            if item[0] >= self.ranking.vector_min_similarity
                        ]
                        for group in groups
                    ]
                else:
                    ranked = [
                        _hybrid_rank(
                            rule_group, vector_group, self.ranking, bool(codes)
                        )
                        for rule_group, vector_group in zip(rules, groups, strict=True)
                    ]
        if not all(ranked):
            return _output(_no_results(base), diagnostics=diagnostics)
        balanced = (
            item for slots in zip_longest(*ranked) for item in slots if item is not None
        )
        hits = [_hit(item, args, plan) for _, item in zip(range(args.limit), balanced)]
        return _output(
            {
                **base,
                "status": "ok",
                "matching_passages": sum(map(len, ranked)),
                "hits": hits,
            },
            diagnostics=diagnostics,
        )

    def _search_evidence(
        self,
        db: sqlite3.Connection,
        manuals: list[IndexedManual],
        applicable: list[IndexedManual],
        args: SearchArgs,
        codes: tuple[str, ...],
        query: str,
    ) -> ToolOutput:
        options = self.evidence
        question = QuestionEvidence(query)
        phases: dict = {
            "rewrite": {"status": "disabled"},
            "requests": {},
            "stage_ms": {},
        }
        failures = {}
        hybrid = self.mode == "evidence_rrf"

        def measured(name, action):
            phases["requests"][name] = phases["requests"].get(name, 0) + 1
            start = time.perf_counter()
            try:
                return action()
            except (VectorError, EvidenceServiceError) as error:
                failures[name] = error.reason
                return None
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                phases["stage_ms"][name] = round(
                    phases["stage_ms"].get(name, 0) + elapsed, 3
                )

        rewrite = None
        if hybrid and options.rewrite != "none":
            phases["rewrite"] = {
                "status": "skipped_scope" if codes else "skipped_preservation"
            }
            projection = None if codes else question.fixed_rewrite()
            if projection:
                candidate = (
                    projection
                    if options.rewriter is None
                    else measured(
                        "rewrite",
                        lambda: options.rewriter.rewrite(args.query, question.terms),
                    )
                )
                if candidate and question.validate_rewrite(candidate):
                    rewrite = candidate
                    phases["rewrite"] = {"status": "applied", "query": rewrite}
                else:
                    phases["rewrite"] = {
                        "status": "service_failure"
                        if "rewrite" in failures
                        else "drift",
                        "query": candidate,
                        "used_original": "rewrite" not in failures or options.fallback,
                    }

        match_codes = codes or (None,)
        depth = options.depth
        original_plan = EvidenceRulePlan(question, bool(codes), len(codes) > 1)
        rule_groups = [
            _rule_rank(db, applicable, code, original_plan) for code in match_codes
        ]
        phases["rules_original"] = [_trace_rank(group, depth) for group in rule_groups]
        if rewrite:
            rewritten_plan = EvidenceRulePlan(
                question, bool(codes), len(codes) > 1, rewrite
            )
            rule_groups = [
                _rule_rank(db, applicable, code, rewritten_plan) for code in match_codes
            ]
        phases["rules"] = [_trace_rank(group, depth) for group in rule_groups]

        ranked = [group[:depth] for group in rule_groups]
        if hybrid:
            documents = corpus(db, manuals)
            manual_ids = {manual.id for manual in applicable}
            vectors = measured(
                "embedding",
                lambda: self.vectors.rank(documents, manual_ids, codes, args.query),
            )
            vector_groups = _group_vectors(vectors or [], codes)
            phases["vector"] = [_trace_rank(group, depth) for group in vector_groups]
            if options.english_vector and rewrite and vectors is not None:
                english = measured(
                    "english_embedding",
                    lambda: self.vectors.rank(documents, manual_ids, codes, rewrite),
                )
                english_groups = _group_vectors(english or [], codes)
                phases["english_vector"] = [
                    _trace_rank(group, depth) for group in english_groups
                ]
                bilingual = RankingOptions(depth=depth, rrf_k=20, vector_weight=1)
                vector_groups = [
                    _rrf(original, translated, bilingual)
                    for original, translated in zip(
                        vector_groups, english_groups, strict=True
                    )
                ]
            fusion = RankingOptions(depth=depth, rrf_k=20, vector_weight=1)
            ranked = [
                _rrf(rule_group, vector_group, fusion)
                for rule_group, vector_group in zip(
                    rule_groups, vector_groups, strict=True
                )
            ]
            phases["fused"] = [_trace_rank(group, len(group)) for group in ranked]
        if options.reranker is not None and any(ranked):
            flat = [item for group in ranked for item in group]

            def rerank():
                scores = options.reranker.score(
                    args.query,
                    [
                        f"{row['title']}\n{row['kind']}\n{row['text']}"
                        for _, _, row in flat
                    ],
                )
                if len(scores) != len(flat) or not all(
                    math.isfinite(score) for score in scores
                ):
                    raise EvidenceServiceError("rerank_invalid_response")
                return iter(scores)

            scores = measured("rerank", rerank)
            if scores is not None:
                ranked = [
                    sorted(
                        ((next(scores), manual, row) for _, manual, row in group),
                        key=lambda item: -item[0],
                    )
                    for group in ranked
                ]
                phases["reranked"] = [
                    _trace_rank(group, len(group)) for group in ranked
                ]

        phases["failures"] = failures
        base = self._mode(self.mode)
        if failures:
            if not options.fallback:
                return _output(
                    {
                        **self._mode(),
                        "status": "evidence_unavailable",
                        "hits": [],
                        "service_errors": failures,
                    },
                    is_error=True,
                    diagnostics=phases,
                )
            base = self._mode(
                self.mode if "embedding" not in failures else "evidence_rules",
                next(iter(failures.values())),
            )
        accepted = []
        decisions = []
        for group in ranked:
            group_hits = []
            for rank, item in enumerate(group, 1):
                _, manual, row = item
                excerpt, reason = select_evidence(
                    question,
                    row["title"],
                    row["code"],
                    row["kind"],
                    row["text"],
                    scoped=bool(codes),
                    min_anchors=options.min_anchors,
                )
                decisions.append(
                    {
                        "passage_id": passage_key(manual, row),
                        "rank": rank,
                        "code": row["code"],
                        "kind": row["kind"],
                        "pdf_page": row["page"],
                        "decision": reason,
                    }
                )
                if excerpt is not None:
                    hit = _hit(item, args, original_plan)
                    if options.complete_excerpts:
                        hit["excerpt"] = excerpt
                    group_hits.append(hit)
            accepted.append(group_hits)
        phases["decisions"] = decisions
        if not all(accepted):
            return _output(_no_results(base), diagnostics=phases)
        hits = (
            hit for slots in zip_longest(*accepted) for hit in slots if hit is not None
        )
        return _output(
            {
                **base,
                "status": "ok",
                "matching_passages": sum(map(len, accepted)),
                "hits": [hit for _, hit in zip(range(args.limit), hits)],
            },
            diagnostics=phases,
        )

    def _read(self, args: LookupArgs | SearchArgs) -> ToolOutput:
        base = _lookup_base(args) if isinstance(args, LookupArgs) else self._mode()
        try:
            with read_index(self.index_dir) as db:
                manuals = read_manuals(db) if db is not None else []
                if db is None or not manuals:
                    return _output(
                        {
                            **base,
                            "status": "empty_index",
                            "message": "Import a local manual edition first.",
                        }
                    )
                family = [
                    manual
                    for manual in manuals
                    if clean(manual.edition.device_series).upper() == _series(args)
                ]
                if not family:
                    return _output(
                        {
                            **base,
                            "status": "unsupported_series",
                            "message": "The device series has no indexed manuals.",
                        }
                    )
                applicable = [manual for manual in family if _matches(manual, args)]
                if not applicable:
                    return _output(
                        {
                            **base,
                            "status": "version_not_applicable",
                            "message": "No indexed edition matches the supplied control unit, firmware, edition or document filter.",
                        }
                    )
                if isinstance(args, LookupArgs):
                    return _output(_lookup(db, applicable, args))
                return self._search(db, manuals, applicable, args)
        except ManualIndexError as exc:
            return _output(
                {**base, "status": "index_read_error", "message": str(exc)},
                is_error=True,
            )

    async def execute(self, args: Any, context: ToolContext) -> ToolOutput:
        return await asyncio.to_thread(self._read, args)


def register_manual_tools(
    registry: ToolRegistry,
    *,
    index_dir: Path | str,
    semantic: bool = True,
    mode: RetrievalMode | None = None,
    embedder: Embedder | None = None,
    fallback_to_rules: bool = False,
    ranking: RankingOptions = RankingOptions(),
    evidence: EvidenceOptions = EvidenceOptions(),
    trace_candidates: bool = False,
) -> None:
    """Register read-only manual tools. The PDF and index paths stay in the application."""
    if mode is None:
        mode = "hybrid_concepts" if semantic else "keyword_only"
    if mode not in {
        "hybrid_concepts",
        "keyword_only",
        "vector",
        "rrf",
        "evidence_rules",
        "evidence_rrf",
    }:
        raise ValueError("Unknown manual retrieval mode")
    if not semantic and mode != "keyword_only":
        raise ValueError("semantic=False requires keyword_only")
    if mode != "evidence_rrf" and (
        evidence.rewrite != "none" or evidence.reranker is not None
    ):
        raise ValueError("Rewrite and reranking require evidence_rrf")
    tools = _ManualTools(
        Path(index_dir),
        mode,
        embedder,
        fallback_to_rules,
        ranking,
        evidence,
        trace_candidates,
    )
    registry.register(
        "lookup_drive_code",
        "Look up a complete SINAMICS fault or alarm code, including the F/A prefix and leading zeros. "
        "Returns an exact manual title, device applicability and an original page reference.",
        LookupArgs,
        tools.execute,
    )
    registry.register(
        "search_manual",
        "Find short source excerpts about symptoms, causes or checks in scoped local SINAMICS manuals. "
        "Describe the symptom in Chinese or English. Supply the edition and a single full code when known. "
        "To compare codes, name them in the query, explicitly ask for a comparison, and omit the code field.",
        SearchArgs,
        tools.execute,
    )
