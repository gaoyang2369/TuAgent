import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..tools import ToolContext, ToolOutput, ToolRegistry
from .contracts import LookupArgs, ManualScope, SearchArgs, clean
from .index import (
    IndexedManual,
    ManualIndexError,
    candidates,
    entry_locations,
    read_entry,
    read_index,
    read_manuals,
)
from .parser import SECTION, SECTION_TITLE
from .retrieval import SearchPlan, concepts, keywords


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


def _search(
    db: sqlite3.Connection,
    manuals: list[IndexedManual],
    args: SearchArgs,
    *,
    semantic: bool,
) -> dict:
    mode = "hybrid_concepts" if semantic else "keyword_only"
    base = {"retrieval_mode": mode}
    if args.section and args.section.casefold() not in {
        SECTION,
        SECTION_TITLE.casefold(),
        f"{SECTION} {SECTION_TITLE}".casefold(),
    }:
        return {
            **base,
            "status": "no_results",
            "message": "Insufficient evidence in the requested section.",
            "hits": [],
        }
    plan = SearchPlan(args.query, semantic)
    ranked: list[tuple[float, IndexedManual, sqlite3.Row]] = []
    for manual in manuals:
        for row in candidates(db, manual.id, args.code, plan):
            score = plan.rank(
                row["code"],
                row["title"],
                row["kind"],
                row["text"],
                code_filter=args.code is not None,
            )
            if score > 0:
                ranked.append((score, manual, row))
    ranked.sort(key=lambda item: (-item[0], item[1].id, item[2]["id"]))
    if not ranked:
        return {
            **base,
            "status": "no_results",
            "message": "Insufficient relevant evidence in the scoped manuals.",
            "hits": [],
        }
    return {
        **base,
        "status": "ok",
        "matching_passages": len(ranked),
        "hits": [
            {
                "code": row["code"],
                "title": row["title"],
                "kind": row["kind"],
                "excerpt": _excerpt(row["text"], plan),
                "reference": _citation(manual, row["page"]),
                "applicability": _applicability(manual, args),
            }
            for _, manual, row in ranked[: args.limit]
        ],
    }


def _output(result: dict, *, is_error: bool = False) -> ToolOutput:
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
    return ToolOutput(content, details=result, is_error=is_error)


class _ManualTools:
    def __init__(self, index_dir: Path, semantic: bool) -> None:
        self.index_dir = index_dir
        self.semantic = semantic

    def _read(self, args: LookupArgs | SearchArgs) -> ToolOutput:
        base = (
            _lookup_base(args)
            if isinstance(args, LookupArgs)
            else {
                "retrieval_mode": "hybrid_concepts"
                if self.semantic
                else "keyword_only",
            }
        )
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
                return _output(_search(db, applicable, args, semantic=self.semantic))
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
) -> None:
    """Register read-only manual tools. The PDF and index paths stay in the application."""
    tools = _ManualTools(Path(index_dir), semantic)
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
        "Describe the symptom in Chinese or English. Supply the full code and edition when known.",
        SearchArgs,
        tools.execute,
    )
