import re
from collections.abc import Iterable
from dataclasses import dataclass, field


SECTION = "4.2"
SECTION_TITLE = "List of faults and alarms"
_SECTION_LINE = re.compile(r"4\.2\s+List of faults and alarms", re.I)
_TITLE = re.compile(r"^([A-Z][0-9]{5})(?: \(([A-Z, ]+)\))?[ \t]{2,}(.+)$")
_FIELD = re.compile(
    r"^(Message class|Reaction|Acknowledge|Cause|Remedy)(?::\s*|\s{2,}|$)(.*)"
)
_FIELD_KINDS = {
    "Message class": "message_class",
    "Reaction": "reaction",
    "Acknowledge": "acknowledge",
    "Cause": "cause",
    "Remedy": "remedy",
}
_SUBHEADINGS = (
    (re.compile(r"^See also\s*:", re.I), "cross_reference"),
    (re.compile(r"^(?:Fault|Alarm) value\b", re.I), "diagnostic_value"),
)


class ManualImportError(ValueError):
    pass


@dataclass(frozen=True)
class Passage:
    kind: str
    page: int
    text: str


@dataclass
class Entry:
    code: str
    title: str
    page: int
    flags: tuple[str, ...] = ()
    passages: list[Passage] = field(default_factory=list)


def _running_header(line: str) -> bool:
    return (
        bool(_SECTION_LINE.fullmatch(line))
        or line == "4 Faults and alarms"
        or bool(re.fullmatch(r"SINAMICS G120 Control Units .+", line))
        or bool(re.fullmatch(r"(?:\d+\s+)?List Manual, .*?(?:\s+\d+)?", line))
        or line.isdecimal()
    )


def parse_pages(pages: Iterable[tuple[int, str]]) -> list[Entry]:
    """Read column-aligned list entries while retaining each passage's PDF page."""
    entries: dict[str, Entry] = {}
    entry = None
    kind = None
    pieces: list[str] = []

    def flush(page: int) -> None:
        if entry is not None and kind is not None and pieces:
            entry.passages.append(Passage(kind, page, " ".join(pieces)))
        pieces.clear()

    for page, text in pages:
        for raw in text.splitlines():
            line = raw.strip()
            if not line or _running_header(line):
                continue
            heading = _TITLE.match(raw)
            if heading:
                flush(page)
                code, flags, title = heading.groups()
                entry, kind = None, None
                if code[0] in "FA":
                    if code in entries:
                        raise ManualImportError(
                            f"Duplicate code heading {code} on PDF page {page}"
                        )
                    entry = entries[code] = Entry(
                        code,
                        title,
                        page,
                        tuple(flags.replace(" ", "").split(",")) if flags else (),
                        [Passage("title", page, title)],
                    )
                continue
            if entry is None:
                continue
            marker = _FIELD.match(raw)
            if marker:
                flush(page)
                kind = _FIELD_KINDS[marker[1]]
                if marker[2].strip():
                    pieces.append(marker[2].strip())
                continue
            if line.startswith(("Product:", "Objects:")):
                continue
            for pattern, subkind in _SUBHEADINGS:
                if pattern.match(line):
                    flush(page)
                    kind = subkind
                    break
            if kind is not None:
                pieces.append(line)
        flush(page)

    checked = [
        entry
        for entry in entries.values()
        if any(passage.kind in ("cause", "remedy") for passage in entry.passages)
    ]
    if not checked:
        raise ManualImportError(
            "No fault or alarm entries with a Cause/Remedy row found in section 4.2"
        )
    if len(checked) != len(entries):
        missing = next(entry.code for entry in entries.values() if entry not in checked)
        raise ManualImportError(
            f"{missing} has no Cause/Remedy body; check the PDF page range"
        )
    return checked
