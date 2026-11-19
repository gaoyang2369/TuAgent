import argparse
import hashlib
import io
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import ManualEdition
from .index import write_index
from .parser import SECTION, SECTION_TITLE, ManualImportError, parse_pages


def _outline_items(outline):
    for item in outline:
        if isinstance(item, list):
            yield from _outline_items(item)
        else:
            yield item


def _outline_pages(pdf: Any) -> tuple[int, int]:
    first = None
    for bookmark in _outline_items(pdf.outline):
        title = " ".join(bookmark.title.split())
        if title.casefold() == f"{SECTION} {SECTION_TITLE}".casefold():
            first = pdf.get_destination_page_number(bookmark) + 1
        elif (
            first is not None
            and not title.startswith(f"{SECTION}.")
            and re.match(r"^(?:[0-9]+(?:\.[0-9]+)*|[A-Z])\s", title)
        ):
            next_page = pdf.get_destination_page_number(bookmark) + 1
            if next_page > first:
                return first, next_page - 1
    if first is None or first < 1:
        raise ManualImportError(
            "Section 4.2 bookmark unavailable; provide --pages FIRST:LAST"
        )
    return first, len(pdf.pages)


def _extract_pdf(data: bytes, pages: tuple[int, int] | None):
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError:
        raise ManualImportError(
            'Install the manual extra with pip install -e ".[manual]"'
        ) from None
    try:
        pdf = PdfReader(io.BytesIO(data))
        first, last = pages or _outline_pages(pdf)
        if not 1 <= first <= last <= len(pdf.pages):
            raise ManualImportError(
                f"PDF page range must be within 1..{len(pdf.pages)}"
            )
        entries = parse_pages(
            (page, pdf.pages[page - 1].extract_text(extraction_mode="layout"))
            for page in range(first, last + 1)
        )
    except PyPdfError as exc:
        raise ManualImportError(f"Cannot extract PDF text: {exc}") from None
    return first, last, entries


def import_manual(
    pdf_path: Path,
    index_dir: Path,
    edition: ManualEdition,
    *,
    expected_sha256: str | None = None,
    pages: tuple[int, int] | None = None,
) -> dict:
    """Validate and parse the local PDF before publishing its edition in one transaction."""
    data = pdf_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256.lower():
        raise ManualImportError(f"PDF SHA-256 mismatch (actual {sha256})")
    first, last, entries = _extract_pdf(data, pages)
    return write_index(index_dir, edition, sha256, first, last, entries)


def _page_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "Use FIRST:LAST with one-based inclusive PDF pages"
        )
    return int(match[1]), int(match[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a PDF fault and alarm list into a local TuAgent index."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--document-number", required=True)
    parser.add_argument("--edition", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--device-series", required=True)
    parser.add_argument(
        "--control-unit",
        action="append",
        required=True,
        help="Repeat for each listed control unit",
    )
    parser.add_argument(
        "--firmware",
        action="append",
        required=True,
        help="Repeat for each listed firmware version",
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--sha256", help="Known SHA-256 of the input PDF")
    parser.add_argument(
        "--pages",
        type=_page_range,
        help="Override the section 4.2 bookmark with FIRST:LAST",
    )
    args = parser.parse_args()
    try:
        edition = ManualEdition(
            document_number=args.document_number,
            edition=args.edition,
            language=args.language,
            device_series=args.device_series,
            control_units=args.control_unit,
            firmware_versions=args.firmware,
            source_url=args.source_url,
        )
        report = import_manual(
            args.pdf,
            args.index_dir,
            edition,
            expected_sha256=args.sha256,
            pages=args.pages,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Manual import failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
