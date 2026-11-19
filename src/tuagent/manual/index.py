import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from .contracts import ManualEdition
from .parser import SECTION, Entry, ManualImportError
from .retrieval import SearchPlan, index_terms


_INDEX_NAME = "manuals.sqlite3"
_INDEX_VERSION = 1
_SCHEMA = (
    """CREATE TABLE manuals (
        id INTEGER PRIMARY KEY, manifest TEXT NOT NULL, sha256 TEXT NOT NULL,
        first_page INTEGER NOT NULL, last_page INTEGER NOT NULL,
        document_key TEXT NOT NULL UNIQUE
    )""",
    """CREATE TABLE entries (
        manual_id INTEGER NOT NULL REFERENCES manuals(id), code TEXT NOT NULL,
        title TEXT NOT NULL, page INTEGER NOT NULL, flags TEXT NOT NULL,
        PRIMARY KEY (manual_id, code)
    ) WITHOUT ROWID""",
    """CREATE TABLE passages (
        id INTEGER PRIMARY KEY, manual_id INTEGER NOT NULL, code TEXT NOT NULL,
        page INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
        CHECK (page > 0),
        FOREIGN KEY (manual_id, code) REFERENCES entries(manual_id, code)
    )""",
    "CREATE INDEX passages_entry ON passages(manual_id, code, id)",
    """CREATE TABLE postings (
        term TEXT NOT NULL, passage_id INTEGER NOT NULL REFERENCES passages(id),
        PRIMARY KEY (term, passage_id)
    ) WITHOUT ROWID""",
)


class ManualIndexError(Exception):
    pass


@dataclass(frozen=True)
class IndexedManual:
    id: int
    edition: ManualEdition
    sha256: str
    first_page: int
    last_page: int


def _key(edition: ManualEdition) -> str:
    return json.dumps(
        [
            value.casefold()
            for value in (
                edition.document_number,
                edition.edition,
                edition.language,
                edition.device_series,
            )
        ]
    )


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def _publish(staging: Path, path: Path) -> None:
    deadline = time.monotonic() + 5
    while True:
        try:
            os.replace(staging, path)
            return
        except PermissionError as exc:
            # A Windows reader may temporarily hold the destination open.
            if (
                os.name != "nt"
                or exc.winerror not in (5, 32, 33)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.1)


def write_index(
    index_dir: Path,
    edition: ManualEdition,
    sha256: str,
    first_page: int,
    last_page: int,
    entries: list[Entry],
) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / _INDEX_NAME
    # Serialize publishers separately so no write or hot journal touches the published index.
    with closing(sqlite3.connect(index_dir / ".import.lock", timeout=15)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        with tempfile.TemporaryDirectory(
            prefix=".build-", dir=index_dir
        ) as staging_dir:
            staging = Path(staging_dir) / _INDEX_NAME
            with closing(sqlite3.connect(staging)) as db:
                if path.is_file():
                    with closing(_connect_readonly(path)) as previous:
                        previous.backup(db)
                with db:
                    status = _store_edition(
                        db, edition, sha256, first_page, last_page, entries
                    )
            if status == "imported":
                _publish(staging, path)
    return {
        "status": status,
        "index_path": str(path),
        "sha256": sha256,
        "section": SECTION,
        "first_pdf_page": first_page,
        "last_pdf_page": last_page,
        "entries": len(entries),
        "passages": sum(len(entry.passages) for entry in entries),
    }


def _store_edition(
    db: sqlite3.Connection,
    edition: ManualEdition,
    sha256: str,
    first_page: int,
    last_page: int,
    entries: list[Entry],
) -> str:
    manifest = edition.model_dump_json()
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("BEGIN IMMEDIATE")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if (
        version == 0
        and not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
        ).fetchone()
    ):
        for statement in _SCHEMA:
            db.execute(statement)
        db.execute(f"PRAGMA user_version = {_INDEX_VERSION}")
    elif version != _INDEX_VERSION:
        raise ManualImportError(
            "Unsupported index format; rebuild in a new index directory"
        )

    key = _key(edition)
    existing = db.execute(
        "SELECT manifest, sha256, first_page, last_page FROM manuals WHERE document_key = ?",
        (key,),
    ).fetchone()
    if existing:
        if existing != (manifest, sha256, first_page, last_page):
            raise ManualImportError(
                "This document and edition already have a different PDF, page range, or manifest"
            )
        return "unchanged"
    manual_id = db.execute(
        "INSERT INTO manuals (manifest,sha256,first_page,last_page,document_key) VALUES (?,?,?,?,?)",
        (manifest, sha256, first_page, last_page, key),
    ).lastrowid
    _write_entries(db, manual_id, entries)
    return "imported"


def _write_entries(
    db: sqlite3.Connection, manual_id: int, entries: Iterable[Entry]
) -> None:
    for entry in entries:
        db.execute(
            "INSERT INTO entries VALUES (?,?,?,?,?)",
            (manual_id, entry.code, entry.title, entry.page, json.dumps(entry.flags)),
        )
        for passage in entry.passages:
            passage_id = db.execute(
                "INSERT INTO passages (manual_id,code,page,kind,text) VALUES (?,?,?,?,?)",
                (manual_id, entry.code, passage.page, passage.kind, passage.text),
            ).lastrowid
            db.executemany(
                "INSERT INTO postings VALUES (?,?)",
                (
                    (term, passage_id)
                    for term in index_terms(passage.text, entry.title, entry.code)
                ),
            )


@contextmanager
def read_index(index_dir: Path) -> Iterator[sqlite3.Connection | None]:
    path = index_dir / _INDEX_NAME
    if not path.is_file():
        yield None
        return
    try:
        with closing(_connect_readonly(path)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if (
                version == 0
                and not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
                ).fetchone()
            ):
                yield None
                return
            if version != _INDEX_VERSION:
                raise ManualIndexError(
                    "Unsupported manual index format; rebuild the local index"
                )
            yield db
    except (sqlite3.DatabaseError, ValueError) as exc:
        raise ManualIndexError("Cannot read the local manual index") from exc


def read_manuals(db: sqlite3.Connection) -> list[IndexedManual]:
    return [
        IndexedManual(
            row["id"],
            ManualEdition.model_validate_json(row["manifest"]),
            row["sha256"],
            row["first_page"],
            row["last_page"],
        )
        for row in db.execute(
            "SELECT id, manifest, sha256, first_page, last_page FROM manuals ORDER BY document_key"
        )
    ]


def read_entry(db: sqlite3.Connection, manual_id: int, code: str) -> sqlite3.Row | None:
    return db.execute(
        "SELECT * FROM entries WHERE manual_id = ? AND code = ?", (manual_id, code)
    ).fetchone()


def entry_locations(
    db: sqlite3.Connection, manual_id: int, code: str
) -> Iterator[sqlite3.Row]:
    return db.execute(
        "SELECT page, kind FROM passages WHERE manual_id = ? AND code = ? "
        "GROUP BY page, kind ORDER BY MIN(id)",
        (manual_id, code),
    )


def candidates(
    db: sqlite3.Connection,
    manual_id: int,
    code: str | None,
    plan: SearchPlan,
) -> Iterator[sqlite3.Row]:
    using_terms = ""
    parameters: list = [manual_id]
    code_filter = ""
    if code:
        code_filter = "AND p.code = ?"
        parameters.append(code)
    if plan.terms:
        placeholders = ", ".join("?" for _ in plan.terms)
        using_terms = (
            f"p.id IN (SELECT passage_id FROM postings WHERE term IN ({placeholders}))"
        )
        parameters.extend(plan.terms)
    selection = [using_terms] if using_terms else []
    if code and plan.intent_only:
        selection.append("p.kind = ?")
        parameters.append(plan.intent)
    if not selection:
        return iter(())
    return db.execute(
        f"SELECT p.*, e.title FROM passages p JOIN entries e USING (manual_id, code) "
        f"WHERE p.manual_id = ? {code_filter} AND ({' OR '.join(selection)}) "
        "AND p.kind IN ('title', 'cause', 'remedy', 'diagnostic_value', 'cross_reference')",
        parameters,
    )
