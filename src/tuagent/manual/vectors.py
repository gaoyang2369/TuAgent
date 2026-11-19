"""Build and query the local dense manual index using Ollama embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from zipfile import BadZipFile

from .index import (
    IndexedManual,
    ManualIndexError,
    _publish,
    passage_key,
    read_index,
    read_manuals,
    scoped_passages,
)

if TYPE_CHECKING:
    from numpy import ndarray


VECTOR_FILENAME = "manual-vectors.npz"
REBUILD = "tuagent-build-manual-vectors --index-dir <manual-index-dir>"
DOCUMENT_TEMPLATE = "Code: {code}\nTitle: {title}\nSection: {kind}\n{text}"
QUERY_TEMPLATE = (
    "Instruct: Given a question about a SINAMICS drive, retrieve the English manual "
    "passages that directly answer it with the relevant fault or alarm condition, cause, "
    "diagnostic value, or remedy.\nQuery: {query}"
)


class VectorError(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class Embedder(Protocol):
    model: str

    def digest(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 10,
        *,
        threads: int | None = None,
    ):
        parts = urlsplit(base_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) service URL")
        if not model.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Supply a model tag and a positive finite timeout")
        if threads is not None and not 1 <= threads <= 256:
            raise ValueError("OLLAMA_EMBED_THREADS must be between 1 and 256")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout
        self.threads = threads
        self._opener = build_opener(ProxyHandler({}))

    @classmethod
    def from_env(cls, timeout: float = 10) -> OllamaEmbedder:
        base_url = os.environ.get("OLLAMA_BASE_URL", "")
        if not base_url:
            raise ValueError("Set OLLAMA_BASE_URL to the Ollama server")
        configured_threads = os.environ.get("OLLAMA_EMBED_THREADS", "")
        try:
            threads = int(configured_threads) if configured_threads else None
        except ValueError:
            raise ValueError("OLLAMA_EMBED_THREADS must be an integer") from None
        return cls(
            base_url,
            os.environ.get("EMBEDDING_MODEL", "qwen3-embedding:8b"),
            timeout,
            threads=threads,
        )

    def _request(self, endpoint: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(f"{self.base_url}/api/{endpoint}", data=data)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._opener.open(request, timeout=self.timeout) as reply:
                return json.load(reply)
        except HTTPError as exc:
            exc.close()
            raise VectorError(
                "embedding_http_error", f"Ollama returned HTTP {exc.code}."
            ) from None
        except (URLError, OSError, HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise VectorError(
                    "embedding_timeout", "Ollama request timed out."
                ) from None
            raise VectorError("embedding_unavailable", "Cannot reach Ollama.") from None
        except (ValueError, UnicodeError):
            raise VectorError(
                "invalid_embedding_response", "Ollama returned invalid JSON."
            ) from None

    def digest(self) -> str:
        payload = self._request("tags")
        if isinstance(payload, dict) and isinstance(payload.get("models"), list):
            for candidate in payload["models"]:
                if not isinstance(candidate, dict):
                    continue
                if self.model in (candidate.get("name"), candidate.get("model")):
                    digest = candidate.get("digest")
                    if isinstance(digest, str) and re.fullmatch(
                        r"(?:sha256:)?[a-fA-F0-9]{64}", digest
                    ):
                        return digest.removeprefix("sha256:").lower()
        raise VectorError(
            "embedding_model_unavailable", "Model tag or digest is absent from Ollama."
        )

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        request = {"model": self.model, "input": list(texts), "truncate": False}
        if self.threads is not None:
            request["options"] = {"num_thread": self.threads}
        payload = self._request("embed", request)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("embeddings"), list
        ):
            raise VectorError(
                "invalid_embedding_response", "Ollama returned no embedding array."
            )
        if payload.get("model") != self.model:
            raise VectorError(
                "embedding_model_mismatch", "Ollama used a different model tag."
            )
        return payload["embeddings"]


def _numpy():
    try:
        import numpy
    except ImportError:
        raise VectorError(
            "vector_dependency_missing",
            'Install tuagent with the "manual-vector" extra.',
        ) from None
    return numpy


def _normalized(values, count: int, dimension: int | None = None) -> ndarray:
    np = _numpy()
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Expected numeric JSON values")
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = raw.astype(np.float32)
    except (TypeError, ValueError, OverflowError):
        raise VectorError(
            "invalid_embeddings", "Embeddings must be a rectangular numeric matrix."
        ) from None
    if matrix.ndim != 2 or matrix.shape[0] != count or not matrix.shape[1]:
        raise VectorError("invalid_embeddings", "Incorrect embedding count or shape.")
    if dimension is not None and matrix.shape[1] != dimension:
        raise VectorError(
            "embedding_dimension_mismatch",
            f"Embedding dimension changed. Rebuild with {REBUILD}.",
        )
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise VectorError(
            "invalid_embeddings", "Embeddings must be finite and have nonzero norm."
        )
    return (matrix / norms[:, None]).astype("<f4")


def _json(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


@dataclass(frozen=True)
class Corpus:
    passages: list[tuple[IndexedManual, sqlite3.Row]]
    ids: list[str]
    inputs: list[str]
    sha256: str


def corpus(db: sqlite3.Connection, manuals: list[IndexedManual]) -> Corpus:
    passages = []
    ids = []
    inputs = []
    fingerprint = hashlib.sha256()
    for manual in manuals:
        edition = manual.edition.model_dump(mode="json")
        fingerprint.update(
            _json([edition, manual.sha256, manual.first_page, manual.last_page])
        )
        for row in scoped_passages(db, manual.id):
            passage_id = passage_key(manual, row)
            document = DOCUMENT_TEMPLATE.format_map(dict(row))
            fingerprint.update(_json([passage_id, document]))
            passages.append((manual, row))
            ids.append(passage_id)
            inputs.append(document)
    return Corpus(passages, ids, inputs, fingerprint.hexdigest())


@dataclass(frozen=True)
class VectorCache:
    manifest: dict
    ids: list[str]
    matrix: ndarray


def _load(path: Path) -> VectorCache:
    np = _numpy()
    try:
        with np.load(path, allow_pickle=False) as saved:
            manifest = json.loads(str(saved["manifest"].item()))
            ids = saved["ids"].tolist()
            matrix = saved["matrix"]
        if (
            manifest["format"] != 1
            or matrix.dtype != np.dtype("<f4")
            or matrix.shape != (len(ids), manifest["dimension"])
            or len(ids) != manifest["passages"]
            or matrix.shape[1] < 1
            or hashlib.sha256(matrix).hexdigest() != manifest["matrix_sha256"]
        ):
            raise ValueError("Unexpected matrix format")
        matrix.setflags(write=False)
        return VectorCache(manifest, ids, matrix)
    except FileNotFoundError:
        raise VectorError(
            "vector_cache_missing", f"Build the vector cache with {REBUILD}."
        ) from None
    except (OSError, EOFError, BadZipFile, TypeError, KeyError, IndexError, ValueError):
        raise VectorError(
            "vector_cache_invalid",
            f"Rebuild the unreadable vector cache with {REBUILD}.",
        ) from None


def _corpus_at(index_dir: Path) -> Corpus:
    with read_index(index_dir) as db:
        if db is not None:
            return corpus(db, read_manuals(db))
    raise VectorError("empty_index", "Import a local manual edition first.")


def _resume_batches(db: sqlite3.Connection, identity: dict) -> list[ndarray]:
    np = _numpy()
    signature = _json(identity).decode()
    with db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS build (id INTEGER PRIMARY KEY, signature TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS batches (offset INTEGER PRIMARY KEY, count INTEGER NOT NULL, "
            "dimension INTEGER NOT NULL, sha256 TEXT NOT NULL, data BLOB NOT NULL)"
        )
        if db.execute("SELECT signature FROM build WHERE id=1").fetchone() != (
            signature,
        ):
            db.execute("DELETE FROM batches")
            db.execute("INSERT OR REPLACE INTO build VALUES (1, ?)", (signature,))
    batches = []
    offset = 0
    dimension = None
    for start, count, width, sha256, data in db.execute(
        "SELECT * FROM batches ORDER BY offset"
    ):
        if (
            start != offset
            or not 0 < count <= identity["passages"] - offset
            or not 0 < width
            or (dimension is not None and dimension != width)
            or len(data) != count * width * 4
            or hashlib.sha256(data).hexdigest() != sha256
        ):
            with db:
                db.execute("DELETE FROM batches WHERE offset >= ?", (start,))
            break
        batches.append(np.frombuffer(data, dtype="<f4").reshape(count, width))
        dimension = width
        offset += count
    return batches


def build_vectors(
    index_dir: Path,
    embedder: Embedder,
    *,
    batch_size: int = 16,
    rebuild: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be between 1 and 128")
    np = _numpy()
    start = time.perf_counter()
    documents = _corpus_at(index_dir)
    if not documents.ids:
        raise VectorError("empty_index", "The manual index has no searchable passages.")
    digest = embedder.digest()
    identity = {
        "format": 1,
        "corpus_sha256": documents.sha256,
        "passages": len(documents.ids),
        "model": embedder.model,
        "model_digest": digest,
        "document_template_version": 1,
        "document_template": DOCUMENT_TEMPLATE,
        "query_template": QUERY_TEMPLATE,
    }
    path = index_dir / VECTOR_FILENAME
    with closing(sqlite3.connect(index_dir / ".vectors.lock", timeout=5)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        if not rebuild and path.is_file():
            try:
                previous = _load(path)
            except VectorError:
                previous = None
            if (
                previous
                and previous.ids == documents.ids
                and all(
                    previous.manifest.get(key) == value
                    for key, value in identity.items()
                )
            ):
                return _build_result("unchanged", path, previous.manifest, start)
        checkpoint_path = index_dir / ".vector-build.sqlite3"
        with closing(sqlite3.connect(checkpoint_path)) as checkpoint:
            batches = _resume_batches(checkpoint, identity)
            resumed = sum(len(batch) for batch in batches)
            dimension = batches[0].shape[1] if batches else None
            if resumed and progress is not None:
                progress(resumed, len(documents.inputs))
            for offset in range(resumed, len(documents.inputs), batch_size):
                texts = documents.inputs[offset : offset + batch_size]
                batch = _normalized(embedder.embed(texts), len(texts), dimension)
                if embedder.digest() != digest:
                    raise VectorError(
                        "embedding_model_mismatch",
                        "Model digest changed during the build. Retry.",
                    )
                dimension = batch.shape[1]
                data = batch.tobytes()
                with checkpoint:
                    checkpoint.execute(
                        "INSERT INTO batches VALUES (?, ?, ?, ?, ?)",
                        (
                            offset,
                            len(batch),
                            dimension,
                            hashlib.sha256(data).hexdigest(),
                            data,
                        ),
                    )
                batches.append(batch)
                if progress is not None:
                    progress(offset + len(texts), len(documents.inputs))
            if _corpus_at(index_dir).sha256 != documents.sha256:
                raise VectorError(
                    "vector_corpus_mismatch",
                    "The manual index changed during the build. Retry.",
                )
            if embedder.digest() != digest:
                raise VectorError(
                    "embedding_model_mismatch",
                    "Model digest changed before publication. Retry.",
                )
            matrix = np.concatenate(batches)
            manifest = {
                **identity,
                "dimension": dimension,
                "matrix_bytes": matrix.nbytes,
                "matrix_sha256": hashlib.sha256(matrix).hexdigest(),
            }
            with tempfile.TemporaryDirectory(
                prefix=".vectors-", dir=index_dir
            ) as staging:
                staged = Path(staging) / VECTOR_FILENAME
                np.savez(
                    staged,
                    manifest=json.dumps(manifest),
                    ids=documents.ids,
                    matrix=matrix,
                )
                _publish(staged, path)
        checkpoint_path.unlink(missing_ok=True)
    return {
        **_build_result("built", path, manifest, start),
        "resumed_passages": resumed,
    }


def _build_result(status: str, path: Path, manifest: dict, start: float) -> dict:
    return {
        "status": status,
        "path": str(path),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "file_bytes": path.stat().st_size,
        **manifest,
    }


class VectorSearch:
    def __init__(self, index_dir: Path, embedder: Embedder | None):
        self.path = index_dir / VECTOR_FILENAME
        self.embedder = embedder
        self._lock = Lock()
        self._cached: tuple[tuple[int, ...], VectorCache] | None = None

    def _cache(self) -> VectorCache:
        with self._lock:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                raise VectorError(
                    "vector_cache_missing", f"Build the vector cache with {REBUILD}."
                ) from None
            except OSError:
                raise VectorError(
                    "vector_cache_invalid",
                    f"Cannot read the vector cache. Rebuild with {REBUILD}.",
                ) from None
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
            if self._cached is None or self._cached[0] != stamp:
                self._cached = stamp, _load(self.path)
            return self._cached[1]

    def rank(
        self,
        documents: Corpus,
        manual_ids: set[int],
        codes: tuple[str, ...],
        query: str,
    ) -> list[tuple[float, IndexedManual, sqlite3.Row]]:
        selection = [
            i
            for i, (manual, row) in enumerate(documents.passages)
            if manual.id in manual_ids and (not codes or row["code"] in codes)
        ]
        if not selection:
            return []
        saved = self._cache()
        manifest = saved.manifest
        if (
            manifest.get("corpus_sha256") != documents.sha256
            or saved.ids != documents.ids
            or manifest.get("document_template") != DOCUMENT_TEMPLATE
            or manifest.get("query_template") != QUERY_TEMPLATE
        ):
            raise VectorError(
                "vector_corpus_mismatch",
                f"Cache does not match the current index or template. Rebuild with {REBUILD}.",
            )
        embedder = self.embedder
        if embedder is None:
            try:
                embedder = OllamaEmbedder.from_env()
            except ValueError as exc:
                raise VectorError("embedding_not_configured", str(exc)) from None
        if embedder.model != manifest.get("model") or embedder.digest() != manifest.get(
            "model_digest"
        ):
            raise VectorError(
                "embedding_model_mismatch",
                f"Cache model or digest changed. Rebuild with {REBUILD}.",
            )
        vector = _normalized(
            embedder.embed([QUERY_TEMPLATE.format(query=query)]),
            1,
            manifest["dimension"],
        )[0]
        if embedder.digest() != manifest["model_digest"]:
            raise VectorError(
                "embedding_model_mismatch",
                f"Model digest changed during search. Rebuild with {REBUILD}.",
            )
        np = _numpy()
        matrix = (
            saved.matrix
            if len(selection) == len(documents.ids)
            else saved.matrix[selection]
        )
        scores = np.einsum("ij,j->i", matrix, vector)
        order = np.argsort(-scores, kind="stable")
        return [(float(scores[i]), *documents.passages[selection[i]]) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--timeout", default=90, type=float)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    def report(done: int, total: int) -> None:
        if done == total or (done // args.batch_size) % 4 == 0:
            print(f"Embedded {done}/{total} passages", file=sys.stderr, flush=True)

    try:
        embedder = OllamaEmbedder.from_env(args.timeout)
        result = build_vectors(
            args.index_dir,
            embedder,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            progress=report,
        )
    except (VectorError, ManualIndexError, ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"Vector build failed: {exc}") from None
    print(json.dumps({**result, "ollama_num_thread": embedder.threads}, indent=2))


if __name__ == "__main__":
    main()
