"""HTTP adapters for optional rewrite and passage reranking services."""

import json
import math
from dataclasses import dataclass
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .evidence import EvidenceServiceError


def _post(base_url: str, path: str, data: dict, timeout: float, stage: str):
    parts = urlsplit(base_url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Use a service HTTP(S) URL without embedded credentials")
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        error.close()
        raise EvidenceServiceError(f"{stage}_http_{error.code}") from None
    except (URLError, OSError, HTTPException) as error:
        reason = getattr(error, "reason", error)
        raise EvidenceServiceError(
            f"{stage}_timeout"
            if isinstance(reason, TimeoutError)
            else f"{stage}_unavailable"
        ) from None
    except ValueError:
        raise EvidenceServiceError(f"{stage}_invalid_response") from None


@dataclass(frozen=True)
class OllamaRewriter:
    base_url: str
    model: str
    timeout: float = 15

    def rewrite(self, question: str, terms: dict[str, str]) -> str:
        if self.model.endswith(":cloud"):
            raise EvidenceServiceError("rewrite_requires_local_model")
        payload = _post(
            self.base_url,
            "/api/generate",
            {
                "model": self.model,
                "system": (
                    "Translate the question into exactly one English search question. "
                    "Use every applicable provided term. Keep negation, comparisons, numbers, units and IDs unchanged. "
                    "Never add a fault code, cause, symptom or suggested action. Output the question alone."
                ),
                "prompt": json.dumps(
                    {"question": question, "glossary": terms}, ensure_ascii=False
                ),
                "stream": False,
                "think": False,
                "keep_alive": "15s",
                "options": {"temperature": 0, "num_predict": 96, "num_ctx": 2048},
            },
            self.timeout,
            "rewrite",
        )
        if (
            not isinstance(payload, dict)
            or payload.get("model") != self.model
            or not isinstance(payload.get("response"), str)
        ):
            raise EvidenceServiceError("rewrite_invalid_response")
        return payload["response"].strip()


@dataclass(frozen=True)
class HttpReranker:
    base_url: str
    model: str
    timeout: float = 10

    def score(self, question: str, documents: list[str]) -> list[float]:
        payload = _post(
            self.base_url,
            "/rerank",
            {"model": self.model, "query": question, "documents": documents},
            self.timeout,
            "rerank",
        )
        rows = payload.get("results") if isinstance(payload, dict) else None
        scores: dict[int, float] = {}
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    break
                index, value = row.get("index"), row.get("relevance_score")
                if (
                    type(index) is not int
                    or index not in range(len(documents))
                    or index in scores
                    or type(value) not in (int, float)
                    or not math.isfinite(value)
                ):
                    break
                scores[index] = value
            else:
                if len(scores) == len(documents):
                    return [scores[index] for index in range(len(documents))]
        raise EvidenceServiceError("rerank_invalid_response")
