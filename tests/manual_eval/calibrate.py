"""Choose ranking parameters using only the frozen development split."""

import argparse
import asyncio
import hashlib
import itertools
import json
from pathlib import Path

from evaluate import DATA, FROZEN_DATA_SHA, run
from tuagent.manual.vectors import OllamaEmbedder


class CalibrationEmbeddings:
    """Memoize dev query vectors for the grid. Timings from this run are discarded."""

    def __init__(self, source):
        self.source = source
        self.model = source.model
        self.revision = source.digest()
        self.vectors = {}

    def digest(self):
        return self.revision

    def embed(self, texts):
        key = tuple(texts)
        if key not in self.vectors:
            self.vectors[key] = self.source.embed(texts)
        return self.vectors[key]


def objective(summary):
    return (
        summary["support_at_5"]
        - 2 * summary["negative_false_hits"]
        - 2 * summary["scope_failures"],
        summary["zh_support_at_5"],
        summary["support_at_1"],
        summary["hit_at_5"],
        summary["mrr_at_5"],
    )


async def calibrate(index_dir, dataset):
    source = OllamaEmbedder.from_env(timeout=60)
    embedder = CalibrationEmbeddings(source)
    thresholds = (0.45, 0.50, 0.55, 0.60, 0.625, 0.65, 0.70, 0.75)
    variants = {
        "vector": ({"vector_min_similarity": t} for t in thresholds),
        "rrf": (
            {"rrf_min_similarity": t, "depth": depth, "rrf_k": k, "vector_weight": w}
            for t, depth, k, w in itertools.product(
                thresholds, (12, 24), (10, 20), (1.0, 1.5, 2.0)
            )
        ),
    }
    fit = {}
    for mode, configurations in variants.items():
        scores = []
        for config in configurations:
            args = argparse.Namespace(
                index_dir=index_dir,
                mode=mode,
                case=None,
                split="dev",
                warmup=0,
                diagnostics=not scores,
                **config,
            )
            report = await run(args, dataset, embedder=embedder)
            summary = report["summary"]
            if summary["errors"] or summary["fallbacks"]:
                failures = [
                    row.get("error") for row in report["cases"] if row["is_error"]
                ]
                raise RuntimeError(
                    f"Calibration requires live vector results: {failures}"
                )
            if not scores and mode == "vector":
                raw = {
                    row["id"]: [
                        [
                            hit["code"],
                            hit["kind"],
                            hit["reference"]["pdf_page"],
                            hit["similarity"],
                        ]
                        for hit in (row.get("diagnostics") or {}).get(
                            "vector_candidates", []
                        )
                    ]
                    for row in report["cases"]
                }
            scores.append(
                {
                    "config": config,
                    "objective": objective(summary),
                    "summary": {
                        k: v
                        for k, v in summary.items()
                        if k not in ("latency_samples", "p95_ms")
                    },
                }
            )
        scores.sort(
            key=lambda item: (
                *item["objective"],
                item["config"].get(
                    "rrf_min_similarity", item["config"].get("vector_min_similarity")
                ),
                -item["config"].get("depth", 0),
                -abs(item["config"].get("rrf_k", 20) - 20),
                -abs(item["config"].get("vector_weight", 1) - 1),
            ),
            reverse=True,
        )
        fit[mode] = scores
    if source.digest() != embedder.revision:
        raise RuntimeError("Ollama model changed during development calibration")
    return {
        "dataset_sha256": FROZEN_DATA_SHA,
        "model": embedder.model,
        "model_digest": embedder.revision,
        "dev_cases": len(
            [case for case in dataset["questions"] if case["split"] == "dev"]
        ),
        "distinct_query_embeddings": len(embedder.vectors),
        "configurations": fit,
        "raw_vector_top_5": raw,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path(".manual/index"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = DATA.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != FROZEN_DATA_SHA:
        parser.error("Frozen evaluation set hash does not match")
    result = asyncio.run(calibrate(args.index_dir, json.loads(source)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                mode: configurations[0]
                for mode, configurations in result["configurations"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
