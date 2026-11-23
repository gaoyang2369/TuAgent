"""Recheck a saved F30002 model report against the visible samples and original PDF."""

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


def verify(record, pdf):
    assert record["status"] == "completed"
    receipts = record["receipts"]
    scope = record["window"]

    def results(name):
        return [
            r["result"] for r in receipts if r["tool"] == name and not r["is_error"]
        ]

    events = [
        event
        for r in receipts
        if r["tool"] == "find_code_events" and not r["is_error"]
        if all(
            json.loads(r["arguments"]).get(k) == scope.get(k)
            for k in ("table_name", "start_ms", "end_ms", "last_seconds")
        )
        if json.loads(r["arguments"]).get("code_field", "fault_code") == "fault_code"
        if not r["result"].get("content_truncated")
        and not r["result"].get("events_truncated")
        for event in r["result"]["events"]
        if event["trigger"]
    ]
    latest = max(events, key=lambda e: e["trigger"]["timestamp_ms"])
    assert latest["code"] == "F30002"
    t = latest["trigger"]["timestamp_ms"]
    points = {}
    for page in results("query_telemetry"):
        if page["source"]["table_name"] != scope["table_name"]:
            continue
        for point in page.get("data", []):
            if abs(point["timestamp_ms"] - t) <= 30000 and point["valid"].get(
                "dc_voltage"
            ):
                points[(point["timestamp_ms"], point["id"])] = point
    assert {ms for ms, _ in points} == set(range(t - 30000, t + 30001, 3000))
    peak = max(
        (points[key] for key in sorted(points)), key=lambda p: p["values"]["dc_voltage"]
    )
    report = record["report"]
    assert all(
        str(value) in report
        for value in (
            peak["values"]["dc_voltage"],
            peak["id"],
            peak["time_beijing"][11:19],
        )
    )
    assert any(
        p["timestamp_ms"] < t
        and p["valid"].get("actual_power")
        and p["values"]["actual_power"] < 0
        for page in results("query_telemetry")
        if page["source"]["table_name"] == scope["table_name"]
        for p in page.get("data", [])
        if p["timestamp_ms"] >= t - 30000
    )
    heading = next(
        h for h in results("lookup_drive_code") if h.get("code") == latest["code"]
    )
    cause = "motor regenerates too much energy"
    hit = next(
        h
        for page in results("search_manual")
        for h in page.get("hits", [])
        if h["code"] == latest["code"]
        and h["kind"] == "cause"
        and cause in h["excerpt"]
    )
    assert (
        heading["reference"]["pdf_page"] == 904 and hit["reference"]["pdf_page"] == 905
    )
    assert hit["applicability"]["status"] == "candidate_only"
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == hit["reference"]["sha256"]
    pages = PdfReader(pdf).pages
    assert latest["code"] in pages[903].extract_text()
    assert cause in " ".join(pages[904].extract_text().split())
    assert cause in report and "905" in report and "candidate_only" in report
    return {
        "event": latest["ref"],
        "valid_points": len(points),
        "peak_V": peak["values"]["dc_voltage"],
        "first_peak": peak["ref"],
        "beijing": peak["time_beijing"],
        "pdf_page": 905,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record", type=Path, default=Path("docs/diagnosis/F30002.record.json")
    )
    parser.add_argument(
        "--pdf", type=Path, default=Path(".manual/g120-cu240be-2-0223-en.pdf")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            verify(json.loads(args.record.read_text(encoding="utf8")), args.pdf),
            indent=2,
        )
    )
