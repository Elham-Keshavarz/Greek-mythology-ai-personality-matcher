"""Export an opt-in, local manual fact review worksheet from an LLM audit.

This module never makes LLM or HTTP requests. It does not assign automatic
faithfulness scores; reviewers decide whether each claim is source-supported.
Review exports can include user questions and generated text: do not commit.

Usage:
 python -m src.rag.faithfulness_review --input reports/local_llm_audit.json \\
     --csv reports/local_faithfulness_review.csv
 python -m src.rag.faithfulness_review --summary reports/local_faithfulness_review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REVIEW_LABELS = {"not_reviewed", "supported", "unsupported", "uncertain", "not_applicable"}
COLUMNS = (
    "question", "mode", "expected_supported", "answer", "approved_passage",
    "source_urls", "citation_ids", "review_hint", "review_label", "review_notes",
)


def make_review_rows(report: dict) -> list[dict]:
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError("Expected audit JSON with a 'results' array")
    rows: list[dict] = []
    for item in report["results"]:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError("Invalid audit result")
        answered = item.get("mode") == "llm"
        rows.append({
            "question": item["question"],
            "mode": str(item.get("mode", "")),
            "expected_supported": str(bool(item.get("expected_supported", False))).lower(),
            "answer": str(item.get("answer", "")),
            "approved_passage": str(item.get("supporting_passage") or ""),
            "source_urls": " | ".join(item.get("background_urls", [])),
            "citation_ids": " | ".join(item.get("citation_ids", [])),
            "review_hint": str(item.get("review_hint", "")),
            "review_label": "not_reviewed" if answered else "not_applicable",
            "review_notes": "",
        })
    return rows


def export_review_csv(report: dict, path: Path) -> int:
    rows = make_review_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_review_csv(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    counts = Counter()
    for row in rows:
        label = row.get("review_label", "").strip().lower()
        if label not in REVIEW_LABELS:
            raise ValueError(f"Invalid review_label {label!r}; use one of {sorted(REVIEW_LABELS)}")
        if row.get("mode") == "llm":
            counts[label] += 1
    reviewed = counts["supported"] + counts["unsupported"]
    return {
        "generated_answer_count": sum(counts.values()),
        "manually_reviewed_binary_count": reviewed,
        "supported_count": counts["supported"],
        "unsupported_count": counts["unsupported"],
        "uncertain_count": counts["uncertain"],
        "not_reviewed_count": counts["not_reviewed"],
        "faithfulness_rate_on_binary_reviewed_answers": (
            counts["supported"] / reviewed if reviewed else None
        ),
        "note": "Manual assessment on reviewed answers only; abstentions excluded. No model-based entailment verification.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="JSON created by evaluate_generation --live")
    group.add_argument("--summary", type=Path, help="Reviewed CSV to summarize offline")
    parser.add_argument("--csv", type=Path, help="Local output CSV path for --input")
    args = parser.parse_args(argv)
    if args.input:
        if args.csv is None:
            parser.error("--csv is required with --input")
        report = json.loads(args.input.read_text(encoding="utf-8"))
        print(f"Exported {export_review_csv(report, args.csv)} audit rows to {args.csv}")
        print("Fill review_label with supported, unsupported, or uncertain for each generated answer.")
        print("Keep this file local; it may contain private prompts and generated answers.")
    else:
        print(json.dumps(summarize_review_csv(args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
