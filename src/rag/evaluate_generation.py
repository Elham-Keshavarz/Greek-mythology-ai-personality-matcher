"""Optional *live* LLM audit; never calls a provider unless --live is given.

This is NOT a faithfulness scorer. It records citation compliance and expected
abstentions and provides model outputs for human factual review.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .answer_service import answer_question
from .llm_client import LLMProviderError, generate_answer, is_configured
from .pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[2]


def audit(store, questions: list[dict], generator) -> dict:
    results = []
    for item in questions:
        answer = answer_question(item["question"], store, generator=generator)
        expected = bool(item["should_answer"])
        accepted = answer.mode == "llm"
        results.append({
            "question": item["question"],
            "expected_supported": expected,
            "mode": answer.mode,
            "produced_answer": accepted,
            "outcome_matches_expected": accepted == expected,
            "citation_ids": list(answer.citations),
            "answer": answer.text,
            "supporting_document_id": answer.supporting_document_id,
            "supporting_passage": next((h.chunk.text for h in answer.hits
                                        if h.chunk.document_id == answer.supporting_document_id), None),
            "background_urls": [h.chunk.source_url for h in answer.hits],
            "review_hint": item.get("review_hint", ""),
            "human_faithfulness_review": "NOT_REVIEWED",
        })
    by_mode = dict(Counter(r["mode"] for r in results))
    return {
        "methodology": (
            "Small corpus-derived exploratory QA. Programmatic checks cover answer-vs-abstain "
            "and citation syntax, NOT claim-level factual entailment. Review outputs manually."
        ),
        "total": len(results),
        "expected_decision_match_count": sum(r["outcome_matches_expected"] for r in results),
        "mode_counts": by_mode,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit LLM answers with mandatory source checks")
    parser.add_argument("--live", action="store_true", help="Explicitly opt in to paid remote API requests")
    parser.add_argument("--limit", type=int, default=7, help="Maximum number of evaluation questions (1..7)")
    parser.add_argument("--output", help="Optional local JSON report path (avoid committing chat data)")
    args = parser.parse_args(argv)
    if not args.live:
        print("Live LLM audit is disabled by default. Add --live to opt in to provider calls.")
        print("Run .venv\\Scripts\\python -m pytest tests/test_llm.py -q for offline contract tests.")
        return 0
    if not is_configured():
        print("Missing OPENAI_API_KEY or OPENAI_MODEL. Add both to local .env before --live.")
        return 2
    if not 1 <= args.limit <= 7:
        parser.error("--limit must be between 1 and 7")
    questions = json.loads((ROOT / "data" / "evaluation" / "llm_qa.json").read_text(encoding="utf-8"))
    try:
        report = audit(build_pipeline(), questions[:args.limit], generate_answer)
    except LLMProviderError as exc:
        print(f"Provider error: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Audit saved to {path} (review the factual claims manually)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
