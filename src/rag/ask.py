"""CLI for local retrieval or explicit hosted LLM generation."""
from __future__ import annotations

import argparse

from .answer_service import answer_question
from .llm_client import LLMConfigurationError, LLMProviderError, generate_answer, is_configured
from .pipeline import build_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mythology RAG question answering")
    parser.add_argument("question", nargs="*", help="English mythology question")
    parser.add_argument("--llm", action="store_true", help="Explicitly enable a paid hosted LLM call")
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip() or "How are apolo and atemis related?"
    if args.llm and not is_configured():
        print("LLM mode needs OPENAI_API_KEY and OPENAI_MODEL in .env. Retrieval-only works without them.")
        return 2
    try:
        result = answer_question(
            question, build_pipeline(), generator=generate_answer if args.llm else None
        )
    except (LLMConfigurationError, LLMProviderError) as exc:
        print(f"LLM unavailable: {exc}")
        return 1
    print(f"Original question: {question}")
    print(f"Normalized query:  {result.normalized_question}")
    if result.corrections:
        print("Corrections: " + ", ".join(f"{c.original} -> {c.corrected}" for c in result.corrections))
    print(f"Mode: {result.mode} | evidence term coverage: {result.evidence_coverage:.2f}")
    print(f"Reason: {result.reason}")
    if result.supporting_document_id:
        print(f"Evidence support: {result.supporting_document_id} | matched: {', '.join(result.matched_concepts)}")
    print()
    print(result.text)
    for i, hit in enumerate(result.hits, 1):
        print(f"\n[S{i}] {hit.chunk.title} | cosine={hit.score:.3f} | {hit.chunk.source_url}")
    if result.mode == "generation_rejected":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
