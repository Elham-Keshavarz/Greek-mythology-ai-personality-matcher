"""Explicit live OpenAI/RAG smoke test for manually triggered GitHub Actions.

This runs ONE hosted generation. The negative case tests the evidence gate without
making an additional provider request. It is not a full faithfulness evaluation.
"""
from __future__ import annotations

from src.rag.answer_service import answer_question
from src.rag.llm_client import LLMProviderError, generate_answer, is_configured
from src.rag.pipeline import build_pipeline


def _no_remote_call(_messages: list[dict[str, str]]) -> str:
    raise AssertionError("Unsupported question reached the OpenAI generator")


def main() -> int:
    if not is_configured():
        print("FAIL: Configure OPENAI_API_KEY as a GitHub Actions secret and OPENAI_MODEL.")
        return 2

    try:
        store = build_pipeline()
        positive = answer_question(
            "Who are Apollo and Artemis's parents?", store, generator=generate_answer
        )
        if positive.mode != "llm" or "S1" not in positive.citations:
            print(f"FAIL: Expected a source-cited LLM answer; got {positive.mode}.")
            return 1
        answer_text = positive.text.lower()
        if "zeus" not in answer_text or "leto" not in answer_text:
            print("FAIL: LLM answer did not contain the expected parents.")
            return 1
        print("PASS: OpenAI generated a cited answer with Zeus and Leto.")

        negative = answer_question(
            "What smartphone did Zeus use?", store, generator=_no_remote_call
        )
        if negative.mode != "insufficient_evidence":
            print(f"FAIL: Unsupported question was not refused: {negative.mode}.")
            return 1
        print("PASS: Unsupported question abstained before an OpenAI call.")
        return 0
    except LLMProviderError as exc:
        print(f"FAIL: OpenAI provider error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
