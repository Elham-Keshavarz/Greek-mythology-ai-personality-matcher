"""LLM-enabled RAG offline generation contract tests: no external models or keys needed."""
from __future__ import annotations

from src.rag.answer_service import ABSTAIN, REJECTED, answer_question, build_prompt
from src.rag.grounding import UNCITED_ABSTENTION, verify_citations
from src.rag.knowledge_base import Document
from src.rag.text_chunker import chunk_documents
from src.rag.vector_store import Hit
from src.rag.evaluate_generation import audit


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, question, top_k=20, min_score=-1.0):
        self.queries.append(question)
        return self.hits


def make_hit(text="Athena is associated with wisdom and strategic warfare.", docid="athena"):
    doc = Document(
        id=docid, title="Athena: wisdom", text=text,
        source_url="https://www.theoi.com/Olympios/Athena.html",
        source_type="curated_summary", source_note="Editorial summary",
    )
    return Hit(chunk_documents([doc])[0], 0.88)


def test_valid_source_citations():
    result = verify_citations("Athena is linked to strategy. [S1]", {"S1"})
    assert result.valid
    assert result.citations == ("S1",)


def test_multiple_valid_citations_are_unique():
    result = verify_citations("A [S1] and B [S2]. [S1]", {"S1", "S2"})
    assert result.valid
    assert result.citations == ("S1", "S2")


def test_reject_unknown_source_citation():
    result = verify_citations("Something [S1] and [S9].", {"S1"})
    assert not result.valid
    assert "S9" in result.reason


def test_reject_missing_source_citation():
    assert not verify_citations("Athena is wise.", {"S1"}).valid


def test_reject_malformed_citation():
    assert not verify_citations("Athena is wise [s1] and [S1]", {"S1"}).valid


def test_empty_or_zero_source_set_is_rejected():
    assert not verify_citations("  ", {"S1"}).valid
    assert not verify_citations("A [S1]", set()).valid


def test_prompt_limited_to_approved_passages_and_no_quiz_data():
    hit = make_hit("Athena is associated with wisdom.")
    messages = build_prompt("Who is Athena?", [hit])
    assert "[S1]" in messages[1]["content"]
    assert "Athena" in messages[1]["content"]
    assert "quiz answers" not in messages[1]["content"].lower()
    assert "untrusted" in messages[0]["content"]


def test_generation_grounded_with_known_citation():
    hits = [make_hit()]
    calls = []

    def model(messages):
        calls.append(messages)
        return "Athena is linked to wisdom and strategy. [S1]"

    result = answer_question("Who is Athena?", FakeStore(hits), generator=model)
    assert result.mode == "llm"
    assert result.citations == ("S1",)
    assert len(result.hits) == 1
    assert len(calls) == 1


def test_bad_source_id_fails_closed():
    result = answer_question("Who is Athena?", FakeStore([make_hit()]),
                             generator=lambda _: "Athena is wise. [S99]")
    assert result.mode == "generation_rejected"
    assert result.text == REJECTED
    assert "Unknown" in result.reason


def test_uncited_generated_answer_fails_closed():
    result = answer_question("Who is Athena?", FakeStore([make_hit()]),
                             generator=lambda _: "Athena is wise.")
    assert result.mode == "generation_rejected"


def test_provider_returns_evidence_abstention():
    result = answer_question("Who is Athena?", FakeStore([make_hit()]),
                             generator=lambda _: UNCITED_ABSTENTION)
    assert result.mode == "llm_abstained"
    assert result.text == ABSTAIN


def test_no_api_call_on_unsupported_question():
    calls = []
    result = answer_question("What smartphone did Athena use?", FakeStore([make_hit()]),
                             generator=lambda x: calls.append(x) or "made up answer [S1]")
    assert result.mode == "insufficient_evidence"
    assert result.text == ABSTAIN
    assert calls == []


def test_no_api_call_with_no_hits():
    calls = []
    result = answer_question("Who is Athena?", FakeStore([]),
                             generator=lambda x: calls.append(x) or "abc")
    assert result.mode == "no_evidence"
    assert calls == []


def test_only_one_supported_passage_sent_to_model():
    first = make_hit("Athena is associated with wisdom and strategic warfare.", "athena")
    irrelevant = make_hit("Zeus is king of the Olympian gods and associated with thunder.", "zeus")
    messages = []

    def model(prompt):
        messages.extend(prompt)
        return "Athena is associated with wisdom. [S1]"

    result = answer_question("Who is Athena?", FakeStore([first, irrelevant]), generator=model)
    assert result.mode == "llm"
    assert len(result.hits) == 1
    assert result.hits[0].chunk.document_id == "athena"
    assert "Zeus is king" not in messages[1]["content"]


def test_offline_audit_marks_review_required():
    result = audit(FakeStore([make_hit()]), [{
        "question": "Who is Athena?",
        "should_answer": True,
        "review_hint": "wisdom"
    }], lambda _: "Athena symbolizes wisdom. [S1]")
    assert result["total"] == 1
    assert result["expected_decision_match_count"] == 1
    assert result["results"][0]["human_faithfulness_review"] == "NOT_REVIEWED"


def test_eval_is_opt_in(monkeypatch, capsys):
    from src.rag import evaluate_generation
    monkeypatch.setattr(evaluate_generation, "build_pipeline", lambda: (_ for _ in ()).throw(AssertionError("network")))
    assert evaluate_generation.main([]) == 0
    assert "disabled by default" in capsys.readouterr().out


def test_cli_without_llm_does_not_require_key(monkeypatch, capsys):
    from src.rag import ask
    monkeypatch.setattr(ask, "build_pipeline", lambda: FakeStore([make_hit()]))
    assert ask.main(["Who", "is", "Athena?"]) == 0
    assert "retrieval_only" in capsys.readouterr().out


def test_cli_llm_requires_configured_key(monkeypatch, capsys):
    from src.rag import ask
    monkeypatch.setattr(ask, "is_configured", lambda: False)
    monkeypatch.setattr(ask, "build_pipeline", lambda: (_ for _ in ()).throw(AssertionError("network")))
    assert ask.main(["--llm", "Who", "is", "Athena?"]) == 2
    assert "needs OPENAI_API_KEY" in capsys.readouterr().out
