"""Regression tests that do not need downloads, FAISS, or an API key."""
from __future__ import annotations

from src.rag.answer_service import ABSTAIN, answer_question
from src.rag.knowledge_base import load_documents
from src.rag.query_normalizer import normalize_query
from src.rag.retriever import RobustRetriever
from src.rag.text_chunker import chunk_documents
from src.rag.vector_store import Hit


class DeterministicStore:
    """Tests the service/reranker/gate without pretending to test embeddings."""
    def __init__(self):
        self.chunks = chunk_documents(load_documents())
        self.queries = []

    def search(self, question, top_k=20, min_score=-1.0):
        self.queries.append(question)
        # Explicitly mix high-semantic-score but irrelevant results with
        # the lower-score relationship evidence to test hybrid reranking.
        scores = {
            "aphrodite-love": 0.51,
            "hades-underworld": 0.48,
            "apollo-artemis": 0.47,
            "profile-apollo": 0.46,
            "profile-artemis": 0.46,
            "zeus-sky": 0.68,
            "profile-zeus": 0.66,
            "hermes-messenger": 0.72,
        }
        selected = []
        for chunk in self.chunks:
            if chunk.document_id in scores:
                selected.append(Hit(chunk, scores[chunk.document_id]))
        selected.sort(key=lambda h: -h.score)
        return [h for h in selected if h.score >= min_score][:top_k]


def test_typos_corrected_without_changing_other_words():
    text, changes = normalize_query("how are apolo and atemis related?")
    assert text == "how are Apollo and Artemis related?"
    assert [(c.original, c.corrected) for c in changes] == [("apolo", "Apollo"), ("atemis", "Artemis")]


def test_common_words_and_exact_names_are_preserved():
    text = "Why are there more gods here like Hera, Apollo and Zeus?"
    assert normalize_query(text) == (text, [])


def test_hybrid_ranking_promotes_a_relationship_fact():
    store = DeterministicStore()
    result = RobustRetriever(store).retrieve("How are apolo and atemis related?")
    assert store.queries[-1] == "How are Apollo and Artemis related?"
    assert result.hits[0].chunk.document_id == "apollo-artemis"
    assert result.status == "supported_candidates"


def test_smartphone_question_abstains_even_with_high_zeus_match():
    store = DeterministicStore()
    calls = []
    result = answer_question("What smartphone did Zeus use?", store, generator=lambda messages: calls.append(messages) or "Fake answer")
    assert result.mode == "insufficient_evidence"
    assert result.text == ABSTAIN
    assert result.hits  # FAISS still found Zeus, but the gate rejects answer generation
    assert calls == []
    assert "subject" in result.reason


def test_supported_topic_without_explicit_deity_can_retrieve_hermes():
    store = DeterministicStore()
    result = answer_question("Which Greek deity represents clever communication and traveling?", store)
    assert result.hits[0].chunk.document_id == "hermes-messenger"
    assert result.mode == "retrieval_only"


def test_unsupported_relation_abstains_without_shared_relationship_note():
    store = DeterministicStore()
    result = answer_question("How are Hera and Atlas related?", store)
    assert result.mode == "insufficient_evidence"
    assert result.text == ABSTAIN


def test_corrected_question_used_in_llm_prompt():
    store = DeterministicStore()
    messages = []
    def generate(prompt):
        messages.extend(prompt)
        return "Apollo and Artemis are twins. [S1]"
    result = answer_question("How are apolo and atemis related?", store, generator=generate)
    assert result.mode == "llm"
    assert "Apollo and Artemis" in messages[1]["content"]
    assert "apolo" not in messages[1]["content"].lower()
    assert len(result.corrections) == 2


def test_general_out_of_scope_questions_are_rejected():
    result = answer_question("Which smartphone did Zeus use?", DeterministicStore())
    assert result.evidence_coverage == 0
