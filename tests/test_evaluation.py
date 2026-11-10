"""Evaluation math and dataset shape, without embedding downloads."""
import pytest

from src.rag.evaluate import evaluate_robust_retrieval, load_suite
from src.rag.knowledge_base import load_documents
from src.rag.text_chunker import chunk_documents
from src.rag.vector_store import Hit


class SingleHitStore:
    def __init__(self):
        chunks = chunk_documents(load_documents())
        self.hit = next(Hit(c, 0.88) for c in chunks if c.document_id == "hypnos-sleep")

    def search(self, question, top_k=4, min_score=-1):
        return [self.hit]


def test_both_benchmark_sets_have_distinct_questions_and_valid_document_ids():
    ids = {d.id for d in load_documents()}
    for name in ("regression", "challenge"):
        examples = load_suite(name)
        positive_q = [q["question"] for q in examples["positive"]]
        negative_q = [q["question"] for q in examples["negative"]]
        assert len(positive_q) == len(set(positive_q))
        assert len(negative_q) == len(set(negative_q))
        assert not set(positive_q) & set(negative_q)
        assert all(item["expected_document_id"] in ids for item in examples["positive"])


def test_benchmark_metrics_separate_ranking_and_gate_decisions():
    metrics = evaluate_robust_retrieval(
        SingleHitStore(),
        [{"question": "What does Hypnos personify?", "expected_document_id": "hypnos-sleep"}],
        [{"question": "What smartphone did Hypnos use?"}],
    )
    assert metrics["hit@5"] == 1.0
    assert metrics["mrr@5"] == 1.0
    assert metrics["positive_gate_accept_rate"] == 1.0
    assert metrics["negative_gate_rejection_rate"] == 1.0
    assert metrics["balanced_gate_accuracy"] == 1.0
    assert metrics["false_reject_count"] == 0
    assert metrics["false_accept_count"] == 0


def test_empty_evaluation_is_invalid():
    with pytest.raises(ValueError):
        evaluate_robust_retrieval(SingleHitStore(), [], [{"question": "X"}])
