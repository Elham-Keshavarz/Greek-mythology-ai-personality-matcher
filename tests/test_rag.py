from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from src.rag.answer_service import ABSTAIN, answer_question, build_prompt
from src.rag.knowledge_base import Document, load_documents
from src.rag.text_chunker import chunk_documents
from src.rag.vector_store import fingerprint


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.query = None

    def search(self, question, top_k=4, min_score=0.25):
        self.query = question
        return self.hits


def sample_docs():
    return [Document(id="a", title="Athena", text="Athena represents wisdom and strategy in Greek myths.",
                     source_url="https://www.theoi.com/Olympios/Athena.html", source_type="curated_summary",
                     source_note="Original editorial summary")]


def test_knowledge_corpus_has_45_profiles_and_18_facts():
    documents = load_documents()
    assert len(documents) == 63
    assert len({d.id for d in documents}) == 63
    assert sum(d.source_type == "project_profile" for d in documents) == 45
    assert sum(d.source_type == "curated_summary" for d in documents) == 18


def test_chunking_preserves_sources_and_overlap():
    document = Document(id="example", title="Example", text="a b c d e f g h",
                        source_url="https://example.com", source_type="curated_summary", source_note="note")
    chunks = chunk_documents([document], max_words=5, overlap_words=2)
    assert [c.text for c in chunks] == ["a b c d e", "d e f g h"]
    assert all(c.source_url == "https://example.com" for c in chunks)


def test_bad_chunk_size():
    with pytest.raises(ValueError):
        chunk_documents(sample_docs(), 10, 10)


def test_fingerprint_changes_with_content_or_model():
    chunks = chunk_documents(sample_docs())
    assert fingerprint(chunks, "a") != fingerprint(chunks, "b")
    assert fingerprint(chunks, "a") != fingerprint(chunk_documents([Document(**{**sample_docs()[0].to_dict(), "text": "changed"})]), "a")


def test_refuse_to_fabricate_without_evidence():
    result = answer_question("Who are they?", FakeStore([]))
    assert result.text == ABSTAIN
    assert result.mode == "no_evidence"


def test_retrieval_only_does_not_call_llm():
    from src.rag.vector_store import Hit
    hits = [Hit(chunk_documents(sample_docs())[0], 0.8)]
    store = FakeStore(hits)
    result = answer_question("Who is Athena?", store)
    assert result.mode == "retrieval_only"
    assert "not an LLM-generated answer" in result.text
    assert result.hits == hits


def test_llm_receives_cited_evidence():
    from src.rag.vector_store import Hit
    hits = [Hit(chunk_documents(sample_docs())[0], 0.8)]
    prompts = []
    def fake_llm(messages):
        prompts.extend(messages)
        return "Athena represents wisdom. [S1]"
    result = answer_question("Who is Athena?", FakeStore(hits), generator=fake_llm)
    assert result.mode == "llm"
    assert "[S1]" in result.text
    assert "Athena represents wisdom" in prompts[1]["content"]
    assert "untrusted" in prompts[0]["content"]


def test_bad_question_and_empty_llm():
    with pytest.raises(ValueError):
        answer_question("  ", FakeStore([]))
    from src.rag.vector_store import Hit
    hits = [Hit(chunk_documents(sample_docs())[0], 0.8)]
    result = answer_question("Who is Athena?", FakeStore(hits), generator=lambda _: "  ")
    assert result.mode == "generation_rejected"


def test_real_faiss_index_with_fake_embedding_model():
    faiss = pytest.importorskip("faiss")
    from src.rag.vector_store import FaissStore

    class FakeEmbedder:
        model_name = "fake-semantic-v1"
        def encode(self, texts):
            results = []
            for text in texts:
                lower = text.lower()
                results.append([1.0, 0.0] if "athena" in lower or "wisdom" in lower else [0.0, 1.0])
            return np.asarray(results, dtype=np.float32)

    docs = sample_docs() + [Document(id="z", title="Zeus", text="Zeus commands thunder and the sky.",
                                    source_url="https://www.theoi.com/Olympios/Zeus.html", source_type="curated_summary",
                                    source_note="Original editorial summary")]
    chunks = chunk_documents(docs)
    with TemporaryDirectory() as d:
        store = FaissStore.load_or_build(chunks, FakeEmbedder(), index_dir=Path(d))
        hits = store.search("Tell me about Athena", min_score=0.2)
        assert hits[0].chunk.document_id == "a"
        assert hits[0].score == pytest.approx(1.0)
        reload = FaissStore.load_or_build(chunks, FakeEmbedder(), index_dir=Path(d))
        assert reload.index.ntotal == 2
        assert reload.search("Tell me about Zeus", min_score=0.2)[0].chunk.document_id == "z"


def test_retrieval_benchmark_calculates_hit_and_mrr():
    from src.rag.evaluate import evaluate_retrieval
    from src.rag.vector_store import Hit
    docs = chunk_documents(sample_docs())
    class Stub:
        def search(self, *args, **kwargs):
            return [Hit(docs[0], 0.9)]
    results = evaluate_retrieval(Stub(), [{"question": "Athena?", "expected_document_id": "a"}], k=3)
    assert results == {"count": 1, "hit@3": 1.0, "mrr@3": 1.0}
