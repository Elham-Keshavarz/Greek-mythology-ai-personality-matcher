"""Fast, offline tests for passage-level evidence gate evidence logic (not embedding accuracy)."""
from __future__ import annotations

import pytest

from src.rag.answer_service import ABSTAIN, answer_question
from src.rag.evidence_gate import concepts, validate_evidence
from src.rag.knowledge_base import load_documents
from src.rag.query_normalizer import deity_names, normalize_query
from src.rag.text_chunker import chunk_documents
from src.rag.vector_store import Hit

CHUNKS = {c.document_id: c for c in chunk_documents(load_documents())}
NAMES = deity_names()


def hit(document_id: str, score: float = 0.72) -> Hit:
    return Hit(CHUNKS[document_id], score)


@pytest.mark.parametrize("question,docs,expected_doc", [
    ("Why did Demeter grieve when Persephone disappeared?", ["demeter-persephone"], "demeter-persephone"),
    ("What does Hypnos personify?", ["hypnos-sleep"], "hypnos-sleep"),
    ("What is the relationship between Apollo and Artemis?", ["apollo-artemis"], "apollo-artemis"),
    ("Which Greek god uses a trident and rules the sea?", ["poseidon-seas"], "poseidon-seas"),
    ("Who carries messages between the Olympian gods?", ["hermes-messenger"], "hermes-messenger"),
    ("Who is Athena?", ["athena-wisdom"], "athena-wisdom"),
    ("Which deity represents clever communication and traveling?", ["hermes-messenger"], "hermes-messenger"),
])
def test_positive_passage_covered(question, docs, expected_doc):
    decision = validate_evidence(question, [hit(x) for x in docs], NAMES)
    assert decision.status == "supported_candidates"
    assert decision.supporting_document_id == expected_doc
    assert decision.coverage >= 0.6


@pytest.mark.parametrize("question,docs", [
    ("What model of car did Apollo drive?", ["profile-apollo", "profile-helios", "apollo-delphi"]),
    ("What smartphone did Zeus use?", ["zeus-sky", "profile-zeus"]),
    ("What was Athena's laptop brand?", ["athena-wisdom"]),
    ("How are Hera and Atlas related?", ["hera-marriage", "profile-atlas"]),
    ("What year was Hypnos born?", ["hypnos-sleep"]),
    ("Tell me", ["athena-wisdom"]),
    ("What are Hermes's social media accounts?", ["hermes-messenger"]),
])
def test_unsupported_passage_abstains(question, docs):
    decision = validate_evidence(question, [hit(x) for x in docs], NAMES)
    assert decision.status == "insufficient_evidence"
    assert decision.supporting_document_id is None


def test_separate_passages_cannot_form_fake_joint_support():
    # Old legacy union-gate union could match terms from Apollo and 'drives' from Helios.
    decision = validate_evidence("What model of car did Apollo drive?",
                                 [hit("profile-apollo"), hit("profile-helios")], NAMES)
    assert decision.status == "insufficient_evidence"
    assert decision.coverage == 0.0


def test_deity_name_correction_integrates_with_gate():
    normalized, corrections = normalize_query("how are apolo and atemis related?")
    assert len(corrections) == 2
    result = validate_evidence(normalized, [hit("apollo-artemis")], NAMES)
    assert result.status == "supported_candidates"


class FixedStore:
    def __init__(self, hits):
        self.hits = hits
    def search(self, *args, **kwargs):
        return self.hits


def test_gate_does_not_call_generator_for_car_query():
    calls = []
    answer = answer_question("What model of car did Apollo drive?",
        FixedStore([hit("profile-apollo", .80), hit("profile-helios", .70)]),
        generator=lambda messages: calls.append(messages) or "Wrong")
    assert answer.mode == "insufficient_evidence" and answer.text == ABSTAIN
    assert calls == []


def test_generation_only_receives_grounded_passage_with_matching_citations():
    prompts = []
    def generator(messages):
        prompts.append(messages)
        return "Hypnos personifies sleep. [S1]"
    ans = answer_question("What does Hypnos personify?",
                          FixedStore([hit("profile-zeus", .92), hit("hypnos-sleep", .80)]), generator=generator)
    assert ans.mode == "llm"
    assert len(ans.hits) == 1
    assert ans.hits[0].chunk.document_id == "hypnos-sleep"
    assert "Hypnos" in prompts[0][1]["content"]
    assert "Zeus" not in prompts[0][1]["content"]


def test_concept_mapping_is_explicit_not_learned():
    assert {"grief", "separation"} == concepts("grieve disappeared", NAMES)
    assert "personify" in concepts("personification", NAMES)
