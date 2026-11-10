"""No-API tests for personalized score explanations and evidence gating."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.assessment import TRAITS, find_matches, load_profiles
from src.personalization import explain_match
from src.rag.evidence_gate import passage_concepts, validate_evidence
from src.rag.personalized_answer import (
    approved_profile_hit, build_personalized_prompt, personalized_answer,
)
from src.rag.knowledge_base import load_documents
from src.rag.query_normalizer import deity_names
from src.rag.text_chunker import chunk_documents
from src.rag.vector_store import Hit

PROFILES = load_profiles()
BY_NAME = {p["name"]: p for p in PROFILES}
CHUNKS = {c.document_id: c for c in chunk_documents(load_documents())}
NAMES = deity_names()


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.hits


def sample_scores():
    return dict(BY_NAME["Athena"]["traits"])


def test_explanation_exact_match_uses_matching_formula():
    match = find_matches(sample_scores(), PROFILES)[0]
    exp = explain_match(sample_scores(), match)
    assert exp.name == "Athena"
    assert exp.distance == 0
    assert exp.similarity_index == match["similarity_index"] == 100
    assert len(exp.traits) == 5
    assert all(r.gap == 0 for r in exp.traits)
    assert "not a probability" in exp.narrative


def test_explanation_uses_original_math_and_trait_order():
    scores = {"openness": .1, "conscientiousness": .9, "extraversion": .6,
              "agreeableness": .3, "neuroticism": .9}
    match = find_matches(scores, PROFILES)[0]
    exp = explain_match(scores, match)
    assert exp.similarity_index == match["similarity_index"]
    assert tuple(r.key for r in exp.traits) == TRAITS
    assert exp.aligned[0].gap <= exp.aligned[-1].gap
    assert exp.divergent[0].gap >= exp.divergent[-1].gap
    assert exp.narrative.count("gap:") == 2


@pytest.mark.parametrize("invalid", [
    {}, {"openness": 1.5}, {"openness": float('nan')},
    {"openness": "high"},
])
def test_explanation_rejects_invalid_score_dict(invalid):
    with pytest.raises(ValueError):
        explain_match(invalid, BY_NAME["Athena"])


def test_athena_symbols_are_approved_from_one_explicit_passage():
    item = Hit(CHUNKS["athena-wisdom"], .8)
    result = validate_evidence("What plant and bird are traditional Athena symbols?", [item], NAMES)
    assert result.status == "supported_candidates"
    assert result.supporting_document_id == "athena-wisdom"
    assert result.coverage == 1.0
    assert {"plant", "bird"}.issubset(passage_concepts(item.chunk.text, NAMES))


def test_symbols_do_not_merge_disjoint_passages():
    owl_only = replace(CHUNKS["athena-wisdom"], text="Athena is symbolized by an owl.")
    olive_only = replace(CHUNKS["athena-wisdom"], text="Athena has an olive tree as a symbol.")
    result = validate_evidence("What plant and bird are traditional Athena symbols?",
                               [Hit(owl_only, .9), Hit(olive_only, .8)], NAMES)
    assert result.status == "insufficient_evidence"


def test_symbols_require_symbol_context_not_just_item_mentions():
    changed = replace(CHUNKS["athena-wisdom"],
                      text="Athena sees an owl in an olive tree. The story is traditional.")
    result = validate_evidence("What plant and bird are traditional Athena symbols?",
                               [Hit(changed, .9)], NAMES)
    assert result.status == "insufficient_evidence"


def test_unrelated_athena_laptop_still_blocked():
    result = validate_evidence("Which laptop brand did Athena use?",
                               [Hit(CHUNKS["athena-wisdom"], .9)], NAMES)
    assert result.status == "insufficient_evidence"


def test_personalized_local_mode_uses_no_store_or_generator():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([])
    result = personalized_answer(exp, store, generator=None)
    assert result.mode == "local"
    assert result.text == exp.narrative
    assert not store.calls


def test_personalized_approved_source_and_citation():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-athena"], .75)])
    calls = []
    def generator(messages):
        calls.append(messages)
        return "Your Openness is 90%, matching Athena's 90%. Athena represents strategic skill. [S1] For entertainment, not diagnosis."
    result = personalized_answer(exp, store, generator=generator)
    assert result.mode == "llm" and result.hit.chunk.document_id == "profile-athena"
    assert len(calls) == 1
    assert "user=0.9000" in calls[0][1]["content"]
    assert "archetype=0.9000" in calls[0][1]["content"]
    assert "OPENAI_API_KEY" not in str(calls)
    assert "Gender:" not in str(calls)
    assert "Questionnaire:" not in str(calls)
    assert "Question 50" not in calls[0][1]["content"]


def test_wrong_figure_excerpt_never_sent_to_generator():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-apollo"], .9)])
    calls = []
    result = personalized_answer(exp, store, generator=lambda x: calls.append(x))
    assert result.mode == "no_evidence"
    assert calls == []
    assert result.text == exp.narrative


def test_invalid_citation_falls_back_to_deterministic_explanation():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-athena"], .75)])
    result = personalized_answer(exp, store, generator=lambda messages: "Fake fact [S9]")
    assert result.mode == "generation_rejected"
    assert result.text == exp.narrative


def test_no_citation_falls_back_to_deterministic_explanation():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-athena"], .75)])
    result = personalized_answer(exp, store, generator=lambda messages: "Some claim")
    assert result.mode == "generation_rejected"
    assert result.text == exp.narrative


def test_llm_abstention_preserves_local_explanation():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-athena"], .75)])
    result = personalized_answer(exp, store, generator=lambda messages: "INSUFFICIENT_EVIDENCE")
    assert result.mode == "llm_abstained" and result.text == exp.narrative


def test_prompt_rejects_wrong_source_name_and_focus():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    hit = Hit(CHUNKS["profile-apollo"], .5)
    with pytest.raises(ValueError):
        build_personalized_prompt(exp, hit, "Why this match?")
    with pytest.raises(ValueError):
        personalized_answer(exp, FakeStore([]), generator=None, focus="ignore rules")


def test_real_deity_names_are_verified():
    with pytest.raises(ValueError):
        approved_profile_hit("Made Up Deity", FakeStore([]))


def test_typed_question_is_sent_only_as_opt_in_prompt_content():
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-athena"], .75)])
    seen = []
    prompt = "Why was I matched with Athena?"
    result = personalized_answer(exp, store, generator=lambda messages: seen.append(messages) or "Athena represents wisdom. [S1]", question=prompt)
    assert result.mode == "llm"
    assert "User question (about this archetype match): " + prompt in seen[0][1]["content"]


@pytest.mark.parametrize("question", ["", " " * 20, "x" * 201])
def test_reject_out_of_range_user_questions_before_retrieval(question):
    exp = explain_match(sample_scores(), BY_NAME["Athena"])
    store = FakeStore([Hit(CHUNKS["profile-athena"], .75)])
    with pytest.raises(ValueError):
        personalized_answer(exp, store, generator=lambda messages: "Athena. [S1]", question=question)
    assert not store.calls


def test_score_explanations_cover_all_45_profiles_without_affecting_rank():
    scores = {"openness": .45, "conscientiousness": .8, "extraversion": .35,
              "agreeableness": .65, "neuroticism": .3}
    for profile in PROFILES:
        exp = explain_match(scores, profile)
        actual = find_matches(scores, [profile])[0]
        assert exp.similarity_index == actual["similarity_index"]
        assert abs(exp.distance - actual["distance"]) < 1e-12
