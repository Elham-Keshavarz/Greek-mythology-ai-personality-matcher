"""Passage-level, intent-aware evidence checks for a *small curated* RAG corpus.

This is NOT an entailment model, an answer-confidence estimate, or a guarantee.
FAISS remains responsible for semantic candidate retrieval. This gate asks whether
one candidate passage covers the entities and concepts required by the question.
It never combines half an answer from unrelated passages to assert support.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .retrieval_terms import content_tokens, named_entities
from .vector_store import Hit

# Small transparent synonym map. These are domain-oriented approximations,
# not a trained semantic reasoner. Inspect / extend with new error examples.
CONCEPTS = {
    "grieve": "grief", "grieved": "grief", "grieving": "grief",
    "mourn": "grief", "mourning": "grief", "sadness": "grief", "sorrow": "grief",
    "disappear": "separation", "disappeared": "separation", "disappearance": "separation",
    "vanish": "separation", "vanished": "separation", "abduction": "separation",
    "abducted": "separation", "kidnapped": "separation", "removal": "separation",
    "removed": "separation", "taken": "separation",
    "personification": "personify", "personified": "personify", "personifying": "personify",
    "personifies": "personify", "embodies": "personify", "embody": "personify", "embodiment": "personify",
    "link": "family", "linked": "family", "connected": "family", "connection": "family",
    "relationship": "family", "relative": "family", "relatives": "family",
    "twin": "family", "twins": "family", "children": "family", "child": "family",
    "mother": "family", "father": "family", "parents": "family", "parent": "family",
    "sister": "family", "brother": "family", "siblings": "family",
    "wife": "family", "husband": "family", "consort": "family", "marriage": "family",
    "reunion": "reunion", "reunited": "reunion",
    "symbols": "symbol", "symbolic": "symbol", "symbolizes": "symbol", "emblem": "symbol",
    "messages": "messenger", "message": "messenger", "messengers": "messenger",
    "carries": "messenger", "carried": "messenger", "carry": "messenger",
    "delivers": "messenger", "delivered": "messenger", "delivery": "messenger",
    "rules": "rule", "ruler": "rule", "ruled": "rule", "reigns": "rule",
    "governs": "rule", "govern": "rule",
    "driven": "drive", "drove": "drive", "drives": "drive", "driving": "drive",
    "automobile": "car", "cars": "car", "vehicle": "car",
    "phones": "smartphone", "phone": "smartphone", "mobile": "smartphone",
    "laptops": "laptop", "computers": "computer", "brands": "brand",
    "passwords": "password", "emailing": "email",
}

# A subject requested explicitly by the question may not be silently dropped.
# These are *critical topic concepts*, not an exhaustive list of impossible myths.
CRITICAL_TOPICS = frozenset({
    "smartphone", "laptop", "computer", "internet", "password",
    "car", "email", "social", "media", "account", "brand", "model",
})

RELATION_CUES = frozenset({"family", "birth", "ancestry", "descendant", "ancestor", "reunion"})
CAUSE_CUES = frozenset({"because", "therefore", "caused", "cause", "reason", "result", "following"})
RELATION_WORDS = re.compile(r"\b(related|relationship|link|parents?|siblings?|twins?|brother|sister|family|connection)\b", re.I)
YEARS = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|[0-9]{1,3}\s?(?:bc|bce|ad|ce))\b", re.I)


def concepts(text: str, names: tuple[str, ...]) -> set[str]:
    """Content words canonicalized to small, readable ontology concepts."""
    return {CONCEPTS.get(token, token) for token in content_tokens(text, names)}


@dataclass(frozen=True)
class EvidenceDecision:
    status: str
    reason: str
    coverage: float
    supporting_document_id: str | None = None
    matched_concepts: tuple[str, ...] = ()


def passage_concepts(text: str, names: tuple[str, ...]) -> set[str]:
    """Conservatively infer two symbol categories from explicit examples.

    A source must explicitly call the objects symbols; simply mentioning an
    owl or an olive tree in an unrelated story cannot support the question.
    This is a narrow corpus rule rather than a general taxonomy or NLI model.
    """
    found = concepts(text, names)
    if "symbol" in found:
        if "owl" in found:
            found.add("bird")
        if {"olive", "tree"}.issubset(found):
            found.add("plant")
    return found


def validate_evidence(question: str, hits: Sequence[Hit], names: tuple[str, ...]) -> EvidenceDecision:
    """Return a candidate-support assessment; no claim of verified factual truth.

    Each hit is checked *independently*. Relevance and answerability are not
    equivalent: this is a conservative heuristic that can still make mistakes.
    """
    if not hits:
        return EvidenceDecision("no_evidence", "No retrieved passage available.", 0.0)

    entities = named_entities(question, names)
    topics = concepts(question, names)
    relationship = len(entities) >= 2 and bool(RELATION_WORDS.search(question))
    why_question = bool(re.match(r"\s*why\b", question, re.I))
    exact_date = bool(re.search(r"\b(what year|which year|exact date|what date)\b", question, re.I))
    requested_critical = topics & CRITICAL_TOPICS
    if not entities and not topics:
        return EvidenceDecision("insufficient_evidence", "The question gives no identifiable mythology subject or answerable topic.", 0.0)
    candidates: list[tuple[float, float, Hit, tuple[str, ...]]] = []

    for hit in hits:
        # Do not use source URLs or profile tags in evidence: source URLs can
        # match names and tokens accidentally, despite giving no answer.
        text = hit.chunk.text
        passage_entities = named_entities(text, names)
        if entities and not entities.issubset(passage_entities):
            continue
        passage_topics = passage_concepts(text, names)
        matched = topics & passage_topics
        coverage = len(matched) / len(topics) if topics else 1.0
        if requested_critical and not requested_critical.issubset(passage_topics):
            continue
        if topics and (not matched or coverage < 0.60):
            continue
        if relationship and not RELATION_CUES.intersection(passage_topics):
            continue
        if exact_date and not YEARS.search(text):
            continue
        if why_question:
            # In the currently tiny corpus, explicit cause words or the
            # specific paired separation/grief concepts allow a candidate.
            # The latter is a documented domain rule, *not proof of causation*.
            has_causal_language = bool(CAUSE_CUES & set(re.findall(r"[a-z]+", text.lower())))
            has_demeter_cause = {"separation", "grief"}.issubset(passage_topics)
            if not (has_causal_language or has_demeter_cause):
                continue
        candidates.append((coverage, hit.score, hit, tuple(sorted(matched))))

    if not candidates:
        # Coverage is computed from best SINGLE passage mentioning the named
        # entities, not the union of unrelated passages.
        best = 0.0
        for hit in hits:
            if entities and not entities.issubset(named_entities(hit.chunk.text, names)):
                continue
            got = topics & passage_concepts(hit.chunk.text, names)
            best = max(best, len(got) / len(topics) if topics else 1.0)
        return EvidenceDecision(
            "insufficient_evidence",
            "No single retrieved passage supports the question's subject, requested entities and intent.",
            best,
        )

    coverage, _, hit, matched = max(candidates, key=lambda item: (item[0], item[1]))
    return EvidenceDecision(
        "supported_candidates",
        "A single passage covers the requested concepts; answer correctness still requires verification.",
        coverage,
        hit.chunk.document_id,
        matched,
    )
