"""Hybrid semantic + token/known-entity retrieval with a conservative evidence gate.

This is a heuristic quality gate, NOT proof of answerability or a calibrated
probability. Keep abstentions visible and benchmark both false accepts/rejects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .query_normalizer import Correction, deity_names, normalize_query
from .retrieval_terms import content_tokens, named_entities, RELATION_TERMS
from .evidence_gate import EvidenceDecision, validate_evidence
from .vector_store import Hit

@dataclass
class RetrievalResult:
    original_question: str
    normalized_question: str
    corrections: list[Correction]
    hits: list[Hit]
    status: str
    reason: str
    evidence_coverage: float
    supporting_document_id: str | None = None
    matched_concepts: tuple[str, ...] = ()


class RobustRetriever:
    def __init__(self, store: Any, names: tuple[str, ...] | None = None, candidate_k: int = 20,
                 evidence_gate: Callable[[str, list[Hit], tuple[str, ...]], EvidenceDecision] = validate_evidence):
        self.store = store
        self.names = names if names is not None else deity_names()
        self.candidate_k = candidate_k
        self.evidence_gate = evidence_gate

    def retrieve(self, question: str, top_k: int = 4, min_score: float = 0.25) -> RetrievalResult:
        if not question.strip():
            raise ValueError("Question must not be blank")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be 1 through 20")
        normalized, corrections = normalize_query(question, self.names)
        entities = named_entities(normalized, self.names)
        terms = content_tokens(normalized, self.names)
        candidates = self.store.search(normalized, top_k=max(self.candidate_k, top_k), min_score=-1.0)

        def rank(hit: Hit) -> float:
            doc_text = f"{hit.chunk.title} {hit.chunk.text}"
            doc_entities = named_entities(doc_text, self.names)
            doc_terms = content_tokens(doc_text, self.names)
            lexical = len((terms - RELATION_TERMS) & doc_terms) / max(len(terms - RELATION_TERMS), 1)
            entity_overlap = len(entities & doc_entities) / len(entities) if entities else 0.0
            curated_bonus = 0.015 if hit.chunk.source_type == "curated_summary" else 0.0
            return ((0.50 * hit.score + 0.20 * lexical + 0.30 * entity_overlap)
                    if entities else (0.75 * hit.score + 0.25 * lexical)) + curated_bonus

        ranked = sorted(candidates, key=lambda h: (-rank(h), -h.score, h.chunk.id))
        hits = [h for h in ranked if h.score >= min_score][:top_k]
        if not hits:
            return RetrievalResult(question, normalized, corrections, [], "no_evidence", "No documents passed the semantic similarity cutoff.", 0.0)

        # A passage must cover the question on its own; merging unrelated
        # passages is not evidence that any one of them answers the question.
        decision = self.evidence_gate(normalized, hits, self.names)
        return RetrievalResult(question, normalized, corrections, hits,
                               decision.status, decision.reason, decision.coverage,
                               decision.supporting_document_id, decision.matched_concepts)
