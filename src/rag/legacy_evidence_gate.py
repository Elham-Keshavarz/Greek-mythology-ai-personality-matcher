"""Legacy union-of-passages gate, retained ONLY to compare on the SAME retrieved hits.

This implementation intentionally keeps the old union-of-passages behavior.
It should never be used for generated answers. Updated passage-level evidence gate tokenization is
shared; hence this is a controlled ablation, not an exact historical-code replay.
"""
from __future__ import annotations

from .evidence_gate import EvidenceDecision
from .retrieval_terms import content_tokens, named_entities, RELATION_TERMS, RELATION_EVIDENCE
from .vector_store import Hit


def validate_legacy(question: str, hits: list[Hit], names: tuple[str, ...]) -> EvidenceDecision:
    if not hits:
        return EvidenceDecision("no_evidence", "No retrieved passages.", 0.0)
    entities = named_entities(question, names)
    terms = content_tokens(question, names)
    evidence = [f"{hit.chunk.title} {hit.chunk.text}" for hit in hits]
    non_relation_terms = terms - RELATION_TERMS
    matched_terms = set().union(*(content_tokens(text, names) for text in evidence)) & non_relation_terms
    coverage = len(matched_terms) / len(non_relation_terms) if non_relation_terms else 1.0
    if len(entities) >= 2 and terms & RELATION_TERMS:
        relation_supported = any(
            entities <= named_entities(text, names)
            and bool(content_tokens(text, names) & RELATION_EVIDENCE)
            for text in evidence
        )
        if not relation_supported:
            return EvidenceDecision("insufficient_evidence", "legacy gate: no explicit relationship passage.", coverage)
    elif non_relation_terms and not matched_terms:
        return EvidenceDecision("insufficient_evidence", "legacy gate: no matching evidence subject.", coverage)
    elif entities and not any(entities & named_entities(text, names) for text in evidence):
        return EvidenceDecision("insufficient_evidence", "legacy gate: missing entity.", coverage)
    return EvidenceDecision("supported_candidates", "legacy gate: union-based token coverage.", coverage)
