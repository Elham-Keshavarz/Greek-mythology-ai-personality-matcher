"""Output-level safety checks for grounded generation.

Citation validity is not semantic faithfulness. The checks here ensure only that
LLM output cites an available passage; they do not prove cited factual claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# One accepted passage is currently provided to generation by the passage-level evidence gate.
CITATION = re.compile(r"\[S([0-9]+)\]")
UNCITED_ABSTENTION = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class CitationCheck:
    valid: bool
    citations: tuple[str, ...]
    reason: str


def verify_citations(answer: str, allowed_ids: set[str]) -> CitationCheck:
    """Reject missing, unknown, or malformed numeric source citations.

    Important: this is only a *syntactic* check. Human or model-assisted
    factual entailment must be evaluated separately.
    """
    if not answer.strip():
        return CitationCheck(False, (), "The model returned an empty response.")
    if not allowed_ids:
        return CitationCheck(False, (), "No approved sources were provided to the model.")
    seen = tuple(dict.fromkeys(f"S{num}" for num in CITATION.findall(answer)))
    if not seen:
        return CitationCheck(False, (), "The model answer has no [S1]-style source citation.")
    unknown = sorted(set(seen) - allowed_ids)
    if unknown:
        return CitationCheck(False, seen, "Unknown source citations: " + ", ".join(unknown))
    # Common typo forms are rejected rather than allowing apparent citations
    # that do not correspond to evidence displayed to the user.
    if re.search(r"\[\s*[sS]\s*\d+\s*\]", answer) and any(
        not CITATION.fullmatch(match.group(0))
        for match in re.finditer(r"\[\s*[sS]\s*\d+\s*\]", answer)
    ):
        return CitationCheck(False, seen, "The response contains malformed source citations.")
    return CitationCheck(True, seen, "Citations refer to supplied sources; factual faithfulness is not verified.")
