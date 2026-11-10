"""Conservative typo correction for known mythology names, not general spell checking."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz.distance import DamerauLevenshtein

from .knowledge_base import PROFILE_PATH

TOKEN = re.compile(r"\b[A-Za-z]+\b")
# Safeguard common English words that happen to be close to short deity names.
COMMON_WORDS = frozenset('''about after again all also among and answer are asked away before being between both carry chat could craft deity deities does every first from gods greek had have hello here heres known later learn light like love many more most names only other their there these those through today travel under which while whose with woman women would year yours question power rates times tell what when where which who whose why will world your used uses using really still please hear heart'''.split())


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str


@lru_cache(maxsize=1)
def deity_names() -> tuple[str, ...]:
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return tuple(p["name"] for p in profiles)


def normalize_query(query: str, names: tuple[str, ...] | None = None) -> tuple[str, list[Correction]]:
    """Correct single-edit, unambiguous name typos; preserve all other words."""
    if not query.strip():
        return query, []
    names = names if names is not None else deity_names()
    candidates = {name.lower(): name for name in names}
    corrections: list[Correction] = []

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in candidates or len(lower) < 5 or lower in COMMON_WORDS:
            return word
        best = [name for name in candidates if abs(len(name) - len(lower)) <= 1
                and DamerauLevenshtein.distance(lower, name, score_cutoff=1) == 1]
        if len(best) == 1:
            corrected = candidates[best[0]]
            corrections.append(Correction(word, corrected))
            return corrected
        return word

    return TOKEN.sub(replace, query), corrections
