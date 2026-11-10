"""Deterministic, transparent matching explanation; no external API involved.

This module never infers a diagnosis or treats a similarity score as a probability.
Every displayed trait difference follows the exact normalized Euclidean matcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

from src.assessment import TRAITS

TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}


@dataclass(frozen=True)
class TraitComparison:
    key: str
    label: str
    user: float
    archetype: float
    gap: float


@dataclass(frozen=True)
class MatchExplanation:
    name: str
    similarity_index: float
    distance: float
    aligned: tuple[TraitComparison, ...]
    divergent: tuple[TraitComparison, ...]
    traits: tuple[TraitComparison, ...]
    description: str
    rationale: str
    source_url: str
    narrative: str


def explain_match(scores: Mapping[str, float], profile: Mapping) -> MatchExplanation:
    """Explain the existing match mathematically, without changing its ranking."""
    if set(scores) != set(TRAITS):
        raise ValueError("Exactly the five Big Five scores are required")
    traits = profile.get("traits")
    if not isinstance(traits, Mapping) or set(traits) != set(TRAITS):
        raise ValueError("Profile must provide exactly the five Big Five traits")
    if any(type(x) not in (float, int) or not 0 <= x <= 1 for x in (*scores.values(), *traits.values())):
        raise ValueError("All trait scores must be numeric and between zero and one")
    for field in ("name", "description", "rationale", "source_url"):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError(f"Profile missing {field}")
    if not profile["source_url"].startswith("https://"):
        raise ValueError("Reference URL must be HTTPS")

    rows = tuple(
        TraitComparison(t, TRAIT_LABELS[t], float(scores[t]), float(traits[t]), abs(float(scores[t])-float(traits[t])))
        for t in TRAITS
    )
    closest = tuple(sorted(rows, key=lambda r: (r.gap, r.key)))
    farthest = tuple(sorted(rows, key=lambda r: (-r.gap, r.key)))
    distance = sqrt(sum(r.gap ** 2 for r in rows) / len(rows))
    similarity = round((1 - distance) * 100, 1)
    narrative = (
        f"Your closest measured trait to {profile['name']} is {closest[0].label} "
        f"(you: {closest[0].user:.0%}; archetype: {closest[0].archetype:.0%}; "
        f"gap: {closest[0].gap:.0%}). "
        f"Your largest difference is {farthest[0].label} "
        f"(you: {farthest[0].user:.0%}; archetype: {farthest[0].archetype:.0%}; "
        f"gap: {farthest[0].gap:.0%}). "
        f"The similarity index is {similarity:.1f}/100, derived from all five "
        "normalized trait differences. It is not a probability, clinical measure, "
        "or validated claim about your personality."
    )
    return MatchExplanation(
        name=profile["name"], similarity_index=similarity, distance=distance,
        aligned=closest, divergent=farthest, traits=rows,
        description=profile["description"], rationale=profile["rationale"],
        source_url=profile["source_url"], narrative=narrative,
    )
