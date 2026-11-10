"""IPIP 50-item Big-Five scoring and a deterministic, non-clinical archetype matcher."""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Mapping

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
SCORING_DOMAINS = ("extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness")

# IPIP 50-item sample questionnaire, in its published order.
# reverse=True means the item is reverse-keyed for its published factor.
IPIP_ITEMS = (
    ("Am the life of the party.", False),
    ("Feel little concern for others.", True),
    ("Am always prepared.", False),
    ("Get stressed out easily.", True),
    ("Have a rich vocabulary.", False),
    ("Don't talk a lot.", True),
    ("Am interested in people.", False),
    ("Leave my belongings around.", True),
    ("Am relaxed most of the time.", False),
    ("Have difficulty understanding abstract ideas.", True),
    ("Feel comfortable around people.", False),
    ("Insult people.", True),
    ("Pay attention to details.", False),
    ("Worry about things.", True),
    ("Have a vivid imagination.", False),
    ("Keep in the background.", True),
    ("Sympathize with others' feelings.", False),
    ("Make a mess of things.", True),
    ("Seldom feel blue.", False),
    ("Am not interested in abstract ideas.", True),
    ("Start conversations.", False),
    ("Am not interested in other people's problems.", True),
    ("Get chores done right away.", False),
    ("Am easily disturbed.", True),
    ("Have excellent ideas.", False),
    ("Have little to say.", True),
    ("Have a soft heart.", False),
    ("Often forget to put things back in their proper place.", True),
    ("Get upset easily.", True),
    ("Do not have a good imagination.", True),
    ("Talk to a lot of different people at parties.", False),
    ("Am not really interested in others.", True),
    ("Like order.", False),
    ("Change my mood a lot.", True),
    ("Am quick to understand things.", False),
    ("Don't like to draw attention to myself.", True),
    ("Take time out for others.", False),
    ("Shirk my duties.", True),
    ("Have frequent mood swings.", True),
    ("Use difficult words.", False),
    ("Don't mind being the center of attention.", False),
    ("Feel others' emotions.", False),
    ("Follow a schedule.", False),
    ("Get irritated easily.", True),
    ("Spend time reflecting on things.", False),
    ("Am quiet around strangers.", True),
    ("Make people feel at ease.", False),
    ("Am exacting in my work.", False),
    ("Often feel blue.", True),
    ("Am full of ideas.", False),
)


def questionnaire() -> list[dict]:
    """List the 50 items with 1-based IDs; hide scoring details in the UI."""
    return [{"id": idx + 1, "text": text} for idx, (text, _) in enumerate(IPIP_ITEMS)]


def score_big_five(answers: Mapping[int, int]) -> dict[str, float]:
    """Return traits on 0..1, with higher neuroticism indicating more reactivity.

    Published IPIP factor IV measures emotional stability, so it is inverted
    to obtain neuroticism. Factor V is Intellect/Imagination, used as a proxy
    for the broad Openness axis in this entertainment-oriented application.
    """
    if set(answers) != set(range(1, 51)):
        raise ValueError("All 50 question IDs (1..50) must be answered exactly once.")
    if any(type(v) is not int or not 1 <= v <= 5 for v in answers.values()):
        raise ValueError("Each answer must be an integer on the 1..5 scale.")
    totals = dict.fromkeys(SCORING_DOMAINS, 0)
    for idx, (_, reverse) in enumerate(IPIP_ITEMS, start=1):
        domain = SCORING_DOMAINS[(idx - 1) % 5]
        value = 6 - answers[idx] if reverse else answers[idx]
        totals[domain] += value
    normalized = {trait: round((total / 10 - 1) / 4, 4) for trait, total in totals.items()}
    return {
        "openness": normalized["openness"],
        "conscientiousness": normalized["conscientiousness"],
        "extraversion": normalized["extraversion"],
        "agreeableness": normalized["agreeableness"],
        "neuroticism": round(1 - normalized["emotional_stability"], 4),
    }


def load_profiles(path: Path | str | None = None) -> list[dict]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "archetype_profiles.json"
    profiles = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Archetype profiles must be a nonempty JSON list.")
    names = set()
    for p in profiles:
        if not all(k in p for k in ("name", "description", "traits", "rationale", "gender", "category", "source_url")):
            raise ValueError("Profile missing required fields.")
        if p["name"] in names:
            raise ValueError("Duplicate profile name.")
        names.add(p["name"])
        if p["gender"] not in ("female", "male"):
            raise ValueError(f"Invalid mythological gender: {p['name']}")
        if not isinstance(p["category"], str) or not p["category"].strip():
            raise ValueError(f"Missing mythological category: {p['name']}")
        if not isinstance(p["source_url"], str) or not p["source_url"].startswith("https://"):
            raise ValueError(f"Invalid reference URL: {p['name']}")
        if set(p["traits"]) != set(TRAITS):
            raise ValueError(f"Invalid trait keys: {p['name']}")
        if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in p["traits"].values()):
            raise ValueError(f"Invalid trait score: {p['name']}")
    return profiles


def find_matches(
    scores: Mapping[str, float],
    profiles: list[dict] | None = None,
    top_k: int = 3,
    gender: str | None = None,
) -> list[dict]:
    """Rank archetypes by normalized Euclidean distance.

    similarity_index is a heuristic scale, NOT a calibrated probability.
    """
    if set(scores) != set(TRAITS):
        raise ValueError("Exactly five Big Five traits are required.")
    if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError("Scores must be numeric values in [0, 1].")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")
    if gender not in (None, "female", "male"):
        raise ValueError("Gender filter must be 'female', 'male', or None")
    profiles = load_profiles() if profiles is None else profiles
    if gender is not None:
        profiles = [p for p in profiles if p["gender"] == gender]
    if not profiles:
        raise ValueError("No deities available for the selected matching preference")
    result = []
    for profile in profiles:
        distance = sqrt(sum((scores[t] - profile["traits"][t]) ** 2 for t in TRAITS) / len(TRAITS))
        result.append({**profile, "distance": distance, "similarity_index": round((1 - distance) * 100, 1)})
    result.sort(key=lambda p: (p["distance"], p["name"]))
    return result[:top_k]
