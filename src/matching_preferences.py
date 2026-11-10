"""Optional identity-based pool selection; gender never changes personality scores."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

GENDER_OPTIONS = ("woman", "man", "non_binary", "prefer_not_to_say")
MATCHING_OPTIONS = ("same_gender", "all_deities")
GENDER_LABELS = {
    "woman": "Woman",
    "man": "Man",
    "non_binary": "Non-binary",
    "prefer_not_to_say": "Prefer not to say",
}


def initialize_preferences(state: MutableMapping[str, Any]) -> None:
    state.setdefault("participant_gender", None)
    state.setdefault("matching_scope", "same_gender")
    state.setdefault("onboarding_completed", False)


def matching_gender(gender: str | None, scope: str) -> str | None:
    """Return a deity pool filter, without inferring personality from gender."""
    if gender not in GENDER_OPTIONS:
        raise ValueError("Choose a supported gender option")
    if scope not in MATCHING_OPTIONS:
        raise ValueError("Choose a supported matching scope")
    if scope == "all_deities":
        return None
    if gender == "woman":
        return "female"
    if gender == "man":
        return "male"
    return None


def save_preferences(state: MutableMapping[str, Any], gender: str, scope: str) -> None:
    matching_gender(gender, scope)
    state["participant_gender"] = gender
    state["matching_scope"] = scope if gender in ("woman", "man") else "all_deities"
    state["onboarding_completed"] = True


def reset_preferences(state: MutableMapping[str, Any]) -> None:
    state["participant_gender"] = None
    state["matching_scope"] = "same_gender"
    state["onboarding_completed"] = False
