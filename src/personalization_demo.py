"""Small offline personalization smoke demo with SYNTHETIC scores (not real quiz answers)."""
from __future__ import annotations

from src.assessment import find_matches, load_profiles
from src.personalization import explain_match


def main() -> None:
    profiles = load_profiles()
    example = next(p for p in profiles if p["name"] == "Athena")
    synthetic = dict(example["traits"])
    synthetic.update(openness=0.88, conscientiousness=0.92, extraversion=0.46, agreeableness=0.53, neuroticism=0.23)
    match = find_matches(synthetic, profiles)[0]
    explanation = explain_match(synthetic, match)
    print("OFFLINE DEMO — synthetic scores, not the user's questionnaire results")
    print("Top archetype:", explanation.name)
    print(explanation.narrative)
    print("No API key or external request is needed for this demo.")


if __name__ == "__main__":
    main()
