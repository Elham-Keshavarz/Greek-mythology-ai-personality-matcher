import pytest

from src.assessment import (
    IPIP_ITEMS, SCORING_DOMAINS, TRAITS, find_matches, load_profiles,
    questionnaire, score_big_five,
)


def test_ipip_questionnaire_structure():
    items = questionnaire()
    assert len(items) == 50
    assert [q["id"] for q in items] == list(range(1, 51))
    assert len(set(q["text"] for q in items)) == 50
    assert len(SCORING_DOMAINS) == 5
    assert len(IPIP_ITEMS) == 50


def test_neutral_answers_produce_neutral_traits():
    assert score_big_five({i: 3 for i in range(1, 51)}) == dict.fromkeys(TRAITS, 0.5)


def test_correct_keying_and_emotional_stability_inversion():
    maximum_keyed = {i: (1 if reverse else 5) for i, (_, reverse) in enumerate(IPIP_ITEMS, 1)}
    result = score_big_five(maximum_keyed)
    assert result == {
        "openness": 1.0, "conscientiousness": 1.0,
        "extraversion": 1.0, "agreeableness": 1.0, "neuroticism": 0.0,
    }
    minimum_keyed = {i: 6 - value for i, value in maximum_keyed.items()}
    result2 = score_big_five(minimum_keyed)
    assert result2 == {
        "openness": 0.0, "conscientiousness": 0.0,
        "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 1.0,
    }


@pytest.mark.parametrize("invalid", [
    {}, {i: 3 for i in range(1, 50)},
    {**{i: 3 for i in range(1, 51)}, 51: 3},
    {**{i: 3 for i in range(1, 51)}, 1: 0},
    {**{i: 3 for i in range(1, 51)}, 1: True},
    {**{i: 3 for i in range(1, 51)}, 1: 3.0},
])
def test_reject_bad_answers(invalid):
    with pytest.raises(ValueError):
        score_big_five(invalid)


def test_profiles_have_exactly_five_normalized_traits():
    profiles = load_profiles()
    assert len(profiles) == 45
    assert len(set(p["name"] for p in profiles)) == len(profiles)
    assert all(set(p["traits"]) == set(TRAITS) for p in profiles)
    assert sum(p["gender"] == "female" for p in profiles) == 25
    assert sum(p["gender"] == "male" for p in profiles) == 20
    assert all(p["source_url"].startswith("https://www.theoi.com/") for p in profiles)


def test_identical_profile_is_best_match():
    profiles = load_profiles()
    athena = next(p for p in profiles if p["name"] == "Athena")
    matches = find_matches(athena["traits"], profiles)
    assert matches[0]["name"] == "Athena"
    assert matches[0]["distance"] == 0
    assert matches[0]["similarity_index"] == 100
    assert len(matches) == 3


def test_reject_malformed_matching_inputs():
    with pytest.raises(ValueError):
        find_matches({"openness": 0.5})
    with pytest.raises(ValueError):
        find_matches(dict.fromkeys(TRAITS, 0.5), top_k=0)


def test_gender_filter_applies_only_to_candidates():
    profiles = load_profiles()
    scores = next(p for p in profiles if p["name"] == "Athena")["traits"]
    women = find_matches(scores, gender="female", top_k=25)
    men = find_matches(scores, gender="male", top_k=20)
    everyone = find_matches(scores, top_k=45)
    assert len(women) == 25
    assert len(men) == 20
    assert len(everyone) == 45
    assert all(p["gender"] == "female" for p in women)
    assert all(p["gender"] == "male" for p in men)
    assert women[0]["name"] == "Athena"
    assert men[0]["name"] != "Athena"
    assert everyone[0]["name"] == "Athena"


def test_gender_filter_handles_invalid_values():
    with pytest.raises(ValueError):
        find_matches(dict.fromkeys(TRAITS, 0.5), gender="non_binary")
    with pytest.raises(ValueError):
        find_matches(dict.fromkeys(TRAITS, 0.5), profiles=[], gender="male")
