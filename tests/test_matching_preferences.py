import pytest

from src.matching_preferences import (
    initialize_preferences, matching_gender, reset_preferences, save_preferences,
)


def test_gender_and_matching_modes():
    assert matching_gender("woman", "same_gender") == "female"
    assert matching_gender("man", "same_gender") == "male"
    assert matching_gender("woman", "all_deities") is None
    assert matching_gender("man", "all_deities") is None
    assert matching_gender("non_binary", "same_gender") is None
    assert matching_gender("prefer_not_to_say", "same_gender") is None


@pytest.mark.parametrize("gender,scope", [("unknown", "all_deities"), ("man", "invalid")])
def test_invalid_preferences(gender, scope):
    with pytest.raises(ValueError):
        matching_gender(gender, scope)


def test_preferences_persist_independently_of_quiz_answers():
    state = {"quiz_answers": {1: 4, 2: 3}, "page": 3}
    initialize_preferences(state)
    assert state["onboarding_completed"] is False
    save_preferences(state, "woman", "same_gender")
    assert state["participant_gender"] == "woman"
    assert state["matching_scope"] == "same_gender"
    assert state["quiz_answers"] == {1: 4, 2: 3}
    save_preferences(state, "man", "all_deities")
    assert state["quiz_answers"] == {1: 4, 2: 3}
    assert state["matching_scope"] == "all_deities"
    reset_preferences(state)
    assert state["onboarding_completed"] is False
    assert state["participant_gender"] is None
    assert state["quiz_answers"] == {1: 4, 2: 3}


def test_nonbinary_uses_all_deities_even_if_same_gender_requested():
    state = {}
    initialize_preferences(state)
    save_preferences(state, "non_binary", "same_gender")
    assert state["matching_scope"] == "all_deities"
    assert matching_gender(state["participant_gender"], state["matching_scope"]) is None
