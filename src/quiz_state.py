"""Durable in-session answers independent from temporary Streamlit widget keys."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

ANSWERS_KEY = "quiz_answers"


def widget_key(question_id: int) -> str:
    return f"_question_{question_id}"


def initialize_quiz(state: MutableMapping[str, Any]) -> None:
    if ANSWERS_KEY not in state:
        state[ANSWERS_KEY] = {}
    if "page" not in state:
        state["page"] = 0
    if "show_results" not in state:
        state["show_results"] = False


def restore_widget(state: MutableMapping[str, Any], question_id: int) -> None:
    """Hydrate a visible radio from durable answers before it is instantiated."""
    state[widget_key(question_id)] = state[ANSWERS_KEY].get(question_id)


def remember_answer(state: MutableMapping[str, Any], question_id: int) -> None:
    """Copy a radio selection to a non-widget key before widgets disappear."""
    value = state.get(widget_key(question_id))
    if value is None:
        state[ANSWERS_KEY].pop(question_id, None)
    elif type(value) is int and 1 <= value <= 5:
        state[ANSWERS_KEY][question_id] = value
    else:
        raise ValueError(f"Invalid response for question {question_id}: {value!r}")


def reset_quiz_state(state: MutableMapping[str, Any], question_ids: list[int]) -> None:
    state[ANSWERS_KEY] = {}
    for question_id in question_ids:
        state.pop(widget_key(question_id), None)
    state["page"] = 0
    state["show_results"] = False
