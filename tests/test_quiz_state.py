from src.quiz_state import (
    ANSWERS_KEY,
    initialize_quiz,
    remember_answer,
    reset_quiz_state,
    restore_widget,
    widget_key,
)


def test_answers_survive_pagination_and_widget_cleanup():
    state = {}
    initialize_quiz(state)
    for question_id in range(1, 11):
        restore_widget(state, question_id)
        assert state[widget_key(question_id)] is None
        state[widget_key(question_id)] = (question_id - 1) % 5 + 1
        remember_answer(state, question_id)

    state["page"] = 1
    for question_id in range(1, 11):
        state.pop(widget_key(question_id))

    for question_id in range(11, 21):
        restore_widget(state, question_id)
        assert state[widget_key(question_id)] is None
        state[widget_key(question_id)] = 4
        remember_answer(state, question_id)

    state["page"] = 0
    for question_id in range(1, 11):
        restore_widget(state, question_id)
        assert state[widget_key(question_id)] == (question_id - 1) % 5 + 1
    assert len(state[ANSWERS_KEY]) == 20


def test_unselect_answer_and_reset():
    state = {}
    initialize_quiz(state)
    restore_widget(state, 1)
    state[widget_key(1)] = 5
    remember_answer(state, 1)
    state[widget_key(1)] = None
    remember_answer(state, 1)
    assert 1 not in state[ANSWERS_KEY]
    state[widget_key(1)] = 2
    remember_answer(state, 1)
    state["page"] = 3
    state["show_results"] = True
    reset_quiz_state(state, list(range(1, 51)))
    assert state[ANSWERS_KEY] == {}
    assert state["page"] == 0
    assert not state["show_results"]
    assert widget_key(1) not in state
