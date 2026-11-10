"""English Streamlit app: personality matching plus optional mythology semantic-search chat."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.assessment import TRAITS, find_matches, load_profiles, questionnaire, score_big_five
from src.personalization import explain_match
from src.rag.personalized_answer import FOCUSES, personalized_answer
from src.matching_preferences import (
    GENDER_LABELS,
    GENDER_OPTIONS,
    initialize_preferences,
    matching_gender,
    reset_preferences,
    save_preferences,
)
from src.quiz_state import (
    ANSWERS_KEY,
    initialize_quiz,
    remember_answer,
    reset_quiz_state,
    restore_widget,
    widget_key,
)

SCALE_LABELS = {
    1: "Very inaccurate",
    2: "Moderately inaccurate",
    3: "Neither",
    4: "Moderately accurate",
    5: "Very accurate",
}
ITEMS_PER_PAGE = 10
ITEMS = questionnaire()
TOTAL_PAGES = math.ceil(len(ITEMS) / ITEMS_PER_PAGE)
PROFILES = load_profiles()

st.set_page_config(page_title="Greek Mythology Personality Matcher", page_icon="🏛️", layout="centered")
initialize_quiz(st.session_state)
initialize_preferences(st.session_state)


def reset_quiz() -> None:
    reset_quiz_state(st.session_state, [item["id"] for item in ITEMS])
    reset_preferences(st.session_state)
    st.session_state.pop("personalized_generation", None)
    for key in ("personalized_selected_archetype", "personalized_focus", "personalized_api_consent"):
        st.session_state.pop(key, None)


def save_answer(question_id: int) -> None:
    remember_answer(st.session_state, question_id)


def onboarding_view() -> None:
    st.header("Before we begin")
    st.write("Please choose your gender. This is used **only to select which deities can appear in your matches**, never to change your Big Five personality scores.")
    previous_gender = st.session_state["participant_gender"]
    selected_gender = st.radio(
        "How do you describe your gender?",
        GENDER_OPTIONS,
        index=GENDER_OPTIONS.index(previous_gender) if previous_gender in GENDER_OPTIONS else None,
        format_func=lambda value: GENDER_LABELS[value],
    )

    scope = "all_deities"
    if selected_gender in ("woman", "man"):
        previous_scope = st.session_state["matching_scope"]
        scope = st.radio(
            "Which mythology characters should we consider?",
            ("same_gender", "all_deities"),
            index=0 if previous_scope == "same_gender" else 1,
            format_func=lambda value: (
                "Match with deities traditionally portrayed as my gender"
                if value == "same_gender" else "Match with all 45 deities"
            ),
            help="Your gender is not a personality characteristic. It only filters candidate mythology figures if you choose.",
        )
    elif selected_gender is not None:
        st.info("All 45 deities will be included. Mythological profiles in this collection use traditional male/female portrayals; your gender is not inferred from them.")

    if st.button("Continue to the 50-question quiz", type="primary", disabled=(selected_gender is None)):
        save_preferences(st.session_state, selected_gender, scope)
        st.rerun()

    with st.expander("Browse the 45 mythological deities in this collection"):
        st.dataframe(
            pd.DataFrame([
                {"Name": p["name"], "Traditional portrayal": p["gender"].title(), "Group": p["category"], "Domain": p["description"]}
                for p in PROFILES
            ]),
            hide_index=True,
            use_container_width=True,
        )


def results_view() -> None:
    answers = dict(st.session_state[ANSWERS_KEY])
    scores = score_big_five(answers)
    pool_gender = matching_gender(st.session_state["participant_gender"], st.session_state["matching_scope"])
    eligible_count = sum(1 for p in PROFILES if pool_gender is None or p["gender"] == pool_gender)
    matches = find_matches(scores, PROFILES, gender=pool_gender)
    winner = matches[0]

    st.header(f"Your closest archetype: {winner['name']}")
    st.write(winner["description"])
    st.caption(f"Matched against {eligible_count} of the 45 deities in the collection.")
    st.info(
        "For entertainment and self-reflection only. Deity trait scores are author-designed, "
        "not psychologically validated; similarity is not a probability or diagnosis. "
        "Gender only affects the deity pool, never your Big Five scores."
    )

    st.subheader("Your Big Five profile")
    plot = pd.DataFrame({
        "Trait": [t.replace("_", " ").title() for t in TRAITS],
        "Score": [round(scores[t] * 100, 1) for t in TRAITS],
    }).set_index("Trait")
    st.bar_chart(plot, y="Score", horizontal=True)
    st.caption(
        "IPIP-50 factors are scaled to 0–100. Emotional Stability is inverted to represent "
        "Neuroticism; Intellect/Imagination stands in for Openness."
    )

    st.subheader("Your top 3 mythological matches")
    for rank, profile in enumerate(matches, 1):
        with st.expander(
            f"{rank}. {profile['name']} — similarity index {profile['similarity_index']:.1f}/100",
            expanded=(rank == 1),
        ):
            st.write(profile["description"])
            st.caption(f"{profile['category']} · traditionally portrayed as {profile['gender']}")
            st.write(profile["rationale"])
            st.markdown(f"[Learn about Greek deities]({profile['source_url']})")
            st.caption("Interpretive, manually curated traits — not a validated personality diagnosis.")

    st.divider()
    st.subheader("🧠 Why did I match this archetype?")
    st.caption(
        "An explanation of the five numerical differences is calculated locally. "
        "The character's Big Five scores are creative, author-assigned values—not scientific measurements."
    )
    selected_name = st.selectbox(
        "Which of your top matches would you like to explore?",
        [p["name"] for p in matches], key="personalized_selected_archetype"
    )
    chosen = next(p for p in matches if p["name"] == selected_name)
    explanation = explain_match(scores, chosen)
    st.write(explanation.narrative)
    st.dataframe(
        pd.DataFrame([{
            "Trait": row.label,
            "Your score": f"{row.user:.0%}",
            "Archetype score": f"{row.archetype:.0%}",
            "Absolute difference": f"{row.gap:.0%}",
        } for row in explanation.traits]),
        hide_index=True, use_container_width=True
    )
    st.caption("All five differences contribute to the same formula used by the matcher; this is not a diagnosis.")

    with st.expander("Optional: ask OpenAI for a sourced explanation"):
        from src.rag.llm_client import generate_answer, is_configured
        st.write(
            "Optional LLM commentary combines the already-computed match with one approved "
            "mythology excerpt. It cannot change your score or your matches."
        )
        st.warning(
            "When you click Generate, the selected archetype, its authored profile, "
            "your FIVE calculated Big Five scores, your question, and a short mythology excerpt "
            "are sent to your configured OpenAI API. Raw questionnaire answers and gender are NOT sent. "
            "The service may charge for the request. No request is sent by opening this section."
        )
        focus = st.selectbox("What should the explanation focus on?", tuple(FOCUSES),
                             key="personalized_focus")
        personal_question = st.text_input(
            "Your question about this match",
            value=f"Why was I matched with {selected_name}?",
            max_chars=200,
            help="Ask about your match and differences. This text is sent only when you click Generate."
        )
        consent = st.checkbox(
            "I agree to share my question and five calculated trait scores with OpenAI for this request",
            value=False, key="personalized_api_consent"
        )
        configured = is_configured()
        if not configured:
            st.info("Configure OPENAI_API_KEY and OPENAI_MODEL in .env to enable optional generation.")
        context_key = (selected_name, focus, personal_question, tuple(scores[t] for t in TRAITS))
        if st.button("Generate explanation with OpenAI", type="primary",
                     disabled=not (consent and configured)):
            try:
                with st.spinner("Retrieving an approved mythology excerpt and generating commentary..."):
                    response = personalized_answer(explanation, get_rag_store(),
                                                   generator=generate_answer, focus=focus,
                                                   question=personal_question)
            except Exception as exc:
                st.error(f"Unable to generate explanation: {exc}")
            else:
                st.session_state["personalized_generation"] = (context_key, response)
        previous = st.session_state.get("personalized_generation")
        if previous and previous[0] == context_key and consent:
            response = previous[1]
            if response.mode == "llm":
                st.markdown(response.text)
            else:
                st.info(response.reason or "Only the local explanation is available.")
            if response.hit is not None:
                st.caption("[S1] Project-authored mythology passage. Linked URL is background reading, not a quotation source.")
                st.markdown(f"[S1: {response.hit.chunk.title}]({response.hit.chunk.source_url})")
            st.caption("Valid source IDs do not guarantee that every generated claim is factual.")

    if st.button("Start again", type="primary"):
        reset_quiz()
        st.rerun()


@st.cache_resource(show_spinner="Loading sentence embeddings and FAISS index...")
def get_rag_store():
    from src.rag.pipeline import build_pipeline
    return build_pipeline()


def mythology_chat_view() -> None:
    from dotenv import load_dotenv
    from src.rag.answer_service import answer_question
    from src.rag.llm_client import generate_answer, is_configured

    load_dotenv()
    st.header("📚 Greek Mythology Knowledge Assistant")
    st.write("Ask English questions about Greek mythology. Semantic retrieval runs locally after the initial embedding-model download.")
    st.caption(
        "The knowledge base currently has 45 short project-authored deity profiles and 18 original factual summaries. "
        "These are not full ancient primary texts; the linked pages are references for further verification."
    )
    st.info(
        "Local retrieval is free and needs no API key. LLM mode sends only your chat question and one "
        "approved mythology passage to the configured provider; it does **not** send quiz answers or gender. "
        "The evidence gate is heuristic and generated citations are syntax-checked, not fact-verified."
    )

    configured = is_configured()
    answer_mode = st.radio(
        "Mode", ("Retrieved evidence", "LLM answer"), horizontal=True,
        help="Set OPENAI_API_KEY and OPENAI_MODEL in a local .env file for generated answers."
    )
    if answer_mode == "LLM answer" and not configured:
        st.warning("LLM mode requires OPENAI_API_KEY and OPENAI_MODEL. Add them to .env, or select Retrieved evidence.")

    left, right = st.columns(2)
    with left:
        if st.button("Clear chat"):
            st.session_state["rag_messages"] = []
            st.rerun()
    with right:
        if st.button("Rebuild knowledge index", help="Regenerate embeddings after editing data/knowledge/curated_facts.json"):
            from src.rag.pipeline import build_pipeline
            with st.spinner("Rebuilding vector index from source documents..."):
                build_pipeline(rebuild=True)
            get_rag_store.clear()
            st.rerun()

    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = []

    for message in st.session_state["rag_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("corrections"):
                st.caption("Query normalization: " + ", ".join(
                    f"{item['original']} → {item['corrected']}" for item in message["corrections"]
                ))
            if message.get("reason"):
                with st.expander("Retrieval diagnostics"):
                    st.write(message["reason"])
                    st.caption("Normalized query: " + message.get("normalized_question", ""))
                    st.caption("Evidence-term coverage: " + f"{message.get('coverage', 0):.0%}" + " (heuristic, not confidence)")
                    if message.get("supporting_document_id"):
                        st.caption("Supporting passage: " + message["supporting_document_id"])
            if message.get("sources"):
                label = ("Approved LLM evidence" if message.get("mode") in ("llm", "generation_rejected", "llm_abstained")
                         else "Retrieved candidates (not necessarily answer support)")
                with st.expander(label):
                    for src in message["sources"]:
                        st.markdown(f"**[{src['id']}] {src['title']}** — cosine similarity {src['score']:.3f}")
                        st.markdown(f"[Background reference]({src['url']})")
                        st.caption(src["note"] + " · The indexed excerpt is project-authored, not a quote from this page.")

    question = st.chat_input("For example: How are Apollo and Artemis related?")
    if question:
        st.session_state["rag_messages"].append({"role": "user", "content": question})
        try:
            store = get_rag_store()
            if answer_mode == "LLM answer" and not configured:
                raise RuntimeError("Configure API credentials or select Retrieved evidence mode.")
            generation = generate_answer if answer_mode == "LLM answer" else None
            result = answer_question(question, store, generator=generation)
        except Exception as exc:
            st.session_state["rag_messages"].pop()
            st.error(f"Unable to answer: {exc}")
            st.caption("Check credentials and dependencies with python -m src.check_environment; the index is built via python -m src.rag.build_index.")
        else:
            sources = [
                {"id": f"S{i}", "title": hit.chunk.title,
                 "url": hit.chunk.source_url, "score": hit.score,
                 "note": hit.chunk.source_note}
                for i, hit in enumerate(result.hits, 1)
            ]
            st.session_state["rag_messages"].append({
                "role": "assistant", "content": result.text, "sources": sources,
                "normalized_question": result.normalized_question,
                "corrections": [{"original": c.original, "corrected": c.corrected} for c in result.corrections],
                "reason": result.reason, "coverage": result.evidence_coverage,
                "supporting_document_id": result.supporting_document_id,
                "mode": result.mode, "citations": list(result.citations),
            })
            st.rerun()


st.title("🏛️ Greek Mythology Personality Matcher")
st.write("Discover which Greek mythological archetype resembles your answers to a public-domain Big Five questionnaire.")
st.caption("The personality quiz needs no API key. Mythology chat uses local embeddings; remote LLM generation is optional.")
with st.sidebar:
    st.header("Navigate")
    feature = st.radio("Feature", ("Personality Quiz", "Mythology RAG Chat"), key="feature_selection")
    st.divider()
    st.header("About this project")
    st.markdown("**Method:** IPIP-50 → Big Five scores → matching with 45 deities")
    st.markdown("**Collection:** 25 traditionally female + 20 traditionally male figures, including Olympians, Titans, and other deities")
    st.markdown("**Status:** 45-deity matching + grounded RAG + optional personalized LLM explanations")
    st.markdown("[Questionnaire source](https://ipip.ori.org/new_ipip-50-item-scale.htm)")
    st.markdown("[Greek deity directory](https://www.theoi.com/greek-mythology/greek-gods.html)")
    if st.session_state["onboarding_completed"]:
        st.divider()
        st.write(f"**Gender:** {GENDER_LABELS[st.session_state['participant_gender']]}")
        if st.button("Change matching preferences"):
            st.session_state["onboarding_completed"] = False
            st.rerun()

if feature == "Mythology RAG Chat":
    mythology_chat_view()
elif not st.session_state["onboarding_completed"]:
    onboarding_view()
elif st.session_state["show_results"]:
    results_view()
else:
    page = st.session_state["page"]
    start = page * ITEMS_PER_PAGE
    end = min(start + ITEMS_PER_PAGE, len(ITEMS))
    answers = st.session_state[ANSWERS_KEY]
    done = len(answers)
    st.progress(done / len(ITEMS), text=f"{done}/{len(ITEMS)} answers completed")
    st.subheader(f"Questions {start + 1}–{end} of {len(ITEMS)}")
    st.caption("Choose how accurately each statement describes you. No answer is pre-selected.")

    for item in ITEMS[start:end]:
        question_id = item["id"]
        restore_widget(st.session_state, question_id)
        st.radio(
            f"{question_id}. {item['text']}",
            options=[1, 2, 3, 4, 5],
            index=None,
            format_func=lambda n: f"{n} · {SCALE_LABELS[n]}",
            horizontal=True,
            key=widget_key(question_id),
            on_change=save_answer,
            args=(question_id,),
        )

    completed_page = all(item["id"] in answers for item in ITEMS[start:end])
    left, right = st.columns(2)
    with left:
        if st.button("Previous", disabled=(page == 0), use_container_width=True):
            st.session_state["page"] -= 1
            st.rerun()
    with right:
        if page + 1 < TOTAL_PAGES:
            if st.button("Next", type="primary", disabled=not completed_page, use_container_width=True):
                st.session_state["page"] += 1
                st.rerun()
        else:
            if st.button("See my archetype", type="primary", disabled=done != len(ITEMS), use_container_width=True):
                st.session_state["show_results"] = True
                st.rerun()
