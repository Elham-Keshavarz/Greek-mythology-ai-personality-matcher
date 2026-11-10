# Personalized Mythology RAG — Learning Guide

## What changed

This implementation fixes a single-vocabulary weakness in the evidence gate and connects the quiz results to **optional, explicitly consented** personalized LLM commentary. It does **not** alter the 50 questions, scoring, gender matching preferences, 45 figures, embeddings, FAISS implementation, or non-personalized mythology RAG chat.

## Files and purpose

| File | Responsibility |
|---|---|
| `src/rag/evidence_gate.py` | New `passage_concepts` adds `bird` for explicit owl symbolism and `plant` for explicit olive-tree symbolism **only when the single passage contains symbol context**. No union across excerpts. |
| `src/personalization.py` | `explain_match`: pure Python computation of all five trait differences using the **same formula as the matcher**. Returns an immutable `MatchExplanation`. No model, storage, or external calls. |
| `src/rag/personalized_answer.py` | Finds and checks a profile passage for the specific deity; builds a source-constrained prompt with calculated trait gaps; calls an optional generator and validates `[S1]` citations. Missing evidence and malformed citations fall back to deterministic explanation. |
| `app.py` | Shows a local explanation and trait comparison table below the existing top-three results. Optional OpenAI section requires user consent for sharing the **five derived scores**. |
| `src/personalization_demo.py` | Runs a visible **synthetic** example offline; does not send personal data to OpenAI. |
| `tests/test_personalization.py` | Verifies numerical agreement, the Athena case, fail-closed behavior, source isolation, offline privacy, and citation rejection. |

## Data flow

```text
50 quiz answers (Streamlit session only)
        │
        ▼
existing score_big_five()
        │
        ├─► existing find_matches() ─► top 3 profiles (ranking unchanged)
        │                                   │
        └───────────────────────────────────┴─► explain_match()
                                                  │
                                                  ├─► local score table + narrative (no API)
                                                  │
                                   only after explicit consent + button
                                                  ▼
                                   FAISS search for *selected* deity
                                                  │
                                    evidence gate on one profile passage
                                                  │
                                 ┌─ no passage ─► keep local explanation
                                 └─ approved passage
                                          │
                                          ▼
                            build source-limited OpenAI prompt
                            (five derived scores, no raw answers/gender)
                                          │
                                          ▼
                            verify [S1] or reject/fallback
```

## The formula

For each of the five traits, the local difference is `abs(user_trait - deity_trait)`. The matcher uses:

`distance = sqrt(sum((user_trait - deity_trait)**2 for each trait) / 5)`

`similarity_index = round((1 - distance) * 100, 1)`

**Never call the index a probability, scientific personality fit, diagnosis, or clinically validated measurement.** The deity values are human-authored interpretations.

## Why the Athena test now passes

The challenge question asks for **plant** and **bird** symbols. The source excerpt says Athena is associated with the **owl** and **olive tree** as symbols. A lexical gate previously did not align category terms (plant/bird) with named examples. `passage_concepts` adds those two categories on the *passage side*, only when the examples **and** the word "symbol" coexist. The validator still demands the **same passage** supports all requested concepts, so two separate documents each containing one example cannot be merged into fake evidence. This fixes one observed false negative, not a general theory of symbolism.

## Generated explanation reliability and privacy

- Quiz answers are not sent to the model; only **derived scores**, the short typed question, selected profile text/rationale, and one indexed project-authored mythology excerpt are sent **after explicit opt-in**.
- The OpenAI prompt tells the model to preserve provided numbers and mark mythology information with `[S1]`. The citation checker validates **source IDs**, not entailment. Every generated paragraph should still be reviewed for fidelity.
- The five locally calculated differences and match ranking are always displayed independently; the LLM cannot change them.
- The app does not intentionally persist quiz answers; Streamlit session state retains results for the active session, and the configured API provider receives prompt contents when used.
- The indexed content is **project-authored summaries**, not scraped primary-source quotations. The linked URL is reading context.

## Commands (Windows PowerShell from project root)

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m pytest -q tests/test_personalization.py
.venv\Scripts\python -m src.personalization_demo
.venv\Scripts\python -m src.rag.ask "What plant and bird are traditional Athena symbols?"
.venv\Scripts\python -m src.rag.evaluate --suite both --compare
.venv\Scripts\python -m streamlit run app.py
```

For the local explanations, **no API key is required**. For optional OpenAI commentary, ensure `.env` is beside `app.py` and set `OPENAI_API_KEY` and `OPENAI_MODEL`. Don't upload `.env`.

## Tests and remaining limitations

`pytest` checks unit behavior without paid provider requests. The embedding-dependent retrieval benchmark can be rerun locally. Document the actual observed results and do not attribute them to tests that were never executed. Before publication, review generated claims manually, add independent/adversarial evaluation cases, capture real screenshots and confirm CI.
