# legacy union-gate — Robust Retrieval (Learning Guide)

## What we observed in raw FAISS baseline

- `Who is Athena?` returned useful Athena evidence.
- Misspelled `apolo` / `atemis` failed even though the correctly spelled question succeeded.
- The question `What smartphone did Zeus use?` returned *Zeus* excerpts even though none answered it.

The third item is **not a FAISS bug**: nearest-neighbor retrieval returns a nearest passage, not an answerability judgment.

## New architecture

```text
User question
  -> query_normalizer.normalize_query()
  -> store.search()                    FAISS dense candidates (up to 20)
  -> retriever.RobustRetriever         semantic + token + entity rerank
  -> evidence gate                    heuristic support check; abstain if missing
  -> answer_service.answer_question() retrieval preview OR optional LLM
  -> Streamlit chat + sources + diagnostic info
```

**The personality quiz, its Big Five scoring, and its 45 mythology archetypes remain unchanged.**

## File-by-file, function-by-function

### `src/rag/query_normalizer.py`

- `deity_names()` reads the 45 known names once and caches them.
- `normalize_query(question, names=None)` scans words. It leaves exact names and common English words alone. It corrects an **unambiguous single-edit** typo in a known deity name using Damerau–Levenshtein distance from RapidFuzz. It returns `(clean_query, corrections)`.
- `Correction` stores `original` and `corrected` for visibility.

A typo like `apolo` has an edit distance of 1 from `Apollo` (missing the second `l`). `atemis` has an edit distance of 1 from `Artemis` (missing `r`).

This is a **conservative name corrector**, not a general spell-checker. It avoids guessing on short/ambiguous tokens; users should be able to see every correction.

### `src/rag/retriever.py`

- `canonical(word)` maps a small set of variants/synonyms to shared terms, e.g. `traveling -> travel`. This dictionary is hand-authored, not learned.
- `content_tokens(text, names)` extracts topical words, filtering common grammar words and recognized deity names.
- `named_entities(text, names)` detects which known deity names appear in a passage.
- `RobustRetriever.retrieve(question, top_k=4, min_score=0.25)` is the main algorithm:
  1. Normalize known deity names.
  2. Ask FAISS for up to 20 candidate passages using the corrected query.
  3. Rerank candidates with their **cosine similarity**, **token overlap** and **named-deity coverage**. Known-deity queries receive a stronger entity weight. This is a lightweight *heuristic hybrid reranker*, **not** a trained CrossEncoder.
  4. Retain the top passages above the minimum cosine threshold.
  5. Check whether those passages address the question beyond merely matching a deity name. For relationship queries with multiple names, require a passage that co-mentions them and includes relationship evidence.
  6. Return `RetrievalResult` with hits, corrected query, status, a reason and approximate topical-token coverage.

Ranking and gating are deliberately separate. A high cosine similarity score alone does not establish that an excerpt answers the question.

### `src/rag/answer_service.py`

- `build_prompt()` sends only retrieved passages as untrusted evidence and instructs the LLM to cite only supplied reference labels.
- `answer_question()` uses `RobustRetriever` first. If there is insufficient evidence, it returns `ABSTAIN` and **does not call the remote LLM**. Otherwise it returns passage excerpts or sends them to the optional LLM provider.
- `Answer` now also contains `normalized_question`, `corrections`, `reason` and `evidence_coverage`.

### `src/rag/vector_store.py`

Unchanged. This is still FAISS `IndexFlatIP` over L2-normalized Sentence Transformer vectors; its raw scores are **cosine similarities**, not confidence or answerability probabilities.

### `src/rag/evaluate.py`

- `evaluate_retrieval()` computes baseline Hit@k and MRR@k for direct FAISS search.
- `evaluate_robust_retrieval()` computes the same for reranked retrieval, plus positive-question gate acceptance and negative-question rejection.
- `--compare` prints both on the **same 13 positive + 6 negative** corpus-derived examples. This is a **regression set**, not a held-out scientific benchmark.

### `app.py`

The quiz flow is unchanged. The **Mythology RAG Chat** now displays corrected deity names, retrieval diagnostics and the evidence-only/abstention outcome. The "Rebuild knowledge index" button now forces an actual disk-index rebuild.

## Try it locally (from the project root)

Install:

```powershell
.venv\Scripts\python -m pip install -r requirements-rag.txt
```

Build the index. If the existing corpus and model are unchanged, the FAISS index is reusable:

```powershell
.venv\Scripts\python -m src.rag.build_index
```

Run the fixed five-question demo:

```powershell
.venv\Scripts\python -m src.rag.ask "Who is Athena and what is she known for?"
.venv\Scripts\python -m src.rag.ask "how are apolo and atemis related?"
.venv\Scripts\python -m src.rag.ask "What is the relationship between Apollo and Artemis?"
.venv\Scripts\python -m src.rag.ask "Which Greek deity represents clever communication and traveling?"
.venv\Scripts\python -m src.rag.ask "What smartphone did Zeus use?"
```

The last question should abstain *even if its diagnostic sources are about Zeus*. Those source snippets are candidate evidence, **not an answer**.

Run the before/after metrics:

```powershell
.venv\Scripts\python -m src.rag.evaluate --compare
.venv\Scripts\python -m pytest -q
.venv\Scripts\streamlit run app.py
```

## What the numbers mean

- `Hit@5`: fraction of positive questions whose designated reference document is in the first five candidates.
- `MRR@5`: mean inverse rank of the first designated reference document, zero if missing.
- `positive_gate_accept_rate`: fraction of in-scope benchmark questions permitted past the evidence gate; **high** is desirable, but does not prove the answer is correct.
- `negative_gate_rejection_rate`: fraction of deliberately unsupported questions rejected by the gate; **high** is desirable, but doesn't prove all out-of-scope questions will be rejected.

These measures help compare **relative changes**; they are not a claim of performance on real-world queries.

## Known limitations and next experiments

1. **Small knowledge base:** 45 short profiles + 18 notes. Missing detailed myths cannot be recovered by changing the ranking algorithm.
2. **Heuristic keyword coverage:** novel paraphrases may be falsely rejected; superficial token matches may be falsely accepted. Review both failure types.
3. **No trained re-ranker:** a real CrossEncoder can improve passage relevance, but must be evaluated on your hardware/data before adopting.
4. **No proven hallucination-free generation:** the prompt helps, but LLM citation and faithfulness must be tested separately.
5. **No broad spelling correction:** only known, unambiguous deity-name typos are attempted.
6. **No new entity embeddings:** the index remains compatible with earlier builds because we did not change indexed text or the model.

Suggested exercise: add five new *unseen* paraphrases and five **unanswerable but mythology-related** questions to a separate evaluation file; evaluate false positives and false negatives before tuning thresholds.
