# Passage-level Evidence Validation — Learning Guide

**Practice notebook:** `notebooks/04_evidence_validation_exploration.ipynb` (runs without an API key or a FAISS download).

**Scope:** Improve RAG retrieval safety without changing the existing personality quiz, Big Five scoring, 45 archetypes, or questionnaire state handling.

The implementation **does not** train a QA/NLI classifier or use an LLM as a judge. It adds a transparent **intent-aware heuristic** on top of the existing local Sentence Transformers + FAISS semantic retrieval. It checks whether a *single retrieved passage* covers the entities and relevant concepts of a question. It can still accept unsupported claims and reject good evidence. Treat its status as a **gate decision**, never as correctness probability.

## What changed and why

In the older union-of-passages approach, a query could pass the evidence gate if one retrieved document mentioned a deity and **another unrelated document** contained a matching topic word. This allowed false support for some impossible questions. Meanwhile exact lexical matching incorrectly rejected understandable paraphrases such as *grieve → grief* and *personify → personification*.

The passage-level approach:

```text
Question → deity-name typo normalization → Sentence Transformers embedding
    → FAISS top 20 → hybrid reranking → top 4 chunks
    → passage-level evidence gate
        ├── REJECT: insufficient_evidence → abstain, no API call
        └── ACCEPT: supported_candidates
             ├── no LLM configured → show retrieved evidence + diagnostics
             └── LLM configured → send only a supporting passage to LLM
```

The phrase `supported_candidates` means *plausibly answerable from a passage*; it does **not** mean that the answer or the source is correct.

## New and updated files and functions

| File | Function | Why it exists |
|---|---|---|
| `src/rag/retrieval_terms.py` | `canonical()` | Normalize common token forms for lexical reranking. |
| | `content_tokens()` | Remove question fillers and preserve domain tokens such as `sea` and `owl`. |
| | `named_entities()` | Recognize known deity names in a query or passage. |
| `src/rag/evidence_gate.py` | `concepts()` | Map a small documented synonym list (`personification → personify`, `grieve → grief`). No learning occurs here. |
| | `validate_evidence()` | Check entity mentions, topic coverage and question intent in **one passage at a time**. Produce `EvidenceDecision`. |
| `src/rag/retriever.py` | `RobustRetriever.retrieve()` | Normalize, retrieve, rerank, invoke selected evidence gate, return source diagnostics. Rank logic is still hybrid. |
| `src/rag/legacy_evidence_gate.py` | `validate_legacy()` | Retained only for a *controlled ablation*: the legacy union-of-passages gate on current retrieved candidates. **Never use it to generate answers.** |
| `src/rag/answer_service.py` | `answer_question()` | Abstain if evidence fails; otherwise, send only the identified supporting chunk to the optional LLM. Other FAISS hits remain visible in retrieval-only mode for inspection. |
| `src/rag/evaluate.py` | `evaluate_robust_retrieval()` | Separately measure retrieval ranking and evidence gate decisions on positive/negative prompts. |
| | `load_suite()` | Load the original regression suite or the new challenge prompts. |
| | `main()` | CLI runner with `--suite`, `--compare` and `--output`. |
| `data/evaluation/retrieval_challenge.json` | *(data)* | Separately authored challenge prompts, **not** an independently sampled/annotated academic benchmark. |
| `app.py` | `mythology_chat_view()` | Show supporting passage ID among retrieval diagnostics; original quiz views stay the same. |

## The main idea, step by step

A user asks: **What model of car did Apollo drive?**

1. FAISS may retrieve an Apollo description and a Helios document mentioning the solar chariot.
2. The question has entities `{Apollo}` and relevant concepts `{model, car, drive}`.
3. `validate_evidence()` analyzes each candidate **individually**: an Apollo document does not establish a car/model, and a Helios chariot passage is not evidence about Apollo.
4. No passage meets the evidence test → `insufficient_evidence` → the generator is not called.

Compare: **What does Hypnos personify?**

1. The query has the entity `Hypnos` and the concept `personify`.
2. A retrieved passage says *Hypnos is the personification of sleep*.
3. `concepts()` maps `personification` and `personify` to a common concept. The passage can be flagged as a candidate for response.

The `why` question about Demeter uses the explicit paired concepts `separation` and `grief` from the curated summary. This is a limited domain-specific rule; it does not prove causal entailment for arbitrary documents.

## What the metrics mean

**Ranking metrics** apply only to questions with an expected document:

- `hit@5` = fraction of queries where an expected source is in the first five returned chunks.
- `mrr@5` = mean reciprocal rank; rank 1 contributes 1, rank 2 contributes 0.5, etc.

**Evidence gate metrics** apply to both answerable and intentionally unsupported prompts:

- `positive_gate_accept_rate` = fraction of answerable prompts accepted (sensitivity / recall on the labeled small set).
- `negative_gate_rejection_rate` = fraction of unsupported prompts rejected (specificity on the labeled small set).
- `balanced_gate_accuracy` = mean of those two rates.
- `false_reject_count` and `false_accept_count` = error counts (different failure modes).

These do **not** measure final answer factuality, faithfulness, citation correctness or deployment accuracy.

## Run on Windows

From the project root:

```powershell
.venv\Scripts\python -m pip install -r requirements-rag.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m src.rag.ask "What does Hypnos personify?"
.venv\Scripts\python -m src.rag.ask "Why did Demeter grieve when Persephone disappeared?"
.venv\Scripts\python -m src.rag.ask "What model of car did Apollo drive?"
.venv\Scripts\python -m src.rag.evaluate --suite both --compare --output reports/local_retrieval.json
.venv\Scripts\streamlit run app.py
```

The index doesn't require rebuilding because no chunk content or embedding model has changed. Run `python -m src.rag.build_index` only when you first prepare the index, change the source texts, or change the embedding model. A Hugging Face authentication warning is not an error; the public embedding model can be downloaded without a token, subject to rate limits.

## Evaluation limits and next experiments

- The `retrieval_regression.json` is a **19-prompt regression set** (13 positives, 6 unsupported).
- The `retrieval_challenge.json` is a **22-prompt challenge set** (12 positives, 10 unsupported) authored for error discovery; it is not independently sampled and shouldn't be reported as a published benchmark.
- The `--compare` path runs legacy union-of-passages gating on the **same current candidate passages and shared tokenizer**. It is not a historical byte-for-byte reproduction of legacy union-gate. To reproduce a historical implementation exactly, run its original code in a separate environment with the same model and index.
- Unit tests mock/construct retrieval hits. They validate gate logic, not actual embedding quality. The complete FAISS + sentence-embedding evaluation requires local model execution on your computer.
- Unanswered-but-retrieved prompts can still be accepted, especially when they use unusual paraphrases or require reasoning across passages. A trained QA/NLI verifier and an independently annotated benchmark are future experiments, **not features of passage-level evidence gate**.
- The small hand-built mythology corpus contains 45 profiles and 18 curated short summaries, not encyclopedic coverage. Linked pages are background reading rather than claim-level verification.

### Learning exercises

1. In `src/rag/evidence_gate.py`, trace how `What model of car did Apollo drive?` produces `requested_critical` and why Helios is excluded.
2. Trace how `grieve` and `disappeared` become two concepts in the Demeter passage.
3. Run `--suite challenge --compare` and keep a small table of **false accepts** and **false rejects**, including source titles. Do not tune on the challenge examples and then report them as held-out accuracy.
4. Try a question that requires combining information from separate documents. Observe that the conservative single-passage gate may abstain; that's a known limitation rather than automatically a bug.
