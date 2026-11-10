# Grounded LLM Answers and Citation Validation — Learning Guide

This is a **learning project**, not a production-grade fact-verification system. This document explains the actual generation and citation-checking code. The existing quiz and 45-deity matching functions are unchanged.

## 1. What changed?

The generation path builds on retrieval and evidence validation. **Generation is checked separately**: only the single vetted evidence passage reaches the LLM; the returned text is checked for valid `[S1]` source references; invalid/no citations are withheld; and provider calls are opt-in.

```text
Question
   |              Local only: no API key required
   v
Query normalization -> FAISS retrieval -> hybrid rerank
   |
   v
Passage-level evidence gate
   |---- REJECT -> no API request -> insufficient-evidence message
   |
   +---- ACCEPT -> Retrieval-only mode: display raw evidence
                   OR
                   LLM mode (user chooses; API credentials configured)
                         |
                         v
                   build_prompt (one approved passage, cited [S1])
                         |
                         v
                   OpenAI-compatible chat completion
                         |
                         v
                   verify_citations
                   |     |            |
               valid   bad IDs    model abstains
                   |     |            |
               answer   withhold     abstain
               + [S1]   untrusted    safely
```

## 2. Files and functions

### `src/rag/answer_service.py`

- `Answer`: data object for the display text, mode, evidence, and `citations`.
- `build_prompt(question, hits)`: creates system/user messages. Excerpts are **untrusted reference data**; instructions embedded in them are not to be followed. The prompt explicitly says the model must answer only from provided information and cite IDs.
- `answer_question(question, store, generator=None)`: retrieves and checks evidence **before** any call to `generator`. Without a generator it returns the original evidence, exactly like earlier versions. With a generator, it passes **one vetted passage** to the model; it rejects unsupported citations and maps the literal `INSUFFICIENT_EVIDENCE` sentinel to a safe abstention.

### `src/rag/grounding.py`

- `CITATION`: a regex for citations such as `[S1]`.
- `verify_citations(answer, allowed_ids)`: checks whether a generated answer contains at least one source ID and whether all IDs belong to the supplied passages. A hallucinated `[S9]` or missing citation leads to **`generation_rejected`**.
- `CitationCheck`: structured outcome with the reason for rejection and extracted citations.

**Important:** citation *format* is not factual *faithfulness*. A model can write a false statement followed by a perfectly valid `[S1]`. This verifier does **not** prove the answer is supported. To make claims about faithfulness, independently annotate and evaluate factual claims or use a separately validated evidence-entailment approach.

### `src/rag/llm_client.py`

- `is_configured()`: loads `.env` if available, then checks that an API key and model name exist.
- `generate_answer(messages)`: uses the `openai` Python SDK (`chat.completions.create`) with an optional provider `OPENAI_BASE_URL`, a 40-second timeout, and one retry. Returns model text. Known provider failures are reported without leaking the API key.
- `LLMConfigurationError` and `LLMProviderError`: distinct errors for missing configuration and network/provider failures.

No model training is involved. This uses an externally hosted **pretrained** LLM. The library is installed by `requirements.txt`; installation itself does not make paid API calls.

### `src/rag/ask.py`

- `main(argv=None)`: CLI accepts a quoted question plus optional `--llm`. Without `--llm`, it **never calls** a hosted model. This is useful for comparing Retrieval against Generation on the same question.

### `src/rag/evaluate_generation.py`

- `audit(store, questions, generator)`: runs labeled positive/negative examples through the same `answer_question` orchestration, stores the text/citations/mode, and marks every answer `NOT_REVIEWED` for human fact checking.
- `main(argv=None)`: by default **does not run live**. You must explicitly use `--live`, have a valid `.env`, and accept provider costs. This tool intentionally does **not** report a faithfulness score.

### `app.py`

The sidebar still offers **Personality Quiz** and **Mythology RAG Chat**. The chat offers **Retrieved evidence** and **LLM answer**. In LLM mode, the UI identifies accepted LLM answers, rejected outputs, abstentions, and supporting sources. It does not share quiz answers, gender or Big Five scores with an external API.

### `tests/test_llm.py`

Offline tests use a fake vector store and fake generators. They check: accepted citations, invalid IDs, no-citation failure, safe abstention, **no LLM calls on evidence-gate rejection**, isolated source passages, CLI mode configuration, and no unexpected live evaluation calls.

## 3. Windows PowerShell setup

Start inside the extracted project folder containing `app.py`:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.check_environment
.venv\Scripts\python -m pytest -q
```

If you already created `.venv`, **do not recreate it**. All dependencies are in the single `requirements.txt` (quiz + FAISS + Sentence Transformers + OpenAI SDK).

Create a local settings file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Inside `.env`, set your **own** API key and model name supported by your provider. `OPENAI_BASE_URL` is optional for OpenAI-compatible gateways. Keep `.env` private; it is excluded via `.gitignore`. API billing is separate from a ChatGPT subscription.

> The local embedding model's Hugging Face warning about unauthenticated requests is **not** an OpenAI API error. An `HF_TOKEN` is optional for model-download rate limits, and it is unrelated to `OPENAI_API_KEY`.

## 4. Verify the two modes

```powershell
# Offline retrieval; no API key required
.venv\Scripts\python -m src.rag.ask "How are Apollo and Artemis related?"

# Only if you explicitly want a hosted LLM call and configured .env
.venv\Scripts\python -m src.rag.ask --llm "How are Apollo and Artemis related?"

# The existing evidence gate should abstain *before* LLM generation
.venv\Scripts\python -m src.rag.ask --llm "What smartphone did Zeus use?"

# Run UI
.venv\Scripts\streamlit run app.py
```

Run the passage-level evidence gate retrieval benchmark too; it should remain comparable:

```powershell
.venv\Scripts\python -m src.rag.evaluate --suite both --compare
```

Run the new generation contract tests without a key:

```powershell
.venv\Scripts\python -m pytest tests/test_llm.py -q
.venv\Scripts\python -m src.rag.evaluate_generation
```

**Optional, paid live audit** (small sample first):

```powershell
.venv\Scripts\python -m src.rag.evaluate_generation --live --limit 3 --output reports/local_llm_audit.json
```

The live-audit report checks expected *answer versus abstain* behavior and citation formatting. It does **not** establish that the generated answer is factually supported. For every generated answer, compare each factual claim with the included passage and external primary mythology sources before reporting a faithfulness metric.

## 5. Key limitations and next experiments

1. **Small knowledge base:** 45 short profiles + 18 original summaries; the displayed Theoi links are background references, **not** retrieved page contents or primary quotes.
2. **Heuristic evidence gate:** one-sentence checks may miss legitimate multi-hop answers and can still accept weakly related passages.
3. **Citation validity ≠ truth:** the code confirms a citation points to a supplied passage, not that the answer's claim actually follows from it.
4. **No independent generalization benchmark:** our small data-derived examples are primarily regression checks.
5. **Quiz-driven LLM personalization:** the personality quiz remains deterministic; user scores stay out of the LLM prompt. The separate, consent-gated personalization path can send five derived scores; see PERSONALIZATION_GUIDE.md.

A good next experiment is to create 30–50 **new** questions not used to develop the evidence gate, annotate their reference answers with supporting passages, then compare generated answers using factual-support judgments performed blind to the model version.
