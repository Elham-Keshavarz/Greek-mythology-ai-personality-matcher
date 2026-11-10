# Human review of LLM-generated mythology answers

## Why this matters

A citation `[S1]` only identifies the passage the model was **allowed** to use. It does **not** prove that all factual statements in the answer follow from the supplied text. A responsible project should not equate syntactic citation checks, high retrieval scores or a passed Evidence Gate with zero hallucinations.

The project exposes an **optional**, explicitly paid end-to-end generation audit and a fully **offline** CSV review utility. It will not report a numeric faithfulness score until at least one generated answer has been manually assigned a binary review outcome.

## Step 1 — Run the live audit (explicit opt-in)

This uses the seven sample questions in `data/evaluation/llm_qa.json`. Ensure `OPENAI_API_KEY` and `OPENAI_MODEL` are configured in your private project-root `.env`.

```powershell
.venv\Scripts\python -m src.rag.evaluate_generation --live --limit 7 --output reports/local_llm_audit.json
```

Without `--live`, the command does **not** call the API. Positive questions may generate a reply; negative ones should abstain before API use. The generated audit contains the response, any allowed source IDs, an approved passage when available, and review hints.

## Step 2 — Export a review worksheet

```powershell
.venv\Scripts\python -m src.rag.faithfulness_review --input reports/local_llm_audit.json --csv reports/local_faithfulness_review.csv
```

Open the CSV in Excel or a text editor. Each row includes the source text used for generation, source URL, generated answer, citations and the intended answer/abstention hint.

For **each** row where `mode` is `llm`, inspect **every factual claim** relative to the approved passage. Replace the `review_label` with:

| Label | Meaning |
|---|---|
| `supported` | All factual claims are backed by the approved excerpt; no invented relations/attributes |
| `unsupported` | At least one claim is absent from or contradicts the excerpt |
| `uncertain` | The reviewer cannot decide with current evidence |
| `not_reviewed` | Not yet reviewed — default for generated answers |
| `not_applicable` | No generated answer, e.g. abstention |

Notes should describe which clause is unsupported, not just a pass/fail judgment. Review the actual **approved passage**, not only external background links.

## Step 3 — Summarize the manually graded worksheet

```powershell
.venv\Scripts\python -m src.rag.faithfulness_review --summary reports/local_faithfulness_review.csv
```

The rate `supported / (supported + unsupported)` uses **only manually reviewed binary judgments**. Unreviewed and uncertain outputs are counted separately; abstentions are excluded rather than automatically scored as faithful. When there are no binary-reviewed answers, the rate is **null**, not a fabricated 100%.

## Limitations

- Seven corpus-derived examples are only a smoke audit, not a scientific faithfulness benchmark.
- The background URLs are not evidence that project-authored summaries are exact historical quotations.
- The review template does not use a second LLM as an oracle; human judgments can still differ.
- For a publication-quality claim, use independent labels, a much larger query set, a clearly defined unit of claims, and multiple reviewers to measure agreement.
- These CSV/JSON files may contain personal questions or AI provider content and are deliberately **Git-ignored**. Do not upload them to a public repository.
