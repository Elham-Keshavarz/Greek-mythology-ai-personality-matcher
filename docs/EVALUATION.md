# Evaluation protocol and observed results

## What was measured?

Two deliberately small, **corpus-derived** retrieval QA suites are stored at:

- `data/evaluation/retrieval_regression.json`: regression set, 13 supported + 6 unsupported questions.
- `data/evaluation/retrieval_challenge.json`: challenge set, 12 supported + 10 unsupported questions.

The positive queries specify the expected document IDs. Negative queries are questions without appropriate support in the **project's own curated corpus**. These were authored with knowledge of the dataset and are **not independent external test data**. No reliable general-world accuracy claim can be made from these checks.

## Observed Windows benchmark run

Reported by a local Windows user run with `python -m src.rag.evaluate --suite both --compare` on 2026-11-10, using local FAISS and `sentence-transformers/all-MiniLM-L6-v2`:

| Metric | Regression | Challenge |
|---|---:|---:|
| Positive items | 13 | 12 |
| Negative items | 6 | 10 |
| `hit@5` hybrid | 1.0000 | 1.0000 |
| `mrr@5` hybrid | 0.9231 | 0.9583 |
| `mrr@5` raw FAISS (historical ranking baseline) | 0.7269 | 0.8611 |
| Gate positive acceptance | 1.0000 | 1.0000 |
| Gate negative rejection | 1.0000 | 1.0000 |
| False rejects | 0 | 0 |
| False accepts | 0 | 0 |

**41/41** is the count of correctly accepted/rejected prompts **on these two provided suites**; it is not the accuracy of the hosted LLM. The Athena symbols challenge question passes. The benchmark file set and scored indices were not generated from independent held-out mythological questions.

A JSON machine-readable transcription of these **observed local benchmark numbers** is available at [`reports/retrieval_observed.json`](../reports/retrieval_observed.json). These figures were observed in a local run and were not re-measured as part of this packaging cleanup; the retrieval logic remains unchanged.

## Metrics explained

- **Hit@5:** expected document ID appears in the first five results.
- **MRR@5:** mean reciprocal rank of the first correct document (zero if absent from top five).
- **Positive gate acceptance:** fraction of questions considered supportable that the gate accepts.
- **Negative gate rejection:** fraction of questions considered unsupported that the gate rejects.
- **Balanced gate accuracy:** average of positive acceptance and negative rejection; do not confuse with general or clinical accuracy.

`--compare` also tests a **legacy union-of-passages gate on current reranked candidates**, not a separate historical implementation. The raw FAISS comparison omits the gate and is ranking-only. These distinctions matter when presenting relative gains.

## Reproduce on your own machine

```powershell
.venv\Scripts\python -m src.rag.build_index
.venv\Scripts\python -m src.rag.evaluate --suite both --compare --output reports/local_retrieval.json
```

The command downloads an embedding model if it is missing. It makes **no paid OpenAI request**. Do not claim fresh results without rerunning the benchmark.

## Faithfulness != citation syntax

A generated answer containing `[S1]` passes a **source-ID** check; this does not establish that all its claims are entailed by `[S1]`. A real assessment needs manually annotated answers (or independently validated reference labels), including adversarial and multi-hop questions, across substantially more independent examples.

The project provides an **opt-in** seven-question live output audit and a **manual** review CSV workflow. No factual faithfulness percentage is claimed for unreviewed outputs. See [human faithfulness review](FAITHFULNESS_REVIEW.md).

## Recommended future work

- Create a larger independent evaluation set with expert labels and inter-annotator agreement.
- Evaluate source-level precision/recall and passage entailed claims, not just supporting-document IDs.
- Measure runtime/cost distributions using a specified machine and model; do not extrapolate from one laptop.
- Improve multi-hop and compositional question handling without sacrificing abstention precision.
