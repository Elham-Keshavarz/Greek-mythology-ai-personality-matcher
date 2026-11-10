# GitHub Release Checklist

This document provides a repeatable checklist for preparing, validating, and publishing updates to the **Greek Mythology AI Personality Matcher** repository.

The repository is already hosted on GitHub. These steps apply to future changes and releases.

## 1. Code Quality and Automated Tests

Before publishing changes:

- [ ] Confirm that the application runs successfully.
- [ ] Run all automated tests.
- [ ] Verify that all required tests pass.
- [ ] Review failed or skipped tests.
- [ ] Ensure new features include appropriate tests.

```powershell
python -m pytest -q
```

## 2. RAG and Retrieval Validation

- [ ] Verify that the FAISS index can be built.
- [ ] Test retrieval with supported mythology questions.
- [ ] Test evidence rejection with unsupported questions.
- [ ] Run the retrieval evaluation benchmark.
- [ ] Review changes in retrieval and evidence-gate metrics.

```powershell
python -m src.rag.build_index
python -m src.rag.evaluate --suite both --compare
```

Benchmark results must include their evaluation methodology and limitations. Results from small, hand-authored datasets must not be presented as general model accuracy.

## 3. OpenAI Integration

- [ ] Confirm that API credentials are loaded from environment variables.
- [ ] Run the live API smoke test when appropriate.
- [ ] Verify that generated answers include valid source citations.
- [ ] Confirm that unsupported questions do not trigger unnecessary LLM calls.
- [ ] Review representative generated answers for factual grounding.

```powershell
python -m tools.live_api_smoke
```

Live API tests require a valid OpenAI API key and may incur usage costs. They are not required for every local change.

## 4. Security and Privacy

- [ ] Confirm that `.env` and virtual environments are excluded from Git.
- [ ] Ensure `.env.example` contains placeholders only.
- [ ] Check notebooks for saved credentials and sensitive outputs.
- [ ] Run the release security scanner.
- [ ] Review staged files before committing.
- [ ] Confirm that no personal information or API credentials are exposed.

```powershell
python -m tools.check_release
git status --short
git --no-pager diff --cached --check
git --no-pager diff --cached --name-only
```

Automated secret scanning is a safeguard, not a guarantee