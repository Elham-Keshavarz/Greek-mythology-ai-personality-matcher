# Privacy, optional AI requests and secrets

This is a **local demonstration app**, not a hosted medical/psychological service. It does not require user accounts or a central database. This document describes what the code is designed to do; it is not a promise about third-party platform logs or retention.

## What stays local by default

- Your raw answers to the 50 items are used to compute five traits in **Streamlit session state**, not persisted to the app's own database.
- The gender choice is used only as an optional mythology candidate filter and is not sent as a model feature.
- Five computed traits, matching ranks and score-gap narratives are calculated locally in Python.
- FAISS retrieval and sentence embeddings execute locally after the initial model download.
- The local evaluation utilities and tests do not call OpenAI by default.

The initial model download contacts Hugging Face and may have its own service logs. Session data is transient and is not a substitute for encrypted personal data storage.

## Optional OpenAI calls

| Feature | Trigger | Sent to configured API |
|---|---|---|
| Mythology chat LLM | User selects **LLM answer** and submits a question that passes the Evidence Gate | Question, one approved project-authored mythology excerpt, generation instructions |
| Personalized commentary | User ticks consent and clicks **Generate explanation with OpenAI** | Five derived trait scores, author-designed selected archetype/rationale, typed question, requested focus, one approved mythology excerpt |

The personalized request **does not include raw item-by-item answers or gender**. Those five derived scores are still personal profile data and may be sensitive; the user must opt in. The remote provider's policies, retention practices and billing are outside this local app's control.

Model prompts may include text typed by the user; do **not** put passwords, financial details or other sensitive content into prompts.

## Logs and artifacts

- `data/rag_index/` is a locally recreated FAISS index excluded from Git.
- `reports/local_*` and `data/evaluation/local_results_*.json` are ignored because audit outputs can contain user questions and provider responses.
- `.env`, `.env.*` except `.env.example`, and `.streamlit/secrets.toml` are ignored by Git. `OPENAI_API_KEY` must not appear in screenshots, terminal transcripts or notebooks.
- The repository's `tools.check_release` performs a **basic** credential-pattern scan; it is not a comprehensive secret-detection system. Always inspect staged changes before pushing.

## Publication checklist

1. Keep `.env` next to `app.py` locally, but not in Git.
2. Run `python -m tools.check_release` and `git status --short`.
3. Check `git diff --cached --name-only` and make sure personal logs/reports are absent.
4. Rotate any secret immediately if it was committed or exposed. Removing it from the latest commit alone does **not** remove it from Git history.
5. Do not publish genuine participant questionnaire responses, private API prompts or filled-in evaluation review worksheets.

For algorithm caveats see [architecture](ARCHITECTURE.md), [evaluation](EVALUATION.md) and [security](../SECURITY.md).
