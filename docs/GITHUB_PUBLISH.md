# GitHub publication checklist (Windows PowerShell)

This project is **prepared for GitHub** but has not been pushed to your account. You remain in control of the repository name, visibility, screenshots and publication.

## Before committing

- [ ] Run the full local test suite (`python -m pytest -q`) and confirm that all tests actually executed pass; review any skipped tests.
- [ ] Verify local FAISS indexing and real RAG responses, including an unsupported question.
- [ ] Test the 50-question Streamlit navigation and personalized local result in a browser.
- [ ] If using hosted LLM, ensure it produces a cited answer and requires consent for sending derived scores.
- [ ] Confirm `README.md` reflects actual behavior and that the small benchmark limitations are conspicuous.
- [ ] Consider capturing **your own screenshots** as described in [SCREENSHOTS.md](SCREENSHOTS.md); no fake previews are bundled.
- [ ] Run the release safety checks below. Never upload `.env` or personal audit CSV/JSON files.

## Run offline checks

From the root containing `app.py`:

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m tools.check_release
```

## Initialize your repository

Create an **empty** repository at GitHub (without a generated README or license, because this project already includes both), then in the project folder:

```powershell
git init
git branch -M main
git add .
git status --short
git diff --cached --name-only
```

**Stop if you see `.env`, `.venv/`, `data/rag_index/`, personally identifying screenshots, private transcripts or local audit reports staged.** Remove them from the staging area before proceeding. `.env.example` is safe: it contains blank credentials.

```powershell
git commit -m "Prepare Greek mythology personality matcher for publication"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your real GitHub choices. If the remote already exists, use `git remote -v` and update it appropriately rather than blindly adding a duplicate. No GitHub token needs to be put in the repository or scripts.

## After pushing

- In **Actions**, check that the Python 3.11 and 3.12 CI matrix runs and passes. These are **offline tests**, not a live OpenAI or full browser test.
- Confirm that the README renders its Mermaid diagram and all relative links.
- Give the repository an informative description and a few accurate topics such as `rag`, `faiss`, `sentence-transformers`, `streamlit`, `openai`, `big-five`.
- Attach only **real** UI screenshots after checking that they contain no API keys, raw questionnaire answers or other private information.

If you already have a local copy, update the source files while preserving `.env` and `.venv`, run tests, and review `git diff` before committing. No new environment is necessary unless dependencies changed.
