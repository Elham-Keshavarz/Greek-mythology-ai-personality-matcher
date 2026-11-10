# Contributing

Thanks for helping improve this educational, English-language mythology RAG demo.

1. Work in a separate branch; use Python 3.11 or 3.12.
2. Install `requirements.txt` and run `python -m pytest -q` before proposing changes.
3. Keep scoring and the original IPIP question ordering stable unless explicitly documenting a scoring-method change.
4. When adding a deity or factual note, preserve the corpus attribution fields; write original summaries instead of copying source pages.
5. Add small deterministic tests for retrieval/reranking/abstention behavior. Avoid tests requiring paid APIs or live websites in CI.
6. Report new metrics only from actual reproducible runs, with dataset size, method, model and limitations.
7. Do not commit real questionnaire responses, audit transcripts, API keys, `.env`, model checkpoints or FAISS index binaries.

For privacy and trust boundaries see [PRIVACY.md](docs/PRIVACY.md) and [SECURITY.md](SECURITY.md).
