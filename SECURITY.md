# Security notes

This is an educational demo and not a production authentication or personal health system.

## Reporting a security issue

Do **not** post API keys, proof-of-concept exploits with real user data, or other secrets in a public issue. If this is your repository, enable GitHub's private vulnerability reporting feature or provide a safe private maintainer contact channel; do not invent a security contact address in the README.

## API key handling

Store `OPENAI_API_KEY` only in a project-root `.env` file excluded by `.gitignore` or in a secure deployment secret manager. Do not place keys in Streamlit UI labels, screenshots, notebook outputs, commits or logs. If any key is leaked, revoke and replace it with your API provider immediately and audit Git history.

## Trust boundaries

- Retrieval and scoring run locally unless explicitly configured otherwise.
- Project-authored and retrieved evidence text is treated as untrusted data in the LLM prompt. This is not guaranteed protection against prompt injection.
- The Evidence Gate is heuristic, and citation-ID validation is **not** semantic fact verification.
- FAISS index binary files should come only from a trusted local build. Do not deserialize arbitrary FAISS indexes downloaded from unknown sources.
- The optional OpenAI integration transmits question text and one supporting excerpt; personalized commentary also transmits five derived scores after consent. See [`docs/PRIVACY.md`](docs/PRIVACY.md).
