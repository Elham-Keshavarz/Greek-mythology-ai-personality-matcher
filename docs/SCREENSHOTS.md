# Capturing honest screenshots for the GitHub README

The ZIP does **not** invent or bundle screenshots of a browser it did not run. Capture them locally from your own Streamlit session:

1. Run `.venv\Scripts\python -m streamlit run app.py`.
2. On the landing/quiz view, capture the introductory explanation and first questionnaire page (use **synthetic / demo answers** if showing results).
3. Complete the quiz with demo answers and capture the top-three archetype matches plus the local five-trait table. Avoid displaying real personal scores or gender without consent.
4. In Mythology RAG Chat, ask `How are Apollo and Artemis related?` and capture an LLM answer with its `[S1]` citation; avoid exposing the terminal, `.env`, browser passwords or billing information.
5. Ask `What smartphone did Zeus use?` and capture the insufficient-evidence behavior.
6. Save cleaned screenshots as `docs/images/quiz.png`, `docs/images/personalized-match.png`, `docs/images/rag-answer.png`, `docs/images/abstention.png`.

After creating the real images you may add the following Markdown to README:

```markdown
![Personality quiz](images/quiz.png)
![Score-grounded match explanation](images/personalized-match.png)
![Mythology assistant with citation](images/rag-answer.png)
![Insufficient evidence behavior](images/abstention.png)
```

Images should not claim results, deployment uptime or performance that have not been tested. Keep GitHub-friendly image sizes and add meaningful alt text.
