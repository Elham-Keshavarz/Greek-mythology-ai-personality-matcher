# RAG — Learning Guide

This is the walkthrough of the **real implementation**, not a generic RAG tutorial. Read the files in this order:

## 1 — `src/rag/knowledge_base.py`

- `Document`: dataclass containing stable `id`, `title`, `text`, `source_url`, `source_type`, and `source_note`.
- `load_documents()`: turns the **45 existing deity summaries** in `data/archetype_profiles.json` plus **18 newly written factual summaries** in `data/knowledge/curated_facts.json` into 63 searchable documents.
- No web scraping takes place at runtime. Profile-source links are general background reading; the 18 summary sources are article links for verification, *not* licensed/full-text snapshots.
- The **personality scores do not enter the RAG corpus**. They are illustrative interpretation, not mythology evidence.

Try: `python -c "from src.rag.knowledge_base import load_documents; print(len(load_documents()))"`

## 2 — `src/rag/text_chunker.py`

- `Chunk`: represents an indexable passage, with its original source metadata.
- `chunk_documents(documents, max_words=120, overlap_words=25)`: word-window chunking for longer texts. Most current entries are shorter than 120 words, so they stay as one chunk.
- Overlap helps preserve context if a longer source is split across chunk boundaries. Word-count chunking is intentionally simple; later we could test semantic or token-aware chunking.

Try: change the max length to 20 and observe the extra passages. **Rebuild the index** after changing chunk settings.

## 3 — `src/rag/embeddings.py`

- `SentenceEmbedder.__init__`: loads `sentence-transformers/all-MiniLM-L6-v2`. The first use downloads model files; subsequent runs reuse the model cache.
- `SentenceEmbedder.encode(texts)`: turns a list of texts into floating-point vectors and applies normalization. This is **inference using a pretrained model**, not fine-tuning.

Think: two differently-worded passages may be nearby in embedding space even without exact word overlap.

## 4 — `src/rag/vector_store.py`

- `fingerprint(chunks, model_name)`: SHA-256 identifier for the dataset, source metadata and embedding model. It lets the cache rebuild when sources or model names change.
- `FaissStore.build`: creates vectors for all chunks, L2-normalizes them and indexes with **`faiss.IndexFlatIP`**. Because vectors have unit length, inner product corresponds to cosine similarity.
- `FaissStore.load_or_build`: reuses the trusted locally generated `data/rag_index/faiss.index` if its metadata matches, or builds a new one.
- `FaissStore.search(question, top_k, min_score)`: vectorizes the question and retrieves the top passages with scores. `min_score` is a heuristic, **not** a proof of factual relevance. The index gives **exact nearest-neighbor search** on this small corpus, not approximate search.

Try: print the top-five hits for the same question while varying `top_k` and `min_score`.

## 5 — `src/rag/answer_service.py`

- `answer_question`: retrieval orchestration. If no passages pass the threshold, returns an abstention. If no LLM generator was provided, returns explicitly labelled **retrieved evidence**, not a generated narrative. Otherwise it calls the configured generator.
- `build_prompt`: constructs a system instruction and user message containing the question and source snippets labelled `[S1]`, `[S2]` etc. It instructs the model to use only the supplied evidence and to say when evidence is insufficient.
- This is **prompt guidance**, not a mathematical guarantee of zero hallucinations. The app shows retrieved sources regardless of whether the LLM cites them correctly.
- The returned similarity score is retrieval cosine similarity, **not a model probability**.

Try: ask an unrelated question such as “What was the price of bread in Athens in 400 BCE?” and examine whether the evidence genuinely supports an answer. We still need a real no-answer benchmark.

## 6 — `src/rag/llm_client.py`

- `is_configured`: checks whether `OPENAI_API_KEY` and `OPENAI_MODEL` exist in environment variables.
- `generate_answer`: uses the OpenAI Python SDK with the optional `OPENAI_BASE_URL` to call an OpenAI-compatible Chat Completions API.
- It sends the **chat question and retrieved snippets**, **not** the user's Big Five answers, to the configured provider. The API call may cost money.
- No credentials are bundled. Never commit `.env`.

## 7 — `src/rag/pipeline.py`, `build_index.py`, `ask.py`, `evaluate.py`

- `build_pipeline`: composes documents → chunks → embedder → FAISS store.
- `build_index.py`: build the index from the command line.
- `ask.py`: try retrieval in a terminal without opening Streamlit.
- `evaluate.py`: tests small **hand-authored** retrieval suites, calculating `Hit@5` and `MRR@5`. This is a learning / regression benchmark, *not* a reliable estimate of production accuracy.

## 8 — `app.py`

- Existing quiz and preference functions are unchanged.
- `feature`: sidebar selector chooses `Personality Quiz` or `Mythology RAG Chat`.
- `get_rag_store`: `@st.cache_resource` loads one model and index per Streamlit process.
- `mythology_chat_view`: displays questions, evidence / LLM response, and source links.
- `st.session_state["rag_messages"]`: remembers chat messages across Streamlit reruns. It is **separate from `quiz_answers`**.
- `Rebuild knowledge index`: clears the Streamlit model/index cache. If you edit input JSONs, rerun or click rebuild.

## What this milestone is (and is not)

**Implemented:** 63 short English knowledge documents, overlap chunking, local pretrained text embeddings, FAISS vector search, optional OpenAI-compatible grounded-answer generation, source metadata, basic retrieval evaluation, testable Python modules, Streamlit chat.

**Not implemented:** complete sourced narratives for all 45 figures, production-grade citation faithfulness checks, historical disambiguation across ancient authors, a robust no-answer classifier, web crawling, a production database, multi-turn context used by the LLM, or LLM personality-report generation.

**Publishing reminder:** the project-authored personality archetypes are fictional and not scientifically validated. RAG factual summaries do not validate them.
