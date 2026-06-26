# Implementation Summary

Built a complete, runnable RAG baseline around the provided `RegsNavyI.pdf`.

What was added:

- `REC26-05.py`: command-line RAG app with `ingest`, `query`, and `evaluate`.
- `requirements.txt`: required Python packages.
- `.env.example`: Gemini configuration template.
- `README.md`: setup, run commands, design choices, and evaluation notes.

Pipeline:

1. Extract text from the PDF using `pypdf`.
2. Split pages into 350-word chunks with 70-word overlap.
3. Embed chunks with Gemini `gemini-embedding-001`, or locally with TF-IDF fallback.
4. Persist the index to `rag_index/`.
5. Retrieve top-k chunks by cosine similarity.
6. Answer with citations using Gemini `gemini-2.5-flash` generation or an offline extractive fallback.
7. Treat low-confidence retrieval as unanswerable instead of hallucinating.

Verification commands used:

```powershell
python REC26-05.py ingest --pdf RegsNavyI.pdf
python REC26-05.py ingest --pdf RegsNavyI.pdf --embedding-provider tfidf --index-dir rag_index_tfidf_test
python REC26-05.py query "Can naval personnel accept gifts from representatives of foreign governments?" --index-dir rag_index_tfidf_test --show-context
python -m py_compile REC26-05.py
```
