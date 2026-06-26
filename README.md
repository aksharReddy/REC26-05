# Plain-Vanilla RAG Pipeline

This repo is a runnable Retrieval-Augmented Generation baseline for a PDF corpus. It ingests the supplied `RegsNavyI.pdf`, chunks the text, embeds the chunks with Gemini embeddings, retrieves the most relevant chunks for a question, and answers with Gemini using citations.

The intended path uses `GEMINI_API_KEY` for both embeddings and answer generation. A local TF-IDF fallback is still available with `--embedding-provider tfidf` if you need to run without a key. The default answer model is `gemini-2.5-flash`.

## Setup

```powershell
pip install -r requirements.txt
```

Gemini configuration:

```powershell
copy .env.example .env
# then set GEMINI_API_KEY in your shell or .env-aware environment
```

## Run

Ingest and index the PDF:

```powershell
python REC26-05.py ingest --pdf RegsNavyI.pdf --embedding-provider gemini
```

If Gemini rate-limits embeddings, use a slower ingest:

```powershell
python REC26-05.py ingest --pdf RegsNavyIV.pdf --embedding-provider gemini --index-dir rag_index_navy_iv --embedding-batch-size 16 --embedding-delay-seconds 3 --embedding-contents-per-minute 80
```

Ask a question:

```powershell
python REC26-05.py query "Can naval personnel accept gifts from representatives of foreign governments?" --show-context
```

Run a question file:

```powershell
python REC26-05.py evaluate --questions questions.txt --output evaluation_results.json
```

`questions.txt` can be one question per line. JSONL is also accepted when each row has a `question` or `query` field.

## Key Choices

- **Chunking:** 350-word chunks with 70-word overlap. This keeps passages focused while staying under Gemini free-tier embedding limits for the supplied PDF.
- **Embedding/index:** Gemini `gemini-embedding-001` by default. Local TF-IDF is kept as a no-key fallback.
- **Gemini rate limiting:** embedding ingest batches texts, sleeps between batches, and pauses after a configurable per-minute content budget.
- **Vector store:** persisted local files in `rag_index/`: `chunks.jsonl`, `gemini_embeddings.npy` or `tfidf.pkl`, and `metadata.json`.
- **Retrieval:** cosine similarity over embedding vectors, default `top_k=5`.
- **Prompting/generation:** Gemini is instructed to answer only from retrieved context and cite source ids/pages. Offline mode ranks retrieved sentences against the question and cites chunk ids/pages.
- **Unanswerable handling:** if the best retrieval score is below `0.08`, the system says the corpus does not provide enough support instead of guessing.

## Evaluation Notes

Metrics I would use:

- **Retrieval quality:** top-k recall/precision when gold supporting documents are known; otherwise manual relevance labels on retrieved chunks.
- **Answer quality:** exactness and completeness against expected answers.
- **Grounding/citation correctness:** every factual claim should be supported by cited chunks.
- **Unanswerable handling:** false-answer rate on questions deliberately absent from the corpus.

Implemented evaluation:

- `evaluate` writes retrieved chunks, scores, answers, and citations to JSON.
- It reports the share of questions considered answerable by the retrieval-confidence threshold and the average top retrieval score.

## Files

- `REC26-05.py` - CLI app with ingest, query, and evaluate commands.
- `RegsNavyI.pdf` - source corpus.
- `requirements.txt` - Python dependencies.
- `.env.example` - Gemini settings.
- `SUMMARY.md` - concise implementation summary.
- `sample_questions.txt` - small smoke-test question set.
- `rag_index/` - generated index after running ingest.
