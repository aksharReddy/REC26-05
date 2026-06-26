#!/usr/bin/env python
"""
Plain-vanilla RAG pipeline for a PDF corpus.

Commands:
  python REC26-05.py ingest --pdf RegsNavyI.pdf
  python REC26-05.py query "What is the policy about ...?"
  python REC26-05.py evaluate --questions questions.txt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_INDEX_DIR = Path("rag_index")
DEFAULT_PDF = Path("RegsNavyI.pdf")
CHUNK_WORDS = 350
CHUNK_OVERLAP = 70
DEFAULT_TOP_K = 5
MIN_CONFIDENCE = 0.08
DEFAULT_EMBEDDING_PROVIDER = "gemini"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_EMBEDDING_BATCH_SIZE = 16
DEFAULT_GEMINI_EMBEDDING_DELAY_SECONDS = 3.0
DEFAULT_GEMINI_EMBEDDING_CONTENTS_PER_MINUTE = 80


@dataclass
class Chunk:
    id: str
    source: str
    page_start: int
    page_end: int
    text: str


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    if not pages:
        raise ValueError("No extractable text found in the PDF.")
    return pages


def word_windows(words: list[str], size: int, overlap: int) -> Iterable[tuple[int, int]]:
    if size <= overlap:
        raise ValueError("chunk size must be greater than overlap")

    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield start, end
        if end == len(words):
            break
        start = end - overlap


def chunk_pages(pages: list[tuple[int, str]], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in pages:
        words = text.split()
        for chunk_number, (start, end) in enumerate(
            word_windows(words, CHUNK_WORDS, CHUNK_OVERLAP),
            start=1,
        ):
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    id=f"{Path(source).stem}-p{page_number}-c{chunk_number}",
                    source=source,
                    page_start=page_number,
                    page_end=page_number,
                    text=chunk_text,
                )
            )
    return chunks


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY, or use --embedding-provider tfidf."
        )
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai") from exc
    return genai.Client(api_key=api_key)


def format_gemini_embedding_input(text: str, model: str, task_type: str) -> str:
    if model != "gemini-embedding-2":
        return text
    if task_type == "RETRIEVAL_QUERY":
        return f"task: question answering | query: {text}"
    return f"title: none | text: {text}"


def gemini_embed_texts(
    texts: list[str],
    model: str,
    task_type: str,
    batch_size: int = DEFAULT_GEMINI_EMBEDDING_BATCH_SIZE,
    delay_seconds: float = DEFAULT_GEMINI_EMBEDDING_DELAY_SECONDS,
    contents_per_minute: int = DEFAULT_GEMINI_EMBEDDING_CONTENTS_PER_MINUTE,
) -> np.ndarray:
    from google.genai import types

    client = get_gemini_client()
    vectors: list[list[float]] = []
    if batch_size < 1:
        raise ValueError("Gemini embedding batch size must be at least 1.")
    if contents_per_minute < 1:
        raise ValueError("Gemini embedding contents-per-minute limit must be at least 1.")
    if model == "gemini-embedding-2":
        batch_size = 1
    window_started = time.monotonic()
    contents_in_window = 0
    for start in range(0, len(texts), batch_size):
        batch = [
            format_gemini_embedding_input(text, model, task_type)
            for text in texts[start : start + batch_size]
        ]
        batch_number = start // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        if contents_in_window + len(batch) > contents_per_minute:
            elapsed = time.monotonic() - window_started
            if elapsed < 65:
                wait_for = int(65 - elapsed)
                print(f"Gemini embedding rate limit pause: waiting {wait_for}s...", file=sys.stderr)
                time.sleep(wait_for)
            window_started = time.monotonic()
            contents_in_window = 0

        config = None
        if model != "gemini-embedding-2":
            config = types.EmbedContentConfig(task_type=task_type)
        for attempt in range(3):
            try:
                print(
                    f"Embedding batch {batch_number}/{total_batches} "
                    f"({len(batch)} texts)...",
                    file=sys.stderr,
                )
                response = client.models.embed_content(model=model, contents=batch, config=config)
                break
            except Exception as exc:
                retry_after = parse_retry_delay_seconds(str(exc)) or (30 + attempt * 20)
                if attempt == 2:
                    raise
                print(
                    f"Gemini embedding rate-limited/unavailable; retrying in {retry_after}s...",
                    file=sys.stderr,
                )
                time.sleep(retry_after)
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            embedding = getattr(response, "embedding", None)
            embeddings = [embedding] if embedding is not None else []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if values is None and isinstance(embedding, dict):
                values = embedding.get("values")
            if not values:
                raise RuntimeError("Gemini returned an empty embedding.")
            vectors.append(list(values))
        contents_in_window += len(batch)
        if delay_seconds > 0 and start + batch_size < len(texts):
            time.sleep(delay_seconds)
    return np.array(vectors, dtype=np.float32)


def parse_retry_delay_seconds(message: str) -> int | None:
    match = re.search(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s", message)
    if not match:
        match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", message)
    if not match:
        return None
    return max(1, int(float(match.group(1))) + 2)


def save_index(
    index_dir: Path,
    chunks: list[Chunk],
    provider: str,
    vectorizer: TfidfVectorizer | None,
    matrix,
    embedding_model: str | None,
    embedding_options: dict | None = None,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    if provider == "gemini":
        np.save(index_dir / "gemini_embeddings.npy", matrix)
    else:
        with (index_dir / "tfidf.pkl").open("wb") as fh:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, fh)
    metadata = {
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "embedding_provider": provider,
        "embedding_model": embedding_model or "sklearn TfidfVectorizer word ngrams",
    }
    if embedding_options:
        metadata["embedding_options"] = embedding_options
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_index(index_dir: Path) -> tuple[list[Chunk], dict, TfidfVectorizer | None, object]:
    chunks_path = index_dir / "chunks.jsonl"
    metadata_path = index_dir / "metadata.json"
    if not chunks_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run: python REC26-05.py ingest --pdf {DEFAULT_PDF}"
        )

    chunks = [
        Chunk(**json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provider = metadata.get("embedding_provider", "tfidf")
    if provider == "gemini":
        vectors_path = index_dir / "gemini_embeddings.npy"
        if not vectors_path.exists():
            raise FileNotFoundError(f"Missing Gemini embedding matrix: {vectors_path}")
        return chunks, metadata, None, np.load(vectors_path)

    vectors_path = index_dir / "tfidf.pkl"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing TF-IDF index: {vectors_path}")
    with vectors_path.open("rb") as fh:
        payload = pickle.load(fh)
    return chunks, metadata, payload["vectorizer"], payload["matrix"]


def retrieve(question: str, index_dir: Path, top_k: int) -> list[tuple[Chunk, float]]:
    chunks, metadata, vectorizer, matrix = load_index(index_dir)
    provider = metadata.get("embedding_provider", "tfidf")
    if provider == "gemini":
        model = metadata.get("embedding_model") or DEFAULT_GEMINI_EMBEDDING_MODEL
        query_vector = gemini_embed_texts([question], model, "RETRIEVAL_QUERY")
    else:
        if vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer missing from index.")
        query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def extractive_answer(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    if not retrieved or retrieved[0][1] < MIN_CONFIDENCE:
        return (
            "I could not find enough support in the corpus to answer this confidently. "
            "The best retrieved chunks had low similarity, so this should be treated as unanswerable."
        )

    supported = [(chunk, score) for chunk, score in retrieved if score >= MIN_CONFIDENCE]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    candidates: list[tuple[str, Chunk]] = []
    for chunk, _score in supported:
        for sentence in split_sentences(chunk.text):
            candidates.append((sentence, chunk))

    if not candidates:
        chunk, _score = retrieved[0]
        return f"{chunk.text[:700].strip()} [{chunk.id}]"

    corpus = [question] + [sentence for sentence, _chunk in candidates]
    vectors = vectorizer.fit_transform(corpus)
    sentence_scores = cosine_similarity(vectors[0], vectors[1:]).ravel()
    best_indexes = np.argsort(sentence_scores)[::-1][:3]

    answer_parts: list[str] = []
    used_chunk_ids: set[str] = set()
    for idx in best_indexes:
        if sentence_scores[idx] <= 0:
            continue
        sentence, chunk = candidates[idx]
        citation = f"[{chunk.id}, p. {chunk.page_start}]"
        answer_parts.append(f"{sentence} {citation}")
        used_chunk_ids.add(chunk.id)

    if not answer_parts:
        chunk, _score = retrieved[0]
        return f"{chunk.text[:700].strip()} [{chunk.id}, p. {chunk.page_start}]"
    return " ".join(answer_parts)


def gemini_answer(question: str, retrieved: list[tuple[Chunk, float]], model: str) -> str | None:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return None

    try:
        from google.genai import types
    except Exception:
        return None

    context = "\n\n".join(
        f"Source: {chunk.id}, page {chunk.page_start}, score {score:.3f}\n{chunk.text}"
        for chunk, score in retrieved
    )
    prompt = f"""
Answer the question using only the provided context.
If the context does not contain the answer, say that the corpus does not answer it.
Cite sources inline using the provided Source ids and page numbers.

Question: {question}

Context:
{context}
""".strip()

    client = get_gemini_client()
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            return (getattr(response, "text", None) or "").strip()
        except Exception as exc:
            if attempt == 2:
                print(f"Gemini generation unavailable; using extractive fallback. {exc}", file=sys.stderr)
                return None
            time.sleep(2 + attempt * 3)


def answer_question(question: str, index_dir: Path, top_k: int, model: str | None) -> dict:
    retrieved = retrieve(question, index_dir, top_k)
    generated = None
    if model and retrieved and retrieved[0][1] >= MIN_CONFIDENCE:
        generated = gemini_answer(question, retrieved, model)
        if generated and generated.startswith("Gemini generation failed"):
            generated = f"{generated}\n\n{extractive_answer(question, retrieved)}"
    answer = generated or extractive_answer(question, retrieved)
    return {
        "question": question,
        "answer": answer,
        "retrieved": [
            {
                "rank": rank,
                "score": round(score, 4),
                "id": chunk.id,
                "source": chunk.source,
                "page": chunk.page_start,
                "preview": chunk.text[:260],
            }
            for rank, (chunk, score) in enumerate(retrieved, start=1)
        ],
    }


def cmd_ingest(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, pdf_path.name)
    provider = args.embedding_provider.lower()

    if provider == "gemini":
        vectorizer = None
        matrix = gemini_embed_texts(
            [chunk.text for chunk in chunks],
            args.embedding_model,
            "RETRIEVAL_DOCUMENT",
            batch_size=args.embedding_batch_size,
            delay_seconds=args.embedding_delay_seconds,
            contents_per_minute=args.embedding_contents_per_minute,
        )
        embedding_model = args.embedding_model
        embedding_options = {
            "batch_size": args.embedding_batch_size,
            "delay_seconds": args.embedding_delay_seconds,
            "contents_per_minute": args.embedding_contents_per_minute,
        }
    elif provider == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.92,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        embedding_model = None
        embedding_options = None
    else:
        raise ValueError("--embedding-provider must be gemini or tfidf")

    save_index(
        Path(args.index_dir),
        chunks,
        provider,
        vectorizer,
        matrix,
        embedding_model,
        embedding_options,
    )

    print(f"Ingested {pdf_path} into {len(chunks)} chunks from {len(pages)} pages.")
    print(f"Embedding provider: {provider}")
    print(f"Index written to {args.index_dir}")


def print_answer(result: dict, show_context: bool) -> None:
    print("\nQuestion")
    print(textwrap.fill(result["question"], width=100))
    print("\nAnswer")
    print(textwrap.fill(result["answer"], width=100))
    print("\nSources")
    for item in result["retrieved"]:
        print(
            f"{item['rank']}. {item['id']} | {item['source']} p. {item['page']} | "
            f"score={item['score']}"
        )
        if show_context:
            print(textwrap.indent(textwrap.fill(item["preview"], width=96), "   "))


def cmd_query(args: argparse.Namespace) -> None:
    result = answer_question(args.question, Path(args.index_dir), args.top_k, args.model)
    print_answer(result, args.show_context)


def read_questions(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        questions = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(payload.get("question") or payload.get("query") or line)
        return questions
    return [
        line.strip().lstrip("-0123456789. )")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cmd_evaluate(args: argparse.Namespace) -> None:
    questions = read_questions(Path(args.questions))
    results = [
        answer_question(question, Path(args.index_dir), args.top_k, args.model)
        for question in questions
    ]
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    answered = sum(
        1
        for result in results
        if result["retrieved"] and result["retrieved"][0]["score"] >= MIN_CONFIDENCE
    )
    avg_top_score = (
        sum(result["retrieved"][0]["score"] for result in results if result["retrieved"])
        / max(1, len(results))
    )
    print(f"Evaluated {len(results)} questions.")
    print(f"Answerable-by-threshold: {answered}/{len(results)}")
    print(f"Average top retrieval score: {avg_top_score:.4f}")
    print(f"Detailed results written to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plain-vanilla PDF RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="extract, chunk, embed, and index a PDF")
    ingest.add_argument("--pdf", default=str(DEFAULT_PDF), help="PDF corpus path")
    ingest.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="index output directory")
    ingest.add_argument(
        "--embedding-provider",
        choices=["gemini", "tfidf"],
        default=os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        help="embedding backend; gemini needs GEMINI_API_KEY",
    )
    ingest.add_argument(
        "--embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL),
        help="Gemini embedding model used when --embedding-provider gemini",
    )
    ingest.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", DEFAULT_GEMINI_EMBEDDING_BATCH_SIZE)),
        help="Gemini embedding texts per batch",
    )
    ingest.add_argument(
        "--embedding-delay-seconds",
        type=float,
        default=float(
            os.getenv("GEMINI_EMBEDDING_DELAY_SECONDS", DEFAULT_GEMINI_EMBEDDING_DELAY_SECONDS)
        ),
        help="seconds to sleep between Gemini embedding batches",
    )
    ingest.add_argument(
        "--embedding-contents-per-minute",
        type=int,
        default=int(
            os.getenv(
                "GEMINI_EMBEDDING_CONTENTS_PER_MINUTE",
                DEFAULT_GEMINI_EMBEDDING_CONTENTS_PER_MINUTE,
            )
        ),
        help="max Gemini embedding texts to send per minute before pausing",
    )
    ingest.set_defaults(func=cmd_ingest)

    query = subparsers.add_parser("query", help="ask one question against the index")
    query.add_argument("question", help="natural-language question")
    query.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="index directory")
    query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="retrieved chunks to use")
    query.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help="Gemini model for abstractive answers; unset with --model '' for extractive mode",
    )
    query.add_argument("--show-context", action="store_true", help="print retrieved previews")
    query.set_defaults(func=cmd_query)

    evaluate = subparsers.add_parser("evaluate", help="run a question file and write JSON results")
    evaluate.add_argument("--questions", required=True, help="txt or jsonl question file")
    evaluate.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="index directory")
    evaluate.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="retrieved chunks to use")
    evaluate.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help="Gemini model for answers",
    )
    evaluate.add_argument("--output", default="evaluation_results.json", help="result JSON path")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
