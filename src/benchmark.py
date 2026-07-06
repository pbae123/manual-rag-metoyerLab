"""
benchmark.py — Compare Full-book RAG vs Chapter-specific RAG

Metrics:
  Runtime:
    - Chunk count (index size proxy)
    - Local chunking time
    - Build time (chunking + embedding + Pinecone upload)
    - Per-query latency (avg, min, max over N trials)

  Accuracy (LLM-as-judge):
    - Each answer scored 1–10 by GPT-4o-mini on relevance + completeness
    - Chapter citation hit rate: does the answer cite the expected chapter?

  Retrieval accuracy (OpenAI vs Gemini, full-book):
    - Avg cosine similarity between each query and its retrieved chunks,
      computed separately per system's own embedding space, then compared

Usage:
    cd /Users/peterbae/Documents/manual-rag-metoyerLab
    python src/benchmark.py

NOTE: build_rag() always calls from_texts(), which uploads new vectors to
Pinecone even if the index already exists. Run this script on fresh indices,
or delete and recreate them beforehand to avoid duplicate vectors.
"""

import importlib.util
import math
import os
import sys
import time
import statistics

import chromadb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.dirname(__file__))
from rag_utils import build_rag, RagPipeline

load_dotenv()

# geminiRag/rag_utils.py shares the module name "rag_utils" with src/rag_utils.py,
# so it's loaded under an aliased module name to avoid clobbering the import above.
_GEMINI_UTILS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "geminiRag", "rag_utils.py"
)
_spec = importlib.util.spec_from_file_location("gemini_rag_utils", _GEMINI_UTILS_PATH)
gemini_rag_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gemini_rag_utils)
build_gemini_rag = gemini_rag_utils.build_gemini_rag

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------
# Mix of chapter-specific and cross-chapter questions.
# "expected_chapters" = chapter numbers that should appear in citations.
TEST_QUERIES = [
    # --- Chapter 2 specific ---
    {
        "question": "What are the Gulf of Execution and Gulf of Evaluation?",
        "expected_chapters": ["two", "2"],
        "type": "chapter_specific",
        "relevant_to": "ch02",
    },
    {
        "question": "Explain the seven stages of action model.",
        "expected_chapters": ["two", "2"],
        "type": "chapter_specific",
        "relevant_to": "ch02",
    },
    # --- Chapter 4 specific ---
    {
        "question": "What are the four kinds of constraints described in the book?",
        "expected_chapters": ["four", "4"],
        "type": "chapter_specific",
        "relevant_to": "ch04",
    },
    {
        "question": "How do physical and semantic constraints guide user behavior?",
        "expected_chapters": ["four", "4"],
        "type": "chapter_specific",
        "relevant_to": "ch04",
    },
    # --- Cross-chapter ---
    {
        "question": "How do affordances and conceptual models work together in design?",
        "expected_chapters": ["one", "two", "three", "1", "2", "3"],
        "type": "cross_chapter",
        "relevant_to": "multiple",
    },
    {
        "question": "What role does feedback play in bridging the gulf between user and system?",
        "expected_chapters": ["one", "two", "four", "1", "2", "4"],
        "type": "cross_chapter",
        "relevant_to": "multiple",
    },
]

QUERY_TRIALS = 3  # how many times to repeat each query for latency averaging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_chunking(chapter_list: list[str], chapters_dir: str = "data/chapters") -> dict:
    """
    Measure local chunking time and chunk count without hitting any API.
    Returns {"chunk_count": int, "char_count": int, "chunking_time_s": float}
    """
    texts = []
    for fname in chapter_list:
        path = os.path.join(chapters_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    full_text = "\n".join(texts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    t0 = time.perf_counter()
    chunks = splitter.split_text(full_text)
    elapsed = time.perf_counter() - t0

    return {
        "chunk_count": len(chunks),
        "char_count": len(full_text),
        "chunking_time_s": elapsed,
    }


def _reset_chroma_collection(collection_name: str, persist_directory: str = "./chroma_db") -> None:
    """
    Delete a Chroma collection if it exists, so repeated benchmark runs start
    from a clean index instead of silently accumulating duplicate chunks.
    Only touches the named collection; other collections in the same
    persist_directory (e.g. from interactive geminiRag/rag_script.py use) are untouched.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def measure_retrieval_similarity(pipeline: RagPipeline, queries: list[dict]) -> dict:
    """
    For each query, embed the question and its retrieved chunks with the
    pipeline's own embedding model, then average their cosine similarity.

    NOTE: OpenAI and Gemini embeddings live in different vector spaces
    (different dimensions, different training), so these averages are only
    comparable as a relative signal across systems, not an exact apples-to-apples score.
    """
    per_query = []
    for q in queries:
        question = q["question"]
        query_vec = pipeline.embeddings.embed_query(question)
        docs = pipeline.retriever.invoke(question)
        if not docs:
            continue
        doc_vecs = pipeline.embeddings.embed_documents([d.page_content for d in docs])
        sims = [cosine_similarity(query_vec, v) for v in doc_vecs]
        per_query.append({
            "question": question,
            "avg_similarity": statistics.mean(sims),
        })

    avg_similarity = (
        statistics.mean(r["avg_similarity"] for r in per_query) if per_query else 0.0
    )
    return {"avg_similarity": avg_similarity, "per_query": per_query}


def score_answer(llm: ChatOpenAI, question: str, answer: str) -> float:
    """
    LLM-as-judge: returns a 1–10 score for relevance + completeness.
    Prompt asks for a single integer on one line to make parsing reliable.
    """
    judge_prompt = f"""You are an impartial judge evaluating the quality of an answer.
Score the following answer on a scale of 1 to 10, where:
  1  = completely wrong or irrelevant
  5  = partially correct, misses key details
  10 = fully correct, comprehensive, well-cited

Question: {question}

Answer:
{answer}

Respond with ONLY a single integer between 1 and 10, nothing else."""

    try:
        raw = llm.invoke(judge_prompt).content.strip()
        # extract first integer found
        for token in raw.split():
            if token.isdigit():
                val = int(token)
                if 1 <= val <= 10:
                    return float(val)
        return 5.0  # fallback if parsing fails
    except Exception as e:
        print(f"    [judge error] {e}")
        return 5.0


def chapter_citation_hit(answer: str, expected_chapters: list[str]) -> bool:
    """
    Returns True if the answer mentions any of the expected chapter identifiers
    (case-insensitive). Checks both the answer text and the 'Chapter(s) used:' line.
    """
    answer_lower = answer.lower()
    return any(ch.lower() in answer_lower for ch in expected_chapters)


def run_queries(chain, queries: list[dict], llm: ChatOpenAI, n_trials: int) -> list[dict]:
    """
    Run each query n_trials times, collect latency, then score once.
    Returns list of result dicts.
    """
    results = []
    for q in queries:
        question = q["question"]
        latencies = []

        # Warm-up + timed trials
        answer = ""
        for i in range(n_trials):
            t0 = time.perf_counter()
            answer = chain.invoke(question)
            latencies.append(time.perf_counter() - t0)
            if i < n_trials - 1:
                time.sleep(0.5)  # avoid rate limiting between trials

        score = score_answer(llm, question, answer)
        citation_hit = chapter_citation_hit(answer, q["expected_chapters"])

        results.append({
            "question": question,
            "type": q["type"],
            "relevant_to": q["relevant_to"],
            "expected_chapters": q["expected_chapters"],
            "answer_snippet": answer[:200].replace("\n", " "),
            "latency_avg_s": statistics.mean(latencies),
            "latency_min_s": min(latencies),
            "latency_max_s": max(latencies),
            "llm_score": score,
            "citation_hit": citation_hit,
        })

        print(f"    Q: {question[:60]}...")
        print(f"       latency avg={results[-1]['latency_avg_s']:.2f}s  "
              f"score={score:.0f}/10  citation={'✓' if citation_hit else '✗'}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(label: str, chunk_stats: dict, build_time: float, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Chunks indexed  : {chunk_stats['chunk_count']}")
    print(f"  Chars indexed   : {chunk_stats['char_count']:,}")
    print(f"  Chunking time   : {chunk_stats['chunking_time_s']*1000:.1f} ms")
    print(f"  Total build time: {build_time:.2f} s  "
          f"(includes embedding + Pinecone upload)")

    all_latencies = [r["latency_avg_s"] for r in results]
    all_scores = [r["llm_score"] for r in results]
    citation_rate = sum(r["citation_hit"] for r in results) / len(results) * 100

    print(f"\n  --- Query Performance ({QUERY_TRIALS} trials each) ---")
    print(f"  Avg query latency : {statistics.mean(all_latencies):.2f} s")
    print(f"  Min query latency : {min(all_latencies):.2f} s")
    print(f"  Max query latency : {max(all_latencies):.2f} s")

    print(f"\n  --- Accuracy ---")
    print(f"  Avg LLM score     : {statistics.mean(all_scores):.1f} / 10")
    print(f"  Citation hit rate : {citation_rate:.0f}%")

    # Per-query breakdown
    print(f"\n  {'Type':<16} {'Score':>6} {'Latency':>9} {'Cite':>5}  Question")
    print(f"  {'-'*16} {'-'*6} {'-'*9} {'-'*5}  {'-'*40}")
    for r in results:
        print(f"  {r['type']:<16} {r['llm_score']:>5.0f}/10 "
              f"{r['latency_avg_s']:>7.2f}s {'✓' if r['citation_hit'] else '✗':>5}  "
              f"{r['question'][:50]}")


def print_comparison(full_results: list[dict], chap_results: list[dict]):
    """Side-by-side delta table."""
    print(f"\n{'='*60}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Question':<42} {'Full score':>10} {'Ch score':>8} {'Full lat':>9} {'Ch lat':>8}")
    print(f"  {'-'*42} {'-'*10} {'-'*8} {'-'*9} {'-'*8}")

    for f, c in zip(full_results, chap_results):
        winner_score = "Full" if f["llm_score"] > c["llm_score"] else (
            "Chap" if c["llm_score"] > f["llm_score"] else " Tie")
        winner_lat = "Full" if f["latency_avg_s"] < c["latency_avg_s"] else "Chap"
        print(f"  {f['question'][:42]:<42} "
              f"{f['llm_score']:>9.0f}/10 "
              f"{c['llm_score']:>7.0f}/10 "
              f"{f['latency_avg_s']:>8.2f}s "
              f"{c['latency_avg_s']:>7.2f}s "
              f"  score→{winner_score} lat→{winner_lat}")

    # Aggregate by query type
    for qtype in ["chapter_specific", "cross_chapter"]:
        f_scores = [r["llm_score"] for r in full_results if r["type"] == qtype]
        c_scores = [r["llm_score"] for r in chap_results if r["type"] == qtype]
        if f_scores and c_scores:
            print(f"\n  [{qtype}] avg score — Full: {statistics.mean(f_scores):.1f}  "
                  f"Chapter: {statistics.mean(c_scores):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CHAPTERS_DIR = "data/chapters"
    all_chapters = sorted(
        [f for f in os.listdir(CHAPTERS_DIR) if f.endswith(".txt")]
    )

    print("=" * 60)
    print("  RAG BENCHMARK: Full-book vs Chapter-specific")
    print("=" * 60)
    print(f"  Chapters available : {all_chapters}")
    print(f"  Test queries       : {len(TEST_QUERIES)}")
    print(f"  Trials per query   : {QUERY_TRIALS}")
    print()

    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ------------------------------------------------------------------
    # 1. Full-book RAG (all chapters)
    # ------------------------------------------------------------------
    print("[1/2] Building Full-book RAG (all chapters)...")
    full_chunk_stats = measure_chunking(all_chapters, CHAPTERS_DIR)
    print(f"      Chunks: {full_chunk_stats['chunk_count']}, "
          f"Chars: {full_chunk_stats['char_count']:,}, "
          f"Local chunk time: {full_chunk_stats['chunking_time_s']*1000:.1f} ms")

    t_build_start = time.perf_counter()
    full_pipeline = build_rag(chapter_list=all_chapters, index_name="manualrag-full")
    full_build_time = time.perf_counter() - t_build_start
    print(f"      Build complete in {full_build_time:.2f}s\n")

    print("      Running queries...")
    full_results = run_queries(full_pipeline.chain, TEST_QUERIES, judge_llm, QUERY_TRIALS)

    # ------------------------------------------------------------------
    # 2. Chapter-specific RAG (ch02 only — most test queries target ch02/ch04)
    # ------------------------------------------------------------------
    # Using ch02 as the representative "chapter-specific" index.
    # It should score well on ch02 questions and poorly on cross-chapter ones.
    CHAPTER_SPECIFIC = ["ch02.txt"]
    print(f"\n[2/2] Building Chapter-specific RAG ({CHAPTER_SPECIFIC})...")
    chap_chunk_stats = measure_chunking(CHAPTER_SPECIFIC, CHAPTERS_DIR)
    print(f"      Chunks: {chap_chunk_stats['chunk_count']}, "
          f"Chars: {chap_chunk_stats['char_count']:,}, "
          f"Local chunk time: {chap_chunk_stats['chunking_time_s']*1000:.1f} ms")

    t_build_start = time.perf_counter()
    chap_pipeline = build_rag(chapter_list=CHAPTER_SPECIFIC, index_name="manualrag-ch2")
    chap_build_time = time.perf_counter() - t_build_start
    print(f"      Build complete in {chap_build_time:.2f}s\n")

    print("      Running queries...")
    chap_results = run_queries(chap_pipeline.chain, TEST_QUERIES, judge_llm, QUERY_TRIALS)

    # ------------------------------------------------------------------
    # 3. Reports (Full-book vs Chapter-specific, OpenAI only)
    # ------------------------------------------------------------------
    print_summary("FULL-BOOK RAG", full_chunk_stats, full_build_time, full_results)
    print_summary("CHAPTER-SPECIFIC RAG (ch02)", chap_chunk_stats, chap_build_time, chap_results)
    print_comparison(full_results, chap_results)

    # ------------------------------------------------------------------
    # 4. OpenAI vs Gemini retrieval cosine similarity (full-book, both sides)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  RETRIEVAL COSINE SIMILARITY: OpenAI vs Gemini (full-book)")
    print(f"{'='*60}")

    GEMINI_COLLECTION = "benchmark-full-book"
    _reset_chroma_collection(GEMINI_COLLECTION)

    chapter_paths = [os.path.join(CHAPTERS_DIR, f) for f in all_chapters]
    print("\nBuilding Gemini full-book RAG (Chroma, same chapter .txt files as OpenAI)...")
    gemini_pipeline = build_gemini_rag(
        file_path=chapter_paths,
        collection_name=GEMINI_COLLECTION,
        search_type="similarity",
        k=10,
    )

    if gemini_pipeline is None:
        print("  Skipped: Gemini pipeline failed to build (see error above).")
    else:
        openai_similarity = measure_retrieval_similarity(full_pipeline, TEST_QUERIES)
        gemini_similarity = measure_retrieval_similarity(gemini_pipeline, TEST_QUERIES)

        openai_avg = openai_similarity["avg_similarity"]
        gemini_avg = gemini_similarity["avg_similarity"]
        leader = "OpenAI" if openai_avg > gemini_avg else "Gemini"
        gap_pct = (
            abs(openai_avg - gemini_avg) / max(openai_avg, gemini_avg) * 100
            if max(openai_avg, gemini_avg) > 0 else 0.0
        )

        print(f"\n  OpenAI avg cosine similarity : {openai_avg:.4f}")
        print(f"  Gemini avg cosine similarity : {gemini_avg:.4f}")
        print(f"  {leader} outperformed by {gap_pct:.1f}%")
        print(
            "\n  Note: OpenAI and Gemini embeddings live in different vector spaces "
            "(1536-dim vs 768-dim), so this is a relative signal, not an exact "
            "apples-to-apples score."
        )

    print("\n  Done.")


if __name__ == "__main__":
    main()
