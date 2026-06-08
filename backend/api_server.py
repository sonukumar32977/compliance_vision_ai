"""
Compliance Vision AI — FastAPI retrieval server (localhost testing)

Run:
    uvicorn api_server:app --reload --host 127.0.0.1 --port 8001

Swagger UI (click Try it out on POST /retrieve):
    http://127.0.0.1:8001/docs

Quick CLI test:
    python test_api.py
"""

from __future__ import annotations

import sys
import time
import re
from contextlib import asynccontextmanager
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import KB_CFG, LLM_CFG
from src.retriever import Retriever

_retriever: Retriever | None = None

# Sample queries for /benchmark and Swagger examples
SAMPLE_QUERIES = [
    "worker without helmet in restricted area",
    "fire extinguisher blocked emergency access",
    "maximum occupancy crowd limit construction site",
    "safety vest high visibility requirement",
    "OSHA head protection hard hat rules",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    if not LLM_CFG.api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env — required for embeddings and retrieval.")
    if not KB_CFG.faiss_index.exists():
        raise RuntimeError(
            f"FAISS index not found at {KB_CFG.faiss_index}. "
            "Run: python build_kb.py --force"
        )
    if not KB_CFG.metadata_json.exists():
        raise RuntimeError(
            f"Metadata not found at {KB_CFG.metadata_json}. "
            "Run: python build_kb.py --force"
        )
    _retriever = Retriever()
    yield
    _retriever = None


app = FastAPI(
    title="Compliance Vision Retrieval API",
    description=(
        "Test your RAG retrieval on localhost.\n\n"
        "**How to test in Swagger:**\n"
        "1. Open **POST /retrieve** → **Try it out**\n"
        "2. Paste a query (or use the example JSON)\n"
        "3. Click **Execute** — check `metrics` (speed + scores) and `chunks` (document text)\n"
        "4. Run **GET /benchmark** to score efficiency across 5 sample queries\n\n"
        "**Score guide:** relevance `score` is 0–1 (higher = better match). "
        f"Scores below {KB_CFG.min_score} are filtered out."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about PPE / safety rules",
        json_schema_extra={"example": "worker without helmet in restricted area"},
    )
    top_k: int = Field(
        default=KB_CFG.top_k,
        ge=1,
        le=20,
        description="How many document chunks to return",
        json_schema_extra={"example": 3},
    )


class RetrieveMetrics(BaseModel):
    total_ms: float
    embed_ms: float
    search_ms: float
    chunk_count: int
    avg_score: float | None
    max_score: float | None
    min_score: float | None
    efficiency: str  # Good | Fair | Poor | No results


class RetrieveResponse(BaseModel):
    query: str
    embedding_model: str
    chunk_count: int
    chunks: list[dict[str, Any]]
    context: str
    metrics: RetrieveMetrics


def _score_efficiency(chunks: list[dict[str, Any]]) -> tuple[RetrieveMetrics, float | None, float | None, float | None]:
    scores = [c["score"] for c in chunks if "score" in c]
    if not scores:
        return (
            RetrieveMetrics(
                total_ms=0,
                embed_ms=0,
                search_ms=0,
                chunk_count=0,
                avg_score=None,
                max_score=None,
                min_score=None,
                efficiency="No results — try a different query or lower min_score in config",
            ),
            None,
            None,
            None,
        )

    avg = sum(scores) / len(scores)
    mx = max(scores)
    mn = min(scores)

    if mx >= 0.55:
        label = "Good — strong match to your safety documents"
    elif mx >= 0.35:
        label = "Fair — related content found; review chunk text"
    else:
        label = "Poor — weak match; rephrase query or rebuild knowledge base"

    return (
        RetrieveMetrics(
            total_ms=0,
            embed_ms=0,
            search_ms=0,
            chunk_count=len(chunks),
            avg_score=round(avg, 4),
            max_score=round(mx, 4),
            min_score=round(mn, 4),
            efficiency=label,
        ),
        avg,
        mx,
        mn,
    )


def _keywords(text: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in stop_words]


def _keyword_retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    if _retriever is None:
        return []

    query_terms = Counter(_keywords(query))
    if not query_terms:
        return []

    ranked: list[tuple[float, Any]] = []
    for rec in _retriever._records:
        text_terms = Counter(_keywords(rec.text))
        overlap = sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items())
        if overlap == 0:
            continue

        coverage = overlap / max(sum(query_terms.values()), 1)
        density = overlap / max(sum(text_terms.values()), 1)
        score = (coverage * 0.85) + (density * 0.15)
        ranked.append((score, rec))

    return [
        {
            "chunk_id": rec.chunk_id,
            "source_file": rec.source_file,
            "page_start": rec.page_start,
            "page_end": rec.page_end,
            "text": rec.text,
            "score": round(float(score), 4),
            "retrieval_mode": "keyword_fallback",
        }
        for score, rec in sorted(ranked, key=lambda item: item[0], reverse=True)[:top_k]
    ]


def _offline_answer(query: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return (
            "No matching safety document chunks were found locally. "
            "Check the PPE knowledge-base files and try a more specific question."
        )

    lines = [
        "{",
        f'  "violation": "Relevant PPE guidance for: {query}",',
        '  "risk_level": "Medium",',
        '  "certainty": "Possible",',
        '  "recommendation": "Review the cited source excerpts and enforce the required PPE before work continues.",',
        '  "cited_rule": {',
        f'    "document": "{chunks[0]["source_file"]}",',
        f'    "page": "{chunks[0]["page_start"]}-{chunks[0]["page_end"]}",',
        '    "section": "Local keyword retrieval fallback"',
        "  }",
        "}",
        "NARRATIVE:",
    ]

    for chunk in chunks[:3]:
        excerpt = re.sub(r"\s+", " ", chunk["text"]).strip()
        lines.append(
            f"- {chunk['source_file']} pages {chunk['page_start']}-{chunk['page_end']}: {excerpt[:320]}"
            f"{'...' if len(excerpt) > 320 else ''}"
        )

    return "\n".join(lines)


def _timed_retrieve(query: str, top_k: int) -> tuple[list[dict[str, Any]], str, RetrieveMetrics]:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    t0 = time.perf_counter()

    t_embed = time.perf_counter()
    from src.embedder import embed_texts

    try:
        vec = embed_texts(_retriever._embedder, [query.strip()])
        embed_ms = (time.perf_counter() - t_embed) * 1000

        t_search = time.perf_counter()
        scores, indices = _retriever._index.search(vec, top_k)
        search_ms = (time.perf_counter() - t_search) * 1000

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < _retriever._min_score:
                continue
            rec = _retriever._records[idx]
            results.append(
                {
                    "chunk_id": rec.chunk_id,
                    "source_file": rec.source_file,
                    "page_start": rec.page_start,
                    "page_end": rec.page_end,
                    "text": rec.text,
                    "score": round(float(score), 4),
                }
            )
    except Exception:
        embed_ms = (time.perf_counter() - t_embed) * 1000
        t_search = time.perf_counter()
        results = _keyword_retrieve(query.strip(), top_k)
        search_ms = (time.perf_counter() - t_search) * 1000

    total_ms = (time.perf_counter() - t0) * 1000
    context = _retriever.format_rag_context(results)
    metrics, _, _, _ = _score_efficiency(results)
    metrics.total_ms = round(total_ms, 1)
    metrics.embed_ms = round(embed_ms, 1)
    metrics.search_ms = round(search_ms, 1)
    if results and results[0].get("retrieval_mode") == "keyword_fallback":
        metrics.efficiency = "Offline keyword fallback - OpenAI embeddings unavailable"

    return results, context, metrics


@app.get("/")
def root():
    return {
        "service": "Compliance Vision Retrieval API",
        "how_to_test": "Open /docs → POST /retrieve → Try it out → Execute",
        "sample_queries": SAMPLE_QUERIES,
        "docs": "/docs",
        "health": "/health",
        "benchmark": "GET /benchmark",
    }


@app.get("/health")
def health():
    index_ok = KB_CFG.faiss_index.exists()
    meta_ok = KB_CFG.metadata_json.exists()
    return {
        "status": "ok" if (index_ok and meta_ok and _retriever is not None) else "degraded",
        "api_key_set": bool(LLM_CFG.api_key),
        "embedding_model": KB_CFG.embedding_model,
        "llm_model": LLM_CFG.model,
        "min_relevance_score": KB_CFG.min_score,
        "index_path": str(KB_CFG.faiss_index),
        "index_exists": index_ok,
        "metadata_exists": meta_ok,
        "retriever_loaded": _retriever is not None,
        "indexed_vectors": _retriever._index.ntotal if _retriever else 0,
        "score_guide": {
            "0.55+": "Good match",
            "0.35-0.54": "Fair match",
            "below_0.35": "Weak match",
            f"filtered_below": KB_CFG.min_score,
        },
    }


@app.get("/sample-queries")
def sample_queries():
    """Copy any of these into POST /retrieve when testing in Swagger."""
    return {"queries": SAMPLE_QUERIES}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    chunks, context, metrics = _timed_retrieve(req.query.strip(), req.top_k)
    return RetrieveResponse(
        query=req.query,
        embedding_model=KB_CFG.embedding_model,
        chunk_count=len(chunks),
        chunks=chunks,
        context=context,
        metrics=metrics,
    )


@app.post("/answer")
def answer(req: RetrieveRequest):
    """Full pipeline: retrieve documents + generate OpenAI answer (slower, uses more API credits)."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    t0 = time.perf_counter()
    chunks, context, metrics = _timed_retrieve(req.query.strip(), req.top_k)

    t_llm = time.perf_counter()
    try:
        text = _retriever.answer(req.query.strip(), top_k=req.top_k)
    except Exception:
        text = _offline_answer(req.query.strip(), chunks)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "query": req.query,
        "answer": text,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "context_preview": context[:500] + ("..." if len(context) > 500 else ""),
        "metrics": {
            **metrics.model_dump(),
            "llm_ms": round(llm_ms, 1),
            "total_with_llm_ms": round(total_ms, 1),
        },
    }


@app.get("/benchmark")
def benchmark():
    """
    Run 5 sample safety queries and return average speed + relevance.
    Use this to quickly judge how efficiently retrieval works.
    """
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    runs = []
    for q in SAMPLE_QUERIES:
        chunks, _, metrics = _timed_retrieve(q, KB_CFG.top_k)
        runs.append(
            {
                "query": q,
                "chunk_count": len(chunks),
                "max_score": metrics.max_score,
                "avg_score": metrics.avg_score,
                "total_ms": metrics.total_ms,
                "efficiency": metrics.efficiency,
                "top_source": chunks[0]["source_file"] if chunks else None,
            }
        )

    with_results = [r for r in runs if r["chunk_count"] > 0]
    avg_latency = sum(r["total_ms"] for r in runs) / len(runs)
    avg_max_score = (
        sum(r["max_score"] for r in with_results if r["max_score"] is not None) / len(with_results)
        if with_results
        else 0
    )

    if avg_max_score >= 0.55:
        overall = "Good"
    elif avg_max_score >= 0.35:
        overall = "Fair"
    else:
        overall = "Needs improvement"

    return {
        "overall_efficiency": overall,
        "average_latency_ms": round(avg_latency, 1),
        "average_max_score": round(avg_max_score, 4),
        "queries_tested": len(SAMPLE_QUERIES),
        "queries_with_results": len(with_results),
        "embedding_model": KB_CFG.embedding_model,
        "indexed_vectors": _retriever._index.ntotal,
        "runs": runs,
    }
