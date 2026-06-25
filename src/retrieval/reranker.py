"""
reranker.py
-----------
Cross-encoder reranker using the Cohere Rerank API.

Takes the top-k RRF-merged chunks and re-scores them with a cross-encoder
model that jointly encodes the query and each chunk together — much more
accurate than the bi-encoder embeddings used in first-stage retrieval.

Cohere free tier: 1,000 rerank API calls per month.
No local GPU required — all compute is on Cohere's servers.

Usage:
    from src.retrieval.reranker import rerank
    reranked = rerank(query="What was Apple's revenue?", chunks=rrf_results, top_n=5)
"""

import os
from typing import Any

import cohere
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

COHERE_RERANK_MODEL = "rerank-v3.5"
DEFAULT_TOP_N       = 5     # chunks passed to the LLM after reranking


# ── Public API ────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved chunks using the Cohere cross-encoder.

    Args:
        query:  The original user query.
        chunks: RRF-merged retrieval results (each must have chunk_id, content, metadata).
        top_n:  Number of top chunks to return after reranking.

    Returns:
        Top-n chunks sorted by descending Cohere relevance score,
        each with an added 'rerank_score' field.
    """
    if not chunks:
        logger.warning("Reranker received empty chunk list")
        return []

    top_n = min(top_n, len(chunks))
    logger.info(f"Reranking {len(chunks)} chunks -> top {top_n}")

    try:
        client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

        response = client.rerank(
            model     = COHERE_RERANK_MODEL,
            query     = query,
            documents = [c["content"] for c in chunks],
            top_n     = top_n,
        )

        reranked: list[dict[str, Any]] = []
        for result in response.results:
            chunk = chunks[result.index]
            reranked.append({
                **chunk,
                "rerank_score": round(result.relevance_score, 4),
            })

        logger.info(
            f"Reranked | top score={reranked[0]['rerank_score']} | "
            f"bottom score={reranked[-1]['rerank_score']}"
        )
        return reranked

    except Exception as exc:
        logger.error(f"Cohere rerank failed: {exc}")
        logger.warning("Falling back to RRF ordering (no reranking)")
        return chunks[:top_n]
