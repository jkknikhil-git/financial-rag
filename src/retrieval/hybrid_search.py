"""
hybrid_search.py
----------------
Hybrid retrieval combining BM25 + semantic vector search, fused with
Reciprocal Rank Fusion (RRF). Augmented with HyDE and RAG Fusion.

Full retrieval flow:
  1. HyDE  — generate a hypothetical answer, embed it instead of the raw query.
             Financial answers share vocabulary with financial chunks: better recall.
  2. RAG Fusion — generate N query variants, run retrieval for each independently,
                  fuse all result sets. Reduces vocabulary mismatch.
  3. BM25  — keyword / exact-match leg on the original query.
             Preserves "$89.5B", "ASC 606", "Q3 FY2024" matching.
  4. RRF   — score(d) = sum(1 / (k + rank)) across all result sets.
             Robust to score-scale differences between retrievers.

Usage:
    from src.retrieval.hybrid_search import hybrid_search
    results = hybrid_search("What was Apple's revenue in FY2024?", n_results=20)
"""

import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.indexing.bm25_index import query_bm25
from src.indexing.vector_store import (
    BGE_QUERY_PREFIX,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

# Fast model for HyDE + variant generation (saves llama-3.3-70b for answer generation)
HYDE_MODEL       = "llama-3.1-8b-instant"
N_QUERY_VARIANTS = 2     # additional variants for RAG Fusion
RRF_K            = 60    # smoothing constant from the original RRF paper

HYDE_SYSTEM_PROMPT = """\
You are an expert financial analyst. Given a question about SEC 10-K filings,
write a short passage (2-3 sentences) that would appear in a 10-K filing and
directly answers this question. Include specific financial metrics, percentages,
and dollar amounts where relevant. Respond with only the passage, no preamble.\
"""

VARIANTS_SYSTEM_PROMPT = """\
You are a financial research assistant. Given a question about SEC filings,
generate {n} alternative phrasings of the same question. Each variant should
approach the topic from a slightly different angle to improve document retrieval.
Return only the variants, one per line, no numbering or bullets.\
"""


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    n_results: int = 20,
    ticker_filter: str | None = None,
    use_hyde: bool = True,
    use_rag_fusion: bool = True,
) -> list[dict[str, Any]]:
    """
    Run hybrid BM25 + vector retrieval with RRF fusion.

    Args:
        query:          Natural language query.
        n_results:      Number of candidates to return after RRF.
        ticker_filter:  Restrict results to a single company ticker.
        use_hyde:       Enable HyDE (hypothetical document embedding).
        use_rag_fusion: Enable RAG Fusion (multiple query variants).

    Returns:
        List of result dicts sorted by descending RRF score.
        Each result: {chunk_id, content, metadata, rrf_score, source}
    """
    logger.info(
        f"Hybrid search | hyde={use_hyde} | rag_fusion={use_rag_fusion} | "
        f"query='{query[:70]}...'"
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model  = _load_model()
    result_sets: list[list[dict[str, Any]]] = []

    # ── Leg 1: BM25 on original query (keyword / exact-match) ────────────────
    bm25_results = query_bm25(query, n_results=n_results)
    if bm25_results:
        result_sets.append(bm25_results)
        logger.debug(f"BM25: {len(bm25_results)} results")

    # ── Leg 2: Vector search (HyDE embedding or plain BGE prefix) ────────────
    if use_hyde:
        primary_embedding = _hyde_embedding(query, client, model)
    else:
        primary_embedding = _encode_query(model, query)

    vector_results = _chromadb_search(primary_embedding, n_results, ticker_filter)
    if vector_results:
        result_sets.append(vector_results)
        logger.debug(f"Vector (HyDE={use_hyde}): {len(vector_results)} results")

    # ── Leg 3: RAG Fusion — additional query variants ─────────────────────────
    if use_rag_fusion:
        variants = _generate_variants(query, client, n=N_QUERY_VARIANTS)
        logger.debug(f"RAG Fusion variants: {variants}")

        for variant in variants:
            # BM25 on variant
            v_bm25 = query_bm25(variant, n_results=n_results)
            if v_bm25:
                result_sets.append(v_bm25)

            # Vector on variant (HyDE if enabled)
            v_embedding = (
                _hyde_embedding(variant, client, model) if use_hyde
                else _encode_query(model, variant)
            )
            v_vector = _chromadb_search(v_embedding, n_results, ticker_filter)
            if v_vector:
                result_sets.append(v_vector)

    # ── RRF merge across all result sets ──────────────────────────────────────
    merged      = _rrf_merge(result_sets, k=RRF_K)
    top_results = merged[:n_results]

    logger.info(
        f"RRF: {len(result_sets)} result sets -> "
        f"{len(merged)} unique chunks -> top {len(top_results)}"
    )
    return top_results


# ── HyDE ─────────────────────────────────────────────────────────────────────

def _hyde_embedding(
    query: str,
    client: Groq,
    model: SentenceTransformer,
) -> list[float]:
    """
    Generate a hypothetical answer to the query and return its embedding.
    Falls back to plain query embedding if the LLM call fails.
    """
    try:
        resp = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            max_tokens=150,
            temperature=0.1,
        )
        hypothetical = resp.choices[0].message.content.strip()
        logger.debug(f"HyDE passage: '{hypothetical[:100]}...'")
        return model.encode(
            [hypothetical],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    except Exception as exc:
        logger.warning(f"HyDE failed ({exc}) — falling back to plain query embedding")
        return _encode_query(model, query)


# ── RAG Fusion ────────────────────────────────────────────────────────────────

def _generate_variants(
    query: str,
    client: Groq,
    n: int = 2,
) -> list[str]:
    """Generate n alternative phrasings of the query for RAG Fusion."""
    try:
        resp = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": VARIANTS_SYSTEM_PROMPT.format(n=n),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0.4,
        )
        raw      = resp.choices[0].message.content.strip()
        variants = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and line.strip().lower() != query.lower()
        ]
        return variants[:n]

    except Exception as exc:
        logger.warning(f"Variant generation failed ({exc}) — skipping RAG Fusion")
        return []


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _chromadb_search(
    embedding: list[float],
    n_results: int,
    ticker_filter: str | None,
) -> list[dict[str, Any]]:
    """Query ChromaDB with a precomputed embedding vector."""
    try:
        db         = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = db.get_collection(COLLECTION_NAME)
        where      = {"ticker": ticker_filter} if ticker_filter else None

        raw = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids       = raw.get("ids",       [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        return [
            {
                "chunk_id": cid,
                "content":  doc,
                "metadata": meta,
                "score":    round(1.0 - dist, 4),
                "source":   "vector",
            }
            for cid, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]

    except Exception as exc:
        logger.error(f"ChromaDB search failed: {exc}")
        return []


def _encode_query(model: SentenceTransformer, query: str) -> list[float]:
    """Encode a query string with the BGE query prefix."""
    return model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()


def _load_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 512
    return model


# ── RRF ──────────────────────────────────────────────────────────────────────

def _rrf_merge(
    result_sets: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion across multiple result sets.

    score(d) = sum(1 / (k + rank(d, r)) for r in result_sets where d appears)

    A chunk that ranks 2nd in BM25 and 4th in vector search scores higher
    than one that ranks 1st in only one of them.

    Args:
        result_sets: List of ranked result lists. Each result needs chunk_id.
        k:           Smoothing constant. 60 is standard from the original paper.

    Returns:
        Single merged list sorted by descending RRF score.
    """
    scores:  dict[str, float] = {}
    doc_map: dict[str, dict]  = {}

    for result_set in result_sets:
        for rank, result in enumerate(result_set, start=1):
            cid = result["chunk_id"]
            scores[cid]  = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in doc_map:
                doc_map[cid] = result

    return [
        {**doc_map[cid], "rrf_score": round(score, 6)}
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
