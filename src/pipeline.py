"""
pipeline.py
-----------
LangGraph orchestration graph for the Enterprise Financial Intelligence RAG.

Graph structure:
    query_input
        |
        v
    hybrid_search          (BM25 + vector + RRF + HyDE + RAG Fusion)
        |
        v
    rerank                 (Cohere cross-encoder reranker)
        |
        v
    generate               (Groq LLM with versioned prompt)
        |
        v
    enforce_citations      (grounding check)
        |
      __|__
     |     |
     v     v
   answer decline          (conditional edge based on grounding score)

Usage:
    python -m src.pipeline --query "What was Apple's revenue in FY2024?"
    python -m src.pipeline --query "Compare NVIDIA and AMD gross margins" --ticker NVDA
"""

import argparse
import os
from typing import Any, Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.generation.citation_enforcer import enforce_citations
from src.generation.generator import generate_answer, load_thresholds
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank

load_dotenv()

# ── Graph state ───────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """State passed between LangGraph nodes."""
    query:          str
    ticker_filter:  str | None
    chunks:         list[dict[str, Any]]
    reranked:       list[dict[str, Any]]
    answer:         str
    grounding_score: float
    cited_chunks:   list[str]
    status:         Literal["pending", "answered", "declined"]


# ── Node functions ────────────────────────────────────────────────────────────

def node_retrieve(state: RAGState) -> RAGState:
    """Run hybrid BM25 + vector retrieval with HyDE and RAG Fusion."""
    results = hybrid_search(
        query         = state["query"],
        n_results     = 20,
        ticker_filter = state.get("ticker_filter"),
        use_hyde      = True,
        use_rag_fusion = True,
    )
    return {**state, "chunks": results}


def node_rerank(state: RAGState) -> RAGState:
    """Rerank the top-20 RRF results with the Cohere cross-encoder."""
    reranked = rerank(
        query  = state["query"],
        chunks = state["chunks"],
        top_n  = 5,
    )
    return {**state, "reranked": reranked}


def node_generate(state: RAGState) -> RAGState:
    """Generate an answer from the reranked chunks using the Groq LLM."""
    answer = generate_answer(
        query  = state["query"],
        chunks = state["reranked"],
    )
    return {**state, "answer": answer}


def node_enforce_citations(state: RAGState) -> RAGState:
    """Validate that the answer is grounded in the retrieved chunks."""
    result = enforce_citations(
        query  = state["query"],
        answer = state["answer"],
        chunks = state["reranked"],
    )
    status = "answered" if result.passed else "declined"
    return {
        **state,
        "answer":          result.answer,
        "grounding_score": result.grounding_score,
        "cited_chunks":    result.cited_chunks,
        "status":          status,
    }


# ── Conditional edge ──────────────────────────────────────────────────────────

def route_after_citation(state: RAGState) -> Literal["answer", "decline"]:
    """Route to 'answer' if grounded, 'decline' if not."""
    return "answer" if state["status"] == "answered" else "decline"


def node_answer(state: RAGState) -> RAGState:
    """Terminal node: answer passed citation check."""
    logger.success(f"Answer | grounding={state['grounding_score']:.0%}")
    return state


def node_decline(state: RAGState) -> RAGState:
    """Terminal node: answer failed citation check."""
    logger.warning(f"Declined | grounding={state['grounding_score']:.0%}")
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Construct and compile the LangGraph RAG pipeline."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",          node_retrieve)
    graph.add_node("rerank",            node_rerank)
    graph.add_node("generate",          node_generate)
    graph.add_node("enforce_citations", node_enforce_citations)
    graph.add_node("answer",            node_answer)
    graph.add_node("decline",           node_decline)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve",          "rerank")
    graph.add_edge("rerank",            "generate")
    graph.add_edge("generate",          "enforce_citations")
    graph.add_conditional_edges(
        "enforce_citations",
        route_after_citation,
        {"answer": "answer", "decline": "decline"},
    )
    graph.add_edge("answer",  END)
    graph.add_edge("decline", END)

    return graph.compile()


# ── Public run function ───────────────────────────────────────────────────────

def run_query(
    query: str,
    ticker_filter: str | None = None,
) -> dict[str, Any]:
    """
    Run the full RAG pipeline for a query.

    Args:
        query:         Natural language financial question.
        ticker_filter: Optionally restrict retrieval to one company.

    Returns:
        Final state dict with answer, grounding_score, cited_chunks, status.
    """
    app = build_graph()

    initial_state: RAGState = {
        "query":          query,
        "ticker_filter":  ticker_filter,
        "chunks":         [],
        "reranked":       [],
        "answer":         "",
        "grounding_score": 0.0,
        "cited_chunks":   [],
        "status":         "pending",
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print(f"QUERY : {query}")
    print("=" * 70)
    print(f"STATUS: {final_state['status'].upper()}")
    print(f"GROUNDING: {final_state['grounding_score']:.0%}")
    print("-" * 70)
    print(final_state["answer"])
    print("=" * 70)

    return final_state


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Enterprise Financial Intelligence RAG pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Financial question to answer.",
    )
    parser.add_argument(
        "--ticker", "-t",
        default=None,
        help="Optional: restrict retrieval to a specific ticker (AAPL, MSFT, etc.)",
    )
    args = parser.parse_args()
    run_query(query=args.query, ticker_filter=args.ticker)


if __name__ == "__main__":
    main()
