"""
Financial RAG
======================================
A production-grade RAG pipeline for SEC 10-K and 10-Q filings.

Modules:
    ingestion   — EDGAR fetching, table-aware PDF parsing, chunking
    indexing    — ChromaDB vector store and BM25 index builders
    retrieval   — Hybrid search (BM25 + vector) with RRF and cross-encoder reranking
    generation  — Citation-enforced answer generation via Groq LLM
    evaluation  — RAGAS metrics and FiQA dataset evaluation
    pipeline    — LangGraph orchestration graph
"""
