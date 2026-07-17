"""
fiqa_loader.py
--------------
Loads financial QA test cases for RAGAS evaluation.

Primary source: FiQA dataset (explodinggradients/fiqa, ragas_eval split)
  - Human-verified Q&A pairs from financial forums and analyst reports
  - Includes ground_truth answers for reference-based metrics
  - Used for: Faithfulness, Answer Relevancy, Answer Correctness

Fallback: Curated test set (if FiQA download fails)
  - Questions specific to our ingested companies (AAPL, MSFT, NVDA, GOOGL, META)
  - No ground truth — reference-free metrics only

Usage:
    from src.evaluation.fiqa_loader import load_test_questions
    questions = load_test_questions()           # FiQA (default)
    questions = load_test_questions("curated")  # curated fallback
"""

from __future__ import annotations

from typing import Literal

from loguru import logger


# ── Curated fallback (no ground truth) ───────────────────────────────────────

CURATED_QUESTIONS: list[dict[str, str]] = [
    {
        "question":     "What was Apple's total net sales for fiscal year 2024?",
        "ground_truth": "",
        "ticker":       "AAPL",
    },
    {
        "question":     "What was Microsoft's total revenue for fiscal year 2024?",
        "ground_truth": "",
        "ticker":       "MSFT",
    },
    {
        "question":     "What was NVIDIA's total revenue for fiscal year 2024?",
        "ground_truth": "",
        "ticker":       "NVDA",
    },
    {
        "question":     "What were the primary risk factors disclosed by Meta Platforms in its most recent 10-K?",
        "ground_truth": "",
        "ticker":       "META",
    },
    {
        "question":     "What was Alphabet's operating income for fiscal year 2023?",
        "ground_truth": "",
        "ticker":       "GOOGL",
    },
    {
        "question":     "What products and services does Apple report as its main revenue segments?",
        "ground_truth": "",
        "ticker":       "AAPL",
    },
    {
        "question":     "What was NVIDIA's data center revenue for fiscal year 2024?",
        "ground_truth": "",
        "ticker":       "NVDA",
    },
    {
        "question":     "What were Microsoft's Intelligent Cloud segment revenues in fiscal year 2024?",
        "ground_truth": "",
        "ticker":       "MSFT",
    },
]


# ── Public API ────────────────────────────────────────────────────────────────

def load_test_questions(
    source: Literal["fiqa", "curated"] = "fiqa",
    limit: int | None = None,
) -> list[dict[str, str]]:
    """
    Load financial QA test questions for RAGAS evaluation.

    Args:
        source: "fiqa" (default) or "curated" (fallback, no ground truth).
        limit:  Maximum number of questions to return. None = all.

    Returns:
        List of dicts with keys: question, ground_truth, (optionally ticker).
        ground_truth is "" for curated questions (reference-free metrics only).
    """
    if source == "fiqa":
        questions = _load_fiqa_hf(limit or 10)
    else:
        questions = list(CURATED_QUESTIONS)

    if limit is not None:
        questions = questions[:limit]

    has_gt = sum(1 for q in questions if q.get("ground_truth", "").strip())
    logger.info(
        f"Loaded {len(questions)} question(s) from source='{source}' "
        f"({has_gt} with ground truth)"
    )
    return questions


def _load_fiqa_hf(limit: int = 10) -> list[dict[str, str]]:
    """
    Load human-verified Q&A pairs from the FiQA HuggingFace dataset.

    Dataset: explodinggradients/fiqa, split: ragas_eval
    Columns used: question, ground_truth (verified answer)
    """
    try:
        from datasets import load_dataset

        logger.info("Downloading FiQA dataset from HuggingFace...")
        ds = load_dataset(
            "explodinggradients/fiqa",
            "ragas_eval",
            split="baseline",
        )

        questions = []
        for row in ds.select(range(min(limit, len(ds)))):
            gt = row.get("ground_truth", "")
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
            gt = str(gt).strip()

            questions.append({
                "question":     row["question"].strip(),
                "ground_truth": gt,
            })

        logger.success(f"Loaded {len(questions)} FiQA Q&A pairs")
        return questions

    except Exception as exc:
        logger.warning(f"FiQA download failed ({exc}) — falling back to curated set")
        return list(CURATED_QUESTIONS)[:limit]
