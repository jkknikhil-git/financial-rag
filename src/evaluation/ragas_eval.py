"""
ragas_eval.py
-------------
Offline RAGAS evaluation for the Financial RAG pipeline.

Metrics:
  Faithfulness       — Is the answer supported by the retrieved context?
  Answer Relevancy   — Is the answer relevant to the question asked?
  Answer Correctness — Does the answer match the human-verified ground truth?
                       (Only runs when ground truth is available — FiQA source)

CI behaviour:
  Exit 0  -> all metrics above thresholds defined in prompts/v2.yaml
  Exit 1  -> one or more metrics below threshold (PR build fails)

Usage:
    python -m src.evaluation.ragas_eval                     # FiQA, 10 questions
    python -m src.evaluation.ragas_eval --limit 3           # quick smoke test
    python -m src.evaluation.ragas_eval --source curated    # curated fallback
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.evaluation.fiqa_loader import load_test_questions
from src.generation.generator import load_thresholds, generate_answer
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank

load_dotenv()

ROOT_DIR    = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "evaluation_results"

INTER_QUERY_DELAY = 3.0   # seconds between pipeline calls — respects Groq rate limits


# ── Public API ────────────────────────────────────────────────────────────────

def run_evaluation(
    source: str = "fiqa",
    limit: int | None = None,
    prompt_version: str = "v2",
) -> dict[str, float]:
    """
    Run the full RAGAS evaluation loop.

    Args:
        source:         "fiqa" (default, human-verified) or "curated".
        limit:          Max questions to evaluate. None = all.
        prompt_version: Prompt config version for thresholds + generation.

    Returns:
        Dict of metric_name -> score.
    """
    thresholds = load_thresholds(prompt_version)
    questions  = load_test_questions(source=source, limit=limit)

    logger.info(
        f"RAGAS evaluation | questions={len(questions)} | "
        f"prompt={prompt_version} | thresholds={thresholds}"
    )

    # ── Step 1: Run pipeline for each question ─────────────────────────────
    samples: list[dict] = []

    for i, q_dict in enumerate(questions, start=1):
        question      = q_dict["question"]
        ticker_filter = q_dict.get("ticker")
        ground_truth  = q_dict.get("ground_truth", "")

        logger.info(f"[{i}/{len(questions)}] {question[:80]}")

        try:
            chunks   = hybrid_search(question, n_results=20, ticker_filter=ticker_filter)
            reranked = rerank(question, chunks, top_n=5)
            answer   = generate_answer(question, reranked, prompt_version=prompt_version)
            contexts = [c["content"] for c in reranked]

            samples.append({
                "user_input":         question,
                "response":           answer,
                "retrieved_contexts": contexts,
                "ground_truth":       ground_truth,
            })
            logger.success(f"  -> Answer: '{answer[:80]}...'")

        except Exception as exc:
            logger.error(f"  -> Pipeline failed: {exc}")

        if i < len(questions):
            time.sleep(INTER_QUERY_DELAY)

    if not samples:
        logger.error("No samples collected — cannot evaluate")
        sys.exit(1)

    logger.info(f"Collected {len(samples)} sample(s) for RAGAS evaluation")

    # ── Step 2: Run RAGAS ─────────────────────────────────────────────────
    scores = _run_ragas(samples)

    # ── Step 3: Save results ──────────────────────────────────────────────
    _save_results(scores, samples, prompt_version)

    # ── Step 4: Check against thresholds ─────────────────────────────────
    _check_thresholds(scores, thresholds)

    return scores


# ── RAGAS runner ──────────────────────────────────────────────────────────────

def _run_ragas(samples: list[dict]) -> dict[str, float]:
    """
    Configure and run RAGAS metrics.
    Targets RAGAS 0.2.x API (LangChain-native, HuggingFace Dataset format).
    """
    try:
        # Shim: ragas 0.2.x imports ChatVertexAI from langchain_community at init
        # but langchain-community 0.4+ removed this submodule.
        import sys as _sys
        from types import ModuleType
        _shim_key = "langchain_community.chat_models.vertexai"
        if _shim_key not in _sys.modules:
            _shim_mod = ModuleType(_shim_key)
            class _DummyChatVertexAI:
                pass
            _shim_mod.ChatVertexAI = _DummyChatVertexAI
            _sys.modules[_shim_key] = _shim_mod

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info("Configuring RAGAS 0.2.x with Groq LLM + BGE embeddings...")

        # Use 70b model with generous token budget for RAGAS sub-prompts
        evaluator_llm = LangchainLLMWrapper(
            ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.environ["GROQ_API_KEY"],
                temperature=0.0,
                max_tokens=2048,
            )
        )

        # Explicit BGE embeddings — prevents ragas falling back to OpenAI
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        # Assign LLM + embeddings on pre-instantiated metric objects (RAGAS 0.2.x style)
        faithfulness.llm            = evaluator_llm
        answer_relevancy.llm        = evaluator_llm
        answer_relevancy.embeddings = evaluator_embeddings
        answer_correctness.llm      = evaluator_llm

        # Groq only supports n=1. AnswerRelevancy generates `strictness` questions
        # per answer in a single call via n=strictness (default 3), which Groq
        # rejects with "'n' : number must be at most 1" — silently zeroing the
        # metric. strictness=1 generates one question per answer: Groq-compatible.
        answer_relevancy.strictness = 1

        # Build dataset — include ground_truth when available (FiQA source)
        has_ground_truth = any(s.get("ground_truth", "").strip() for s in samples)

        data: dict = {
            "question": [s["user_input"]        for s in samples],
            "answer":   [s["response"]           for s in samples],
            "contexts": [s["retrieved_contexts"] for s in samples],
        }

        if has_ground_truth:
            data["ground_truth"] = [s.get("ground_truth", "") for s in samples]
            metrics_to_run = [faithfulness, answer_relevancy, answer_correctness]
            logger.info(
                "Ground truth available (FiQA) — running Faithfulness + "
                "Answer Relevancy + Answer Correctness"
            )
        else:
            metrics_to_run = [faithfulness, answer_relevancy]
            logger.info("No ground truth — running Faithfulness + Answer Relevancy")

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation (this may take a few minutes)...")
        result = evaluate(
            dataset = dataset,
            metrics = metrics_to_run,
        )

        df = result.to_pandas()

        import math
        def _safe(col: str) -> float:
            if col not in df.columns:
                return 0.0
            v = float(df[col].mean())
            return 0.0 if math.isnan(v) else round(v, 4)

        scores = {
            "faithfulness":     _safe("faithfulness"),
            "answer_relevancy": _safe("answer_relevancy"),
        }
        if has_ground_truth:
            scores["answer_correctness"] = _safe("answer_correctness")

        logger.info(f"RAGAS scores: {scores}")
        return scores

    except ImportError as exc:
        logger.error(f"RAGAS import failed: {exc}")
        logger.info("Install with: pip install 'ragas>=0.2.0,<0.3.0'")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"RAGAS evaluation failed: {exc}")
        raise


# ── Threshold check ───────────────────────────────────────────────────────────

def _check_thresholds(
    scores: dict[str, float],
    thresholds: dict[str, float],
) -> None:
    """
    Compare scores against thresholds from the prompt config.
    Exits with code 1 if any metric fails — causes GitHub Actions CI to fail.
    """
    divider = "=" * 60
    logger.info(divider)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info(divider)

    metric_map = {
        "faithfulness":       "faithfulness",
        "answer_relevancy":   "answer_relevance",
        "answer_correctness": "answer_correctness",
    }

    all_passed = True
    import math

    for metric, yaml_key in metric_map.items():
        if metric not in scores:
            continue
        raw       = scores[metric]
        score     = 0.0 if (raw is None or math.isnan(raw)) else raw
        threshold = thresholds.get(yaml_key, 0.0)
        passed    = score >= threshold
        icon      = "PASS" if passed else "FAIL"

        logger.info(
            f"  {icon}  {metric:<22} {score:.4f}  "
            f"(threshold: {threshold:.2f})"
        )
        if not passed:
            all_passed = False

    logger.info(divider)

    if all_passed:
        logger.success("All metrics passed — build GREEN")
        sys.exit(0)
    else:
        logger.error("One or more metrics below threshold — build FAILED")
        sys.exit(1)


# ── Results persistence ───────────────────────────────────────────────────────

def _save_results(
    scores: dict[str, float],
    samples: list[dict],
    prompt_version: str,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp":      timestamp,
        "prompt_version": prompt_version,
        "scores":         scores,
        "n_samples":      len(samples),
        "samples": [
            {
                "question":     s["user_input"],
                "answer":       s["response"],
                "ground_truth": s.get("ground_truth", ""),
            }
            for s in samples
        ],
    }
    result_path = RESULTS_DIR / f"ragas_{timestamp}.json"
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(f"Results saved to {result_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Financial RAG pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["fiqa", "curated"],
        default="fiqa",
        help="Question source (fiqa = human-verified, curated = SEC-specific).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate.",
    )
    parser.add_argument(
        "--prompt-version",
        default="v2",
        dest="prompt_version",
    )
    args = parser.parse_args()
    run_evaluation(
        source         = args.source,
        limit          = args.limit,
        prompt_version = args.prompt_version,
    )


if __name__ == "__main__":
    main()
