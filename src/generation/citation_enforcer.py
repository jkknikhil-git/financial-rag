"""
citation_enforcer.py
--------------------
Validates that an LLM-generated answer is grounded in the retrieved chunks.

The enforcer checks that every factual claim in the answer maps to at least
one retrieved chunk. If the answer contains claims not supported by any chunk,
it is declined rather than returned — preventing hallucinated financial data
from reaching the user.

Grounding check strategy:
  1. Extract key factual phrases from the answer (numbers, named entities, metrics).
  2. For each phrase, check if it appears in any retrieved chunk.
  3. If > UNGROUNDED_THRESHOLD of phrases are ungrounded, decline the answer.

Usage:
    from src.generation.citation_enforcer import enforce_citations
    result = enforce_citations(query, answer, chunks)
"""

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

# Fraction of extracted facts that must be grounded for the answer to pass
GROUNDING_THRESHOLD = 0.60

DECLINE_MESSAGE = (
    "The retrieved documents do not contain sufficient information to answer "
    "this question with confidence. Please rephrase your question or ask about "
    "a specific company and fiscal year."
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CitationResult:
    passed:           bool         # True if answer is grounded enough to return
    answer:           str          # Final answer (original or decline message)
    grounding_score:  float        # Fraction of facts found in chunks
    ungrounded_facts: list[str]    # Facts not found in any chunk
    cited_chunks:     list[str]    # Chunk IDs that supported the answer


# ── Public API ────────────────────────────────────────────────────────────────

def enforce_citations(
    query: str,
    answer: str,
    chunks: list[dict[str, Any]],
    threshold: float = GROUNDING_THRESHOLD,
) -> CitationResult:
    """
    Check whether an LLM answer is grounded in the retrieved chunks.

    Args:
        query:     The original user question (used for logging).
        answer:    The LLM-generated answer to validate.
        chunks:    Retrieved and reranked chunks used to generate the answer.
        threshold: Minimum fraction of facts that must be grounded.

    Returns:
        CitationResult with pass/fail, grounding score, and final answer.
    """
    if not answer or not answer.strip():
        logger.warning("Empty answer received — declining")
        return CitationResult(
            passed=False,
            answer=DECLINE_MESSAGE,
            grounding_score=0.0,
            ungrounded_facts=[],
            cited_chunks=[],
        )

    # Check if the LLM already self-declined (respecting the system prompt)
    if _is_self_declined(answer):
        logger.info("LLM self-declined — passing through decline message")
        return CitationResult(
            passed=False,
            answer=answer,
            grounding_score=0.0,
            ungrounded_facts=[],
            cited_chunks=[],
        )

    # Build a searchable corpus from all retrieved chunks
    chunk_corpus = [c.get("content", "").lower() for c in chunks]
    chunk_ids    = [c.get("chunk_id", "") for c in chunks]

    # Extract key facts from the answer
    facts = _extract_facts(answer)

    if not facts:
        # No extractable facts — treat as grounded (likely a short explanation)
        logger.debug("No facts extracted from answer — passing as-is")
        return CitationResult(
            passed=True,
            answer=answer,
            grounding_score=1.0,
            ungrounded_facts=[],
            cited_chunks=chunk_ids,
        )

    # Check each fact against the chunk corpus
    grounded_facts:   list[str] = []
    ungrounded_facts: list[str] = []
    cited_chunk_ids:  set[str]  = set()

    for fact in facts:
        found = False
        for i, chunk_text in enumerate(chunk_corpus):
            if _fact_in_chunk(fact, chunk_text):
                grounded_facts.append(fact)
                cited_chunk_ids.add(chunk_ids[i])
                found = True
                break
        if not found:
            ungrounded_facts.append(fact)

    grounding_score = len(grounded_facts) / len(facts) if facts else 1.0
    passed          = grounding_score >= threshold

    logger.info(
        f"Citation check | {len(grounded_facts)}/{len(facts)} facts grounded "
        f"({grounding_score:.0%}) | {'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        logger.warning(f"Ungrounded facts: {ungrounded_facts}")

    return CitationResult(
        passed=passed,
        answer=answer if passed else DECLINE_MESSAGE,
        grounding_score=round(grounding_score, 4),
        ungrounded_facts=ungrounded_facts,
        cited_chunks=list(cited_chunk_ids),
    )


# ── Fact extraction ───────────────────────────────────────────────────────────

def _extract_facts(text: str) -> list[str]:
    """
    Extract key verifiable facts from an answer.
    Focuses on financial figures, percentages, and named metrics.
    """
    facts: list[str] = []

    # Dollar amounts: $89.5 billion, $2.3B, $1,234 million
    facts += re.findall(r"\$[\d,\.]+\s*(?:billion|million|trillion|B|M|T)?", text, re.IGNORECASE)

    # Percentages: 15.2%, 3.8 percent
    facts += re.findall(r"[\d,\.]+\s*(?:%|percent)", text, re.IGNORECASE)

    # Year references: FY2023, Q3 2024, fiscal 2022
    facts += re.findall(r"(?:FY|Q[1-4]|fiscal\s+year\s+)\s*20\d{2}", text, re.IGNORECASE)

    # Standalone years in financial context
    facts += re.findall(r"\b20(?:2[0-9]|1[5-9])\b", text)

    # Large plain numbers: 1,234, 89.5
    facts += re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", text)

    # Deduplicate, normalise, and filter noise
    seen  = set()
    clean = []
    for f in facts:
        normalised = f.strip().lower()
        if normalised not in seen and len(normalised) > 1:
            seen.add(normalised)
            clean.append(normalised)

    return clean


def _fact_in_chunk(fact: str, chunk_text: str) -> bool:
    """
    Check whether a fact appears in a chunk, tolerant of unit differences.

    SEC tables express amounts in millions, so the LLM may produce
    "$391,035 million" while the chunk only contains "391,035".
    We normalise both sides before comparing.
    """
    # 1. Direct substring match
    if fact in chunk_text:
        return True

    # 2. Unit-normalised match
    #    "$391,035 million" -> "391035"  should match chunk "391,035" -> "391035"
    def _normalise(s: str) -> str:
        s = re.sub(r"[\$,\s]", "", s)
        s = re.sub(
            r"(?:million|billion|trillion|thousand)",
            "", s, flags=re.IGNORECASE,
        )
        return s.strip()

    norm_fact  = _normalise(fact)
    norm_chunk = _normalise(chunk_text)

    if norm_fact and len(norm_fact) > 2 and norm_fact in norm_chunk:
        return True

    # 3. Year-only match — "fiscal year 2024" -> look for "2024" in chunk
    for year in re.findall(r"20\d{2}", fact):
        if year in chunk_text:
            return True

    # 4. Percentage numeric match — "15.2 percent" should match "15.2%"
    pct = re.search(r"([\d\.]+)\s*(?:%|percent)", fact, re.IGNORECASE)
    if pct and pct.group(1) in chunk_text:
        return True

    return False


def _is_self_declined(answer: str) -> bool:
    """Check if the LLM already declined to answer per the system prompt."""
    decline_phrases = [
        "do not contain sufficient information",
        "insufficient information",
        "cannot answer",
        "not enough information",
        "unable to answer",
    ]
    lower = answer.lower()
    return any(phrase in lower for phrase in decline_phrases)
