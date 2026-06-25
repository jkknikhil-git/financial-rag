"""
generator.py
------------
Groq LLM answer generator with versioned YAML prompt config.

Loads prompts and model settings from src/generation/prompts/v1.yaml so that
every prompt change is committed as a versioned file — evaluation history stays
traceable to specific prompt versions.

Usage:
    from src.generation.generator import generate_answer
    answer = generate_answer(query="What was Apple's revenue in FY2024?", chunks=reranked)
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from groq import Groq
from loguru import logger

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT_DIR / "src" / "generation" / "prompts"
DEFAULT_PROMPT_VERSION = "v1"


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    """
    Generate a grounded answer using the Groq LLM and retrieved chunks.

    Args:
        query:          The user's question.
        chunks:         Reranked chunks to use as context.
        prompt_version: Prompt config version to load (e.g. "v1").

    Returns:
        LLM-generated answer string.
    """
    config = _load_prompt_config(prompt_version)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context  = _format_context(chunks)
    # Use replace() instead of format() to avoid conflicts with
    # {company}, {filing_type} etc. in the citation instruction text
    user_msg = (
        config["qa_prompt"]
        .replace("{context}", context)
        .replace("{question}", query)
    )

    logger.info(
        f"Generating answer | model={config['model']['name']} | "
        f"chunks={len(chunks)} | prompt={prompt_version}"
    )

    try:
        response = client.chat.completions.create(
            model=config["model"]["name"],
            messages=[
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user",   "content": user_msg},
            ],
            temperature=config["model"]["temperature"],
            max_tokens=config["model"]["max_tokens"],
        )
        answer = response.choices[0].message.content.strip()
        logger.debug(f"Answer preview: '{answer[:120]}...'")
        return answer

    except Exception as exc:
        logger.error(f"Groq generation failed: {exc}")
        raise


def load_thresholds(prompt_version: str = DEFAULT_PROMPT_VERSION) -> dict[str, float]:
    """Load RAGAS evaluation thresholds from the prompt config."""
    config = _load_prompt_config(prompt_version)
    return config.get("thresholds", {})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_prompt_config(version: str) -> dict[str, Any]:
    """Load and parse a versioned YAML prompt config file."""
    config_path = PROMPTS_DIR / f"{version}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Prompt config not found: {config_path}\n"
            f"Available versions: {[f.stem for f in PROMPTS_DIR.glob('*.yaml')]}"
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded prompt config: {version}.yaml (v{config.get('version', '?')})")
    return config


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    Each chunk includes its source metadata so the LLM can cite it.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta    = chunk.get("metadata", {})
        ticker  = meta.get("ticker", "Unknown")
        filing  = meta.get("filing_type", "")
        accession = meta.get("accession_number", "")
        c_type  = chunk.get("content_type", "text")
        content = chunk.get("content", "")

        header = f"[Chunk {i} | {ticker} {filing} | {accession} | type={c_type}]"
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)
