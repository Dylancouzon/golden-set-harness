"""Project-wide configuration loaded from .env (or process environment).

Everything else in the project imports `settings` from here. Keep this file the
single source of truth so swapping models, k values, or sample sizes is a
one-line change.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Loads .env from CWD or any parent. Safe to call repeatedly.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Frozen container for env-driven config.

    `field(default_factory=lambda: ...)` is used (instead of plain default
    values) because frozen dataclass field defaults are evaluated at *class
    definition time* — that would require the env vars to be set at import
    time on the very first import, which is brittle. Lazy default_factory
    delays the read until `Settings()` is instantiated.
    """
    # --- Required: hard-fail if missing ---
    qdrant_url: str = field(default_factory=lambda: os.environ["QDRANT_URL"])
    qdrant_api_key: str = field(default_factory=lambda: os.environ["QDRANT_API_KEY"])
    anthropic_api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])

    # --- Optional with sensible defaults ---
    collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "fiqa_eval"))
    dense_model: str = field(default_factory=lambda: os.getenv("DENSE_MODEL", "text-embedding-3-small"))
    dense_dim: int = field(default_factory=lambda: int(os.getenv("DENSE_DIM", "1536")))
    sparse_model: str = field(default_factory=lambda: os.getenv("SPARSE_MODEL", "Qdrant/bm25"))
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"))

    # Generator vs judge default to *different* model families on purpose —
    # self-judging biases the score upward (the judge is generous to its own
    # voice). The Streamlit sidebar surfaces a warning when the picked pair
    # collides on family.
    generator_model: str = field(default_factory=lambda: os.getenv("GENERATOR_MODEL", "claude-sonnet-4-6"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-5.4"))

    eval_sample_size: int = field(default_factory=lambda: int(os.getenv("EVAL_SAMPLE_SIZE", "300")))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))
    top_k_retrieve: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVE", "50")))
    top_k_rerank: int = field(default_factory=lambda: int(os.getenv("TOP_K_RERANK", "10")))


# Sidebar dropdowns. The same model can appear in both lists; the family-
# collision warning lives in app.py.
GENERATOR_CHOICES = ["claude-sonnet-4-6", "claude-opus-4-7", "gpt-5.4"]
JUDGE_CHOICES = ["gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-7"]

# If the configured judge_model 404s at the provider, scoring.py falls back to
# this. gpt-4o is widely available and a common default. See
# results/tutorial_gaps.md for context.
JUDGE_FALLBACK = "gpt-4o"

# The grounding prompt — verbatim from the tutorial. Critical that this stays
# stable across the demo because it's part of the evaluation surface (changing
# the prompt changes the answer, which changes the scores).
DEFAULT_PROMPT_TEMPLATE = """You are answering questions using retrieved source material.

Answer the question below using only the provided context.
If the context does not contain the answer, say so explicitly.
Do not rely on outside knowledge.

Context:
{retrieved_context}

Question:
{query_text}
"""


def model_family(model_name: str) -> str:
    """Coarse family bucket — 'anthropic', 'openai', or 'unknown'.

    Used by the family-collision warning in the Streamlit sidebar to flag
    when the generator and judge are both Anthropic (or both OpenAI),
    which leaks self-evaluation bias into the scores.
    """
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gpt") or model_name.startswith("o"):
        return "openai"
    return "unknown"


# Singleton — every other module imports this rather than re-reading env.
settings = Settings()
