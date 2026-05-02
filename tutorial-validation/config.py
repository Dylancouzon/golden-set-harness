import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = field(default_factory=lambda: os.environ["QDRANT_URL"])
    qdrant_api_key: str = field(default_factory=lambda: os.environ["QDRANT_API_KEY"])
    anthropic_api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])

    collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "fiqa_eval"))
    dense_model: str = field(default_factory=lambda: os.getenv("DENSE_MODEL", "text-embedding-3-small"))
    dense_dim: int = field(default_factory=lambda: int(os.getenv("DENSE_DIM", "1536")))
    sparse_model: str = field(default_factory=lambda: os.getenv("SPARSE_MODEL", "Qdrant/bm25"))
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"))

    generator_model: str = field(default_factory=lambda: os.getenv("GENERATOR_MODEL", "claude-sonnet-4-6"))
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-5.4"))

    eval_sample_size: int = field(default_factory=lambda: int(os.getenv("EVAL_SAMPLE_SIZE", "300")))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))
    top_k_retrieve: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVE", "50")))
    top_k_rerank: int = field(default_factory=lambda: int(os.getenv("TOP_K_RERANK", "10")))


# Available models for live demo dropdowns
GENERATOR_CHOICES = ["claude-sonnet-4-6", "claude-opus-4-7", "gpt-5.4"]
JUDGE_CHOICES = ["gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-7"]

# Fallback judge model used when the configured judge_model 404s. See
# results/tutorial_gaps.md — gpt-5.4 is the value the tutorial pins, but it
# may not exist at OpenAI today; the demo drops to gpt-4o.
JUDGE_FALLBACK = "gpt-4o"

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
    """Coarse family for the family-collision warning."""
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gpt") or model_name.startswith("o"):
        return "openai"
    return "unknown"


settings = Settings()
