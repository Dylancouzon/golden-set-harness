"""End-to-end one-query function: retrieve → generate → eval record."""
from __future__ import annotations

from typing import Any

from .generation import generate
from .retrieval import retrieve


def run_one(
    query_text: str,
    *,
    k_retrieve: int,
    k_rerank: int,
    hybrid: bool,
    rerank: bool,
    generator_model: str,
    prompt_template: str,
) -> dict[str, Any]:
    hits = retrieve(
        query_text,
        k_retrieve=k_retrieve,
        k_rerank=k_rerank,
        hybrid=hybrid,
        rerank=rerank,
    )
    contexts = [h["text"] for h in hits]
    answer, usage = generate(
        query_text,
        contexts,
        model=generator_model,
        prompt_template=prompt_template,
    )
    return {
        "query_text": query_text,
        "contexts": contexts,
        "answer": answer,
        "hits": hits,
        "generator_usage": usage,
    }
