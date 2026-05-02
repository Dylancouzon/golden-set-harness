"""End-to-end one-query function: retrieve → generate → eval record.

This is the integration point between the retrieval and generation modules.
Returns the dict shape that scoring.py / Streamlit / batch eval all consume.
"""
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
    hnsw_ef: int | None = None,
) -> dict[str, Any]:
    """Run one (query → retrieve → generate) cycle.

    Output keys:
      - query_text:      the original question
      - contexts:        list[str] — passage texts only (what the prompt sees)
      - answer:          str — the generator's response
      - hits:            list[dict] — full retrieval hits (text, score, doc_id)
      - generator_usage: token counts for the cost meter
    """
    hits = retrieve(
        query_text,
        k_retrieve=k_retrieve,
        k_rerank=k_rerank,
        hybrid=hybrid,
        rerank=rerank,
        hnsw_ef=hnsw_ef,
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
