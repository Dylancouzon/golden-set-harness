"""Hybrid retrieve + cross-encoder rerank, with toggles.

Single entry point: `retrieve(query, k_retrieve, k_rerank, *, hybrid, rerank)`.
Returns a list of dicts: [{"doc_id", "text", "score"}, ...].
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

from config import settings


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )


@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def _sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.sparse_model)


_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder (~120MB) so demo doesn't pay startup when rerank is off."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def embed_dense(text: str) -> list[float]:
    resp = _openai().embeddings.create(model=settings.dense_model, input=[text])
    return resp.data[0].embedding


def embed_sparse(text: str):
    return next(_sparse_model().embed([text]))


def retrieve(
    query: str,
    k_retrieve: int = 50,
    k_rerank: int = 10,
    *,
    hybrid: bool = True,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    dense_vec = embed_dense(query)
    prefetch = [models.Prefetch(query=dense_vec, using="dense", limit=k_retrieve)]

    if hybrid:
        sparse_vec = embed_sparse(query)
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=k_retrieve,
            )
        )

    if hybrid:
        results = _qdrant().query_points(
            collection_name=settings.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k_retrieve,
            with_payload=True,
        ).points
    else:
        results = _qdrant().query_points(
            collection_name=settings.collection,
            query=dense_vec,
            using="dense",
            limit=k_retrieve,
            with_payload=True,
        ).points

    if rerank and results:
        pairs = [(query, p.payload["text"]) for p in results]
        scores = _get_reranker().predict(pairs)
        results = [
            p
            for p, _ in sorted(
                zip(results, scores), key=lambda x: x[1], reverse=True
            )
        ]

    top = results[:k_rerank]
    return [
        {
            "doc_id": p.payload.get("doc_id", str(p.id)),
            "text": p.payload["text"],
            "score": float(p.score) if p.score is not None else 0.0,
        }
        for p in top
    ]
