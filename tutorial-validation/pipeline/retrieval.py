"""Hybrid retrieve + cross-encoder rerank, with toggles.

Single entry point: `retrieve(query, k_retrieve, k_rerank, *, hybrid, rerank)`.
Returns a list of dicts: [{"doc_id", "text", "score"}, ...].

Pipeline shape (when both toggles are on):
    1. Embed query (dense + BM25 sparse)
    2. Qdrant query_points with two prefetches + RRF fusion → k_retrieve hits
    3. Cross-encoder rerank → top k_rerank
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

from config import settings


# --- Lazy singletons ---------------------------------------------------------
# Qdrant + OpenAI clients are cheap to construct but we want exactly one of
# each so connection pools are reused across calls. lru_cache is the simplest
# way to memoise without adding a class.

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
    """fastembed BM25 — downloads ~5 MB on first use, then cached on disk."""
    return SparseTextEmbedding(model_name=settings.sparse_model)


# Cross-encoder is ~120 MB and slow to load. Only construct it when reranking
# is actually requested — the demo's "rerank toggle off" path skips this
# entirely. Module-level None + lazy assignment is simpler than lru_cache here
# because we want to allow the toggle path to never touch it.
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


# --- Embedding helpers (used by ingest.py too) -------------------------------

def embed_dense(text: str) -> list[float]:
    """OpenAI embedding for a single query string."""
    resp = _openai().embeddings.create(model=settings.dense_model, input=[text])
    return resp.data[0].embedding


def embed_sparse(text: str):
    """BM25 sparse vector. fastembed.embed yields one item per input."""
    return next(_sparse_model().embed([text]))


# --- Main entry point --------------------------------------------------------

def retrieve(
    query: str,
    k_retrieve: int = 50,
    k_rerank: int = 10,
    *,
    hybrid: bool = True,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve k_rerank passages for `query`.

    With `hybrid=True`, runs dense + sparse prefetches and fuses with RRF.
    With `hybrid=False`, runs dense-only.
    With `rerank=True`, applies the cross-encoder to the prefetch result.
    """
    dense_vec = embed_dense(query)

    # Always include a dense prefetch. Sparse is optional.
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

    # The hybrid branch uses FusionQuery(RRF) and lets Qdrant fuse the two
    # prefetches. The dense-only branch skips prefetch entirely and queries
    # the named "dense" vector directly — that's faster and matches what the
    # tutorial shows.
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
        # Cross-encoder scores the (query, passage) pair directly. We sort by
        # the new score, replacing Qdrant's similarity score in the output.
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
