"""Ingest BeIR/fiqa corpus into Qdrant with named dense + sparse vectors.

This is a one-shot batch script. The demo loads a warm collection and never
re-ingests on a tweak.

Pipeline:
  1. Stream BeIR/fiqa corpus from HuggingFace (~57.6k finance Q&A passages)
  2. Create the Qdrant collection with two named vectors:
       - dense  → 1536-dim cosine (text-embedding-3-small)
       - sparse → BM25 with IDF modifier (Qdrant/bm25 via fastembed)
  3. Embed in batches of 256 (dense via OpenAI, sparse via fastembed)
  4. Upsert each batch with wait=False for throughput

Idempotent: if the collection already has >50k points, it skips the run. To
re-ingest, delete the collection first.

Cost: ~$1 in OpenAI embedding charges, ~15-25 minutes wall-clock.
"""
from __future__ import annotations

import sys
import time
import uuid
from typing import Iterator

from datasets import load_dataset
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from config import settings


# Batch size controls both how often we embed and how often we upsert.
# 256 is a sweet spot — large enough to amortise per-call overhead, small
# enough that a single retryable failure doesn't waste much progress.
BATCH = 256

# Used to deterministically hash any non-integer IDs into UUIDs. Qdrant
# requires point IDs to be either ints or UUIDs, and FiQA's _id is usually
# integer-stringy but we guard against the rare exception.
COLLECTION_NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _to_point_id(raw_id: str) -> str | int:
    """Map FiQA's _id to a Qdrant-acceptable point ID."""
    try:
        return int(raw_id)
    except ValueError:
        return str(uuid.uuid5(COLLECTION_NAMESPACE_UUID, raw_id))


def _ensure_collection(client: QdrantClient) -> bool:
    """Create the collection if missing. Return True if it was just created.

    The sparse vector config uses Modifier.IDF — BM25 needs IDF
    aggregation across the corpus to weight rare-terms correctly, and
    Qdrant won't compute it unless we ask.
    """
    if client.collection_exists(settings.collection):
        return False
    client.create_collection(
        collection_name=settings.collection,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.dense_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    return True


def _iter_corpus() -> Iterator[dict]:
    """Stream FiQA passages from HuggingFace, skipping empty rows.

    FiQA passages are already short (avg ~85 tokens) — we don't chunk
    further. Each row carries _id, title, text.
    """
    ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        yield {
            "doc_id": row["_id"],
            "text": text,
            "title": (row.get("title") or "").strip(),
        }


def _embed_dense_batch(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """One OpenAI embedding call per BATCH passages."""
    resp = openai_client.embeddings.create(model=settings.dense_model, input=texts)
    return [d.embedding for d in resp.data]


def main() -> int:
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,
    )
    openai_client = OpenAI(api_key=settings.openai_api_key)
    sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)

    created = _ensure_collection(qdrant)
    if not created:
        existing = qdrant.count(settings.collection, exact=True).count
        if existing > 50_000:
            print(
                f"Collection '{settings.collection}' already populated ({existing:,} points). "
                "Skipping ingest. Delete the collection to re-ingest."
            )
            return 0
        print(
            f"Collection '{settings.collection}' exists but only has {existing:,} points; "
            "continuing ingest."
        )

    print(f"Loading BeIR/fiqa corpus...")
    corpus = list(_iter_corpus())
    print(f"Loaded {len(corpus):,} passages.")

    upserted = 0
    t0 = time.time()
    pbar = tqdm(total=len(corpus), desc="ingest", unit="doc")
    for i in range(0, len(corpus), BATCH):
        batch = corpus[i : i + BATCH]
        texts = [r["text"] for r in batch]

        # Embeddings are computed serially per batch but the two providers
        # (OpenAI, fastembed) are independent. fastembed is local and ~free;
        # OpenAI dominates the wall-clock here.
        dense_vecs = _embed_dense_batch(openai_client, texts)
        sparse_vecs = list(sparse_model.embed(texts))

        points = []
        for row, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=_to_point_id(row["doc_id"]),
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "text": row["text"],
                        "title": row["title"],
                        "doc_id": row["doc_id"],
                    },
                )
            )
        # wait=False fires-and-forgets the upsert so we can keep embedding
        # the next batch while Qdrant indexes the previous one.
        qdrant.upsert(
            collection_name=settings.collection,
            points=points,
            wait=False,
        )
        upserted += len(points)
        pbar.update(len(points))
    pbar.close()

    # Final wait — block until previous wait=False writes have flushed so the
    # reported count is accurate. We can't pass an empty points list (Qdrant
    # rejects it as a bad request); a count() call forces a sync round-trip
    # to the same collection, which is enough.
    _ = qdrant.count(collection_name=settings.collection, exact=True).count

    final = qdrant.count(settings.collection, exact=True).count
    elapsed = time.time() - t0
    print(f"Done. Upserted {upserted:,} points; collection now reports {final:,}. "
          f"Wall: {elapsed/60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
