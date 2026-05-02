"""Ingest BeIR/fiqa corpus into Qdrant with named dense + sparse vectors.

One-shot batch script. Idempotent: skips ingest if a fully-populated collection
already exists. Run with: python ingest.py
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


BATCH = 256
COLLECTION_NAMESPACE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _to_point_id(raw_id: str) -> str | int:
    """FiQA corpus uses integer string IDs. Convert when possible, else hash to UUID5."""
    try:
        return int(raw_id)
    except ValueError:
        return str(uuid.uuid5(COLLECTION_NAMESPACE_UUID, raw_id))


def _ensure_collection(client: QdrantClient) -> bool:
    """Create the collection if missing. Return True if newly created."""
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
    """Stream the FiQA corpus from HuggingFace."""
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
        qdrant.upsert(
            collection_name=settings.collection,
            points=points,
            wait=False,
        )
        upserted += len(points)
        pbar.update(len(points))
    pbar.close()

    # Final flush
    qdrant.upsert(collection_name=settings.collection, points=[], wait=True)

    final = qdrant.count(settings.collection, exact=True).count
    elapsed = time.time() - t0
    print(f"Done. Upserted {upserted:,} points; collection now reports {final:,}. "
          f"Wall: {elapsed/60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
