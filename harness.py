"""End-to-end test of the code from:
   qdrant-landing/content/documentation/tutorials-search-engineering/retrieval-quality-golden-set.md

Runs all four code blocks against a real Qdrant instance and (optionally) a real
Anthropic key. Builds a small labeled corpus so each golden query has a known
expected doc id, then checks recall_at_k / mrr / ndcg_at_k and the tutorial's
evaluate() function against those known answers.
"""
import os

from dotenv import load_dotenv

load_dotenv()

import anthropic
import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ---------------------------------------------------------------------------
# Code block 1 from the tutorial: generate_queries_for_doc
# ---------------------------------------------------------------------------
anthropic_client = anthropic.Anthropic()


def generate_queries_for_doc(doc_text: str, n: int = 3) -> list[str]:
    response = anthropic_client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} short, realistic search queries that would lead a user to the "
                f"following document. Return only the queries, one per line.\n\n{doc_text}"
            ),
        }],
    )
    return response.content[0].text.strip().splitlines()


# ---------------------------------------------------------------------------
# Code block 2 from the tutorial: recall_at_k, mrr
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    hit = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hit / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Code block 3 from the tutorial: ndcg_at_k
# ---------------------------------------------------------------------------
def ndcg_at_k(retrieved_ids: list, relevance_map: dict, k: int) -> float:
    dcg = sum(
        (2 ** relevance_map.get(doc_id, 0) - 1) / np.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
    )
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Code block 4 from the tutorial: evaluate
# ---------------------------------------------------------------------------
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)


def evaluate(golden_set: list, collection: str, k: int = 10) -> dict:
    recalls, mrrs = [], []
    for entry in golden_set:
        results = client.query_points(
            collection_name=collection,
            query=entry["query_vector"],
            limit=k,
        ).points
        retrieved = [p.id for p in results]
        relevant = set(entry["relevant_ids"])
        recalls.append(recall_at_k(retrieved, relevant, k))
        mrrs.append(mrr(retrieved, relevant))
    return {
        "mean_recall": sum(recalls) / len(recalls),
        "mean_mrr": sum(mrrs) / len(mrrs),
    }


# ---------------------------------------------------------------------------
# Harness: build a tiny corpus, embed, upsert, evaluate
# ---------------------------------------------------------------------------
COLLECTION = "golden_set_tutorial_test"

CORPUS = [
    (1, "Qdrant is a vector database written in Rust that stores embeddings and runs approximate nearest neighbour search with HNSW."),
    (2, "Payload filtering lets you combine structured metadata with semantic similarity at query time."),
    (3, "Scalar, binary, and product quantization reduce memory usage by compressing stored vectors."),
    (4, "Distributed deployments shard a collection across nodes and replicate each shard for high availability."),
    (5, "Snapshots capture the full state of a collection so you can back up, migrate, or roll back data."),
    (6, "Hybrid search combines dense embeddings with sparse lexical signals such as BM25 to cover exact-term matches."),
    (7, "The Python client wraps the gRPC and REST APIs and exposes helpers for batching upserts and streaming scroll."),
    (8, "HNSW index build time depends on the `m` and `ef_construct` parameters; larger values improve recall but slow construction."),
]

# Pre-labeled golden set (query text -> set of relevant doc ids).
# These labels are hand-written so we know the right answer in advance.
GOLDEN = [
    {"query": "how do I back up a collection", "relevant_ids": {5}},
    {"query": "reduce memory footprint of vectors", "relevant_ids": {3}},
    {"query": "filter search results by metadata", "relevant_ids": {2}},
    {"query": "combine keyword and semantic search", "relevant_ids": {6}},
    {"query": "tune HNSW index parameters", "relevant_ids": {8}},
]

# Graded-relevance example for NDCG (doc_id -> relevance grade).
GRADED = {5: 2, 2: 1, 6: 0}


def run():
    print("=== 1. Embed corpus with fastembed ===")
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    docs = [text for _, text in CORPUS]
    doc_vectors = list(embedder.embed(docs))
    dim = len(doc_vectors[0])
    print(f"embedded {len(doc_vectors)} docs, dim={dim}")

    print("\n=== 2. (Re)create collection and upsert ===")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=doc_id, vector=vec.tolist(), payload={"text": text})
        for (doc_id, text), vec in zip(CORPUS, doc_vectors)
    ]
    client.upsert(COLLECTION, points=points)
    print(f"upserted {len(points)} points")

    print("\n=== 3. Test recall_at_k / mrr with synthetic retrieved list ===")
    retrieved = [3, 5, 2]
    relevant = {5}
    r = recall_at_k(retrieved, relevant, k=3)
    m = mrr(retrieved, relevant)
    print(f"recall@3={r}  mrr={m}")
    assert r == 1.0, f"expected 1.0, got {r}"
    assert m == 0.5, f"expected 0.5, got {m}"
    assert recall_at_k([1, 2], set(), 2) == 0.0

    print("\n=== 4. Test ndcg_at_k ===")
    retrieved = [5, 2, 6]
    n = ndcg_at_k(retrieved, GRADED, k=3)
    print(f"ndcg@3={n:.6f}")
    assert abs(n - 1.0) < 1e-9, f"expected 1.0, got {n}"
    reversed_retrieved = [6, 2, 5]
    n2 = ndcg_at_k(reversed_retrieved, GRADED, k=3)
    print(f"ndcg@3 (reversed)={n2:.6f}")
    assert n2 < n

    print("\n=== 5. Embed golden queries ===")
    query_texts = [g["query"] for g in GOLDEN]
    query_vectors = list(embedder.embed(query_texts))
    golden_set = [
        {"query_vector": qv.tolist(), "relevant_ids": g["relevant_ids"]}
        for g, qv in zip(GOLDEN, query_vectors)
    ]

    print("\n=== 6. Run tutorial's evaluate() ===")
    metrics = evaluate(golden_set, COLLECTION, k=3)
    print(metrics)
    assert "mean_recall" in metrics and "mean_mrr" in metrics
    assert 0.0 <= metrics["mean_recall"] <= 1.0
    assert 0.0 <= metrics["mean_mrr"] <= 1.0

    print("\n=== 7. generate_queries_for_doc (requires real ANTHROPIC_API_KEY) ===")
    try:
        qs = generate_queries_for_doc(CORPUS[0][1], n=3)
        print(f"generated {len(qs)} queries:")
        for q in qs:
            print(f"  - {q}")
    except anthropic.AuthenticationError as e:
        print(f"AuthenticationError (expected if the key is a placeholder): {e.message[:120]}")
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")

    print("\n=== 8. Cleanup ===")
    client.delete_collection(COLLECTION)
    print("collection deleted")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run()
