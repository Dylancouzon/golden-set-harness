"""
Test Tutorial 1: Measuring ANN Precision.

Exercises the `avg_precision_at_k` helper from the Automate-in-CI-with-Python
section. Compares ANN and exact-kNN top-k over a batch of realistic query
vectors (SQuAD questions, embedded with the same FastEmbed model the corpus
was indexed with).

Writes the precision@k for k=10 into artifacts/tutorial_1_results.json.
"""
import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

COLLECTION = "tutorial_validation_wiki"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
N_TEST_QUERIES = 50
K = 10

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


# ---------- Tutorial 1's helper, copied verbatim ----------
def avg_precision_at_k(
    client: QdrantClient,
    collection_name: str,
    test_vectors: list,
    k: int,
) -> float:
    precisions = []
    for vector in test_vectors:
        ann_ids = {
            p.id for p in client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=k,
            ).points
        }
        knn_ids = {
            p.id for p in client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=k,
                search_params=models.SearchParams(exact=True),
            ).points
        }
        precisions.append(len(ann_ids & knn_ids) / k)

    return sum(precisions) / len(precisions)


def main() -> None:
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Collect N distinct questions from SQuAD as test queries.
    print(f"Building test set of {N_TEST_QUERIES} query vectors...")
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen = set()
    question_texts: list[str] = []
    for item in ds:
        q = item["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        question_texts.append(q)
        if len(question_texts) >= N_TEST_QUERIES:
            break

    embedder = TextEmbedding(model_name=MODEL_NAME)
    test_vectors = [v.tolist() for v in embedder.embed(question_texts)]
    print(f"  embedded {len(test_vectors)} questions")

    # Run the helper.
    print(f"\nRunning avg_precision_at_k for k={K}...")
    avg_prec = avg_precision_at_k(
        client=client,
        collection_name=COLLECTION,
        test_vectors=test_vectors,
        k=K,
    )
    print(f"  avg precision@{K} = {avg_prec:.4f}")

    results = {
        "collection": COLLECTION,
        "k": K,
        "n_queries": len(test_vectors),
        "avg_precision_at_k": round(avg_prec, 4),
        "interpretation": (
            "With the default HNSW config on 500 points (small index), "
            "ANN usually matches exact kNN closely. Scores near 1.0 are "
            "expected. The helper ran end-to-end without errors."
        ),
        "sample_question_texts": question_texts[:5],
    }
    out_path = ARTIFACTS / "tutorial_1_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  results written to {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
