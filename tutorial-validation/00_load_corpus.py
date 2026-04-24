"""
Load a public corpus, embed it with FastEmbed, and upload to Qdrant.

Creates a Qdrant collection named `tutorial_validation_wiki` with:
  - FastEmbed BAAI/bge-small-en-v1.5 vectors (384 dims, cosine)
  - Integer point IDs (0..N-1)
  - "text" field in payload containing the paragraph

The raw corpus is also saved to artifacts/corpus.json so downstream
scripts can reference documents by ID without re-loading the dataset.
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
VECTOR_SIZE = 384
N_DOCS = 500

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def load_corpus(n: int = N_DOCS) -> list[dict]:
    """Load N unique paragraph contexts from SQuAD v1 validation split."""
    ds = load_dataset("rajpurkar/squad", split="validation")
    seen = set()
    docs = []
    for item in ds:
        ctx = item["context"].strip()
        if ctx in seen:
            continue
        seen.add(ctx)
        docs.append({"doc_id": len(docs), "text": ctx, "title": item["title"]})
        if len(docs) >= n:
            break
    return docs


def main() -> None:
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Load corpus
    print(f"Loading corpus ({N_DOCS} docs from SQuAD v1 validation)...")
    corpus = load_corpus()
    print(f"  got {len(corpus)} unique paragraphs")
    (ARTIFACTS / "corpus.json").write_text(json.dumps(corpus, indent=2))
    print(f"  saved to artifacts/corpus.json")

    # Embed
    print(f"\nEmbedding with {MODEL_NAME}...")
    embedder = TextEmbedding(model_name=MODEL_NAME)
    vectors = list(embedder.embed([d["text"] for d in corpus]))
    print(f"  got {len(vectors)} vectors, dim {len(vectors[0])}")

    # (Re)create collection
    print(f"\nPreparing collection '{COLLECTION}'...")
    if client.collection_exists(COLLECTION):
        print(f"  collection exists; deleting to get a fresh index")
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"  created collection (cosine, {VECTOR_SIZE} dims)")

    # Upload
    print("\nUploading points...")
    points = [
        models.PointStruct(
            id=doc["doc_id"],
            vector=vec.tolist(),
            payload={"text": doc["text"], "title": doc["title"]},
        )
        for doc, vec in zip(corpus, vectors)
    ]
    client.upload_points(collection_name=COLLECTION, points=points, wait=True)
    info = client.get_collection(COLLECTION)
    print(f"  uploaded; collection now reports {info.points_count} points")

    print("\nDone.")


if __name__ == "__main__":
    main()
