"""Narrow re-test of only the generate_queries_for_doc block from the tutorial."""
import os

from dotenv import load_dotenv

load_dotenv()

import anthropic

client = anthropic.Anthropic()


def generate_queries_for_doc(doc_text: str, n: int = 3) -> list[str]:
    response = client.messages.create(
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


DOCS = [
    "Qdrant is a vector database written in Rust that stores embeddings and runs approximate nearest neighbour search with HNSW.",
    "Scalar, binary, and product quantization reduce memory usage by compressing stored vectors.",
    "Hybrid search combines dense embeddings with sparse lexical signals such as BM25 to cover exact-term matches.",
]

for i, doc in enumerate(DOCS, 1):
    print(f"--- doc {i} ---")
    print(f"doc: {doc}")
    qs = generate_queries_for_doc(doc, n=3)
    print(f"returned {len(qs)} lines:")
    for q in qs:
        print(f"  {q!r}")
    print()
