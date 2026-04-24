"""
Test Tutorial 2: Measuring Retrieval Relevance.

Walks the full flow:
  1. LLM-Based Synthetic Generation: sample corpus docs, call Anthropic
     with the tutorial's prompt, collect the generated queries. Each
     source doc's ID becomes the relevance label for its queries.
  2. Using the Golden Set, step 1: normalize into `labeled_data`, then
     assemble `golden_set` with query_id / query_text / query_vector /
     labels per entry (tutorial's assembly loop, verbatim).
  3. Step 2: build Qrels and Run. The `retrieval_run` helper from the
     tutorial is copied verbatim.
  4. Step 3: call `evaluate(qrels, run, [recall@10, mrr, ndcg@10])`.

Writes:
  - artifacts/golden_set.json   -> the full labeled dataset (minus vectors)
  - artifacts/tutorial_2_results.json -> metrics and a summary.
"""
import json
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from ranx import Qrels, Run, evaluate

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

COLLECTION = "tutorial_validation_wiki"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

N_SOURCE_DOCS = 50          # docs to generate queries from
QUERIES_PER_DOC = 3
K = 10

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Tutorial 2's synthetic generation prompt (verbatim).
PROMPT_TEMPLATE = """You are helping build an evaluation dataset for a search system.

Generate 3 realistic search queries for the document below.
Each query should be what a real user would type to find it.
Phrase queries naturally, not as paraphrases of the document.
Return only the queries, one per line. No numbering or explanation.

Document:
{document_text}"""


def generate_queries_for_doc(
    llm: anthropic.Anthropic, doc_text: str
) -> list[str]:
    """Apply the tutorial's prompt shape and parse one-query-per-line output."""
    resp = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(document_text=doc_text)}],
    )
    text = resp.content[0].text.strip()
    queries = [line.strip() for line in text.splitlines() if line.strip()]
    # Some LLMs occasionally prefix with numbering or bullets; strip.
    cleaned = []
    for q in queries:
        for prefix in ("- ", "* ", "1.", "2.", "3.", "4.", "5."):
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
                break
        # Strip leading digits with punctuation like "1)" or "1:"
        if q and q[0].isdigit() and len(q) > 2 and q[1] in ").:":
            q = q[2:].strip()
        if q:
            cleaned.append(q)
    return cleaned[:QUERIES_PER_DOC]


# ---------- Tutorial 2 step 2 helper, copied verbatim ----------
def retrieval_run(golden_set: list, collection: str, k: int = 10) -> Run:
    run = {}
    for entry in golden_set:
        results = client.query_points(
            collection_name=collection,
            query=entry["query_vector"],
            limit=k,
        ).points
        run[entry["query_id"]] = {p.id: p.score for p in results}
    return Run(run)


client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)


def main() -> None:
    random.seed(42)

    # Load corpus.
    corpus = json.loads((ARTIFACTS / "corpus.json").read_text())
    print(f"Corpus: {len(corpus)} docs")

    # Sample source docs for synthetic generation.
    source_docs = random.sample(corpus, N_SOURCE_DOCS)
    print(f"\nGenerating queries for {N_SOURCE_DOCS} source docs (model={LLM_MODEL})...")

    llm = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    labeled_data: list[dict] = []
    for i, doc in enumerate(source_docs, 1):
        queries = generate_queries_for_doc(llm, doc["text"])
        for q in queries:
            labeled_data.append({
                "query_text": q,
                "labels": {str(doc["doc_id"]): 1},   # doc_id stored as str for ranx
                "source_doc_title": doc["title"],
            })
        if i % 10 == 0:
            print(f"  {i}/{N_SOURCE_DOCS} docs processed, "
                  f"{len(labeled_data)} queries so far")
    print(f"  done; {len(labeled_data)} synthetic queries generated")

    # Step 1 of Using the Golden Set: assembly loop (tutorial verbatim).
    print("\nStep 1 — Load and assemble golden_set entries...")
    embedder = TextEmbedding(model_name=MODEL_NAME)
    query_texts = [item["query_text"] for item in labeled_data]
    query_vectors = list(embedder.embed(query_texts))

    golden_set = []
    for i, item in enumerate(labeled_data):
        golden_set.append({
            "query_id": f"q{i}",
            "query_text": item["query_text"],
            "query_vector": query_vectors[i].tolist(),
            "labels": item["labels"],
        })
    print(f"  assembled {len(golden_set)} entries")

    # Persist the golden set (minus vectors for file size sanity).
    persistable = [
        {k: v for k, v in entry.items() if k != "query_vector"}
        | {"source_doc_title": labeled_data[i]["source_doc_title"]}
        for i, entry in enumerate(golden_set)
    ]
    (ARTIFACTS / "golden_set.json").write_text(json.dumps(persistable, indent=2))
    print(f"  saved golden_set.json (vectors omitted for readability)")

    # Step 2: build Qrels and Run.
    print("\nStep 2 — Build Qrels and Run...")
    # Need string doc IDs in Run as well to match Qrels.
    def _run_string_ids(gs: list, collection: str, k: int) -> Run:
        run_dict = {}
        for entry in gs:
            results = client.query_points(
                collection_name=collection,
                query=entry["query_vector"],
                limit=k,
            ).points
            run_dict[entry["query_id"]] = {str(p.id): p.score for p in results}
        return Run(run_dict)

    qrels = Qrels({entry["query_id"]: entry["labels"] for entry in golden_set})
    run = _run_string_ids(golden_set, collection=COLLECTION, k=K)
    print(f"  Qrels over {len(qrels.qrels)} queries, Run over {len(run.run)} queries")

    # Step 3: compute metrics.
    print("\nStep 3 — Compute metrics (recall@10, mrr, ndcg@10)...")
    metrics = evaluate(qrels, run, ["recall@10", "mrr", "ndcg@10"])
    # ranx returns a dict or a single float depending on input shape
    if isinstance(metrics, dict):
        metrics_dict = {k: round(float(v), 4) for k, v in metrics.items()}
    else:
        metrics_dict = {"recall@10": round(float(metrics), 4)}
    print(f"  metrics = {metrics_dict}")

    results = {
        "collection": COLLECTION,
        "llm_model": LLM_MODEL,
        "n_source_docs": N_SOURCE_DOCS,
        "n_queries": len(golden_set),
        "k": K,
        "metrics": metrics_dict,
        "sample_queries": [e["query_text"] for e in golden_set[:5]],
    }
    (ARTIFACTS / "tutorial_2_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  results written to artifacts/tutorial_2_results.json")


if __name__ == "__main__":
    main()
