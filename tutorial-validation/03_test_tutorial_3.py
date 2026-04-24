"""
Test Tutorial 3: Evaluating Pipeline Output Quality.

Walks the full flow from the tutorial:
  1. Extend the golden set with query_text + ground_truth.
  2. Run retrieval + generation to build `SingleTurnSample` records.
  3. Score with Ragas (faithfulness, answer_relevancy, context_precision).
  4. Drill into per-query scores via scores.to_pandas().

Gaps vs the tutorial (noted in the README):
  - Ragas defaults to OpenAI for its judge LLM. We have Anthropic only, so
    we wire an Anthropic judge + FastEmbed embeddings explicitly. The
    tutorial's verbatim `evaluate(dataset, metrics=[...])` call would
    fail without `OPENAI_API_KEY` set.

Subsamples to 30 entries for Tutorial 3 to keep judge-call cost contained
(Ragas makes multiple judge calls per sample per metric; 30 × 3 × ~3 ~=
270 LLM calls).
"""
import json
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# Ragas config imports (for wiring an Anthropic judge).
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Tutorial 3's imports (verbatim).
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

COLLECTION = "tutorial_validation_wiki"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
N_SAMPLES = 30
K = 10

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Tutorial 3's prompt template (verbatim).
PROMPT_TEMPLATE = """You are answering questions using retrieved source material.

Answer the question below using only the provided context.
If the context does not contain the answer, say so explicitly.
Do not rely on outside knowledge.

Context:
{retrieved_context}

Question:
{query_text}
"""

# ---------- Clients (tutorial verbatim) ----------
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def generate_answer(query_text: str, contexts: list) -> str:
    """Tutorial 3's generate_answer, verbatim."""
    prompt = PROMPT_TEMPLATE.format(
        retrieved_context="\n\n".join(contexts),
        query_text=query_text,
    )
    response = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def build_eval_set(golden_set: list, collection: str, k: int = 10) -> list:
    """Tutorial 3's build_eval_set, verbatim."""
    samples = []
    for entry in golden_set:
        results = client.query_points(
            collection_name=collection,
            query=entry["query_vector"],
            limit=k,
        ).points
        contexts = [p.payload["text"] for p in results]
        answer = generate_answer(entry["query_text"], contexts)
        samples.append(SingleTurnSample(
            user_input=entry["query_text"],
            retrieved_contexts=contexts,
            response=answer,
            reference=entry.get("ground_truth", ""),
        ))
    return samples


def synthesize_ground_truth(query_text: str, source_text: str) -> str:
    """For each synthetic query, ask Anthropic for a short reference answer
    using only the source document. Gives context_precision a reference."""
    resp = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "Using only the source text below, answer the question in one "
                "or two concise sentences. If the source does not contain the "
                "answer, respond: 'Not answerable from source.'\n\n"
                f"Source:\n{source_text}\n\nQuestion: {query_text}"
            ),
        }],
    )
    return resp.content[0].text.strip()


def main() -> None:
    random.seed(42)

    # Load the golden set assembled by tutorial 2.
    gs_raw = json.loads((ARTIFACTS / "golden_set.json").read_text())
    corpus = json.loads((ARTIFACTS / "corpus.json").read_text())
    corpus_by_id = {str(d["doc_id"]): d for d in corpus}

    # Subsample.
    sampled = random.sample(gs_raw, N_SAMPLES)
    print(f"Sampled {N_SAMPLES} queries from the golden set of {len(gs_raw)}")

    # Re-embed query_text (vectors weren't persisted) and synthesize ground_truth.
    print(f"\nExtending golden_set with query_vector + ground_truth...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vectors = list(embedder.embed([e["query_text"] for e in sampled]))

    golden_set = []
    for i, (entry, vec) in enumerate(zip(sampled, vectors), 1):
        # The only label doc is the source; look up its text.
        source_doc_id = next(iter(entry["labels"]))
        source_text = corpus_by_id[source_doc_id]["text"]
        ground_truth = synthesize_ground_truth(entry["query_text"], source_text)

        golden_set.append({
            "query_id": entry["query_id"],
            "query_text": entry["query_text"],
            "query_vector": vec.tolist(),
            "labels": entry["labels"],
            "ground_truth": ground_truth,
        })
        if i % 10 == 0:
            print(f"  {i}/{N_SAMPLES} entries extended")

    # Run retrieval + generation.
    print(f"\nRunning retrieve-generate-record loop (k={K})...")
    samples = build_eval_set(golden_set, collection=COLLECTION, k=K)
    print(f"  built {len(samples)} SingleTurnSample records")

    # Save samples (raw) so they can be inspected.
    samples_dump = [{
        "user_input": s.user_input,
        "retrieved_context_preview": [c[:120] + "..." for c in s.retrieved_contexts[:3]],
        "response": s.response,
        "reference": s.reference,
    } for s in samples]
    (ARTIFACTS / "tutorial_3_samples.json").write_text(json.dumps(samples_dump, indent=2))
    print(f"  saved samples to artifacts/tutorial_3_samples.json")

    # Wire an Anthropic judge + FastEmbed embeddings into Ragas.
    # (The tutorial's code doesn't show this; see README for the gap.)
    print("\nConfiguring Ragas with Anthropic judge + FastEmbed embeddings...")
    llm = LangchainLLMWrapper(ChatAnthropic(
        model=LLM_MODEL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=2048,  # avoids "LLM did not finish" on longer judge outputs
    ))
    embed_wrapper = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Tutorial 3's scoring code, with llm/embeddings kwargs added.
    dataset = EvaluationDataset(samples=samples)
    print(f"\nScoring with Ragas (faithfulness, answer_relevancy, context_precision)...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embed_wrapper,
    )
    print(f"\nAggregate scores:\n  {scores}")

    # Per-query drill.
    print("\nPer-query drill (scores.to_pandas().nsmallest(5, 'faithfulness')):")
    per_query = scores.to_pandas()
    worst = per_query.nsmallest(5, "faithfulness")
    print(worst[["user_input", "faithfulness", "answer_relevancy", "context_precision"]].to_string())

    # Save aggregate + per-query results.
    aggregate = {k: round(float(v), 4) for k, v in scores._repr_dict.items()} \
        if hasattr(scores, "_repr_dict") else {
            k: round(float(v), 4) for k, v in scores.items()
        }

    results = {
        "collection": COLLECTION,
        "llm_model": LLM_MODEL,
        "n_samples": len(samples),
        "k": K,
        "aggregate_scores": aggregate,
        "per_query_preview": per_query.head(10).to_dict(orient="records"),
    }
    (ARTIFACTS / "tutorial_3_results.json").write_text(json.dumps(results, indent=2, default=str))
    per_query.to_csv(ARTIFACTS / "tutorial_3_per_query.csv", index=False)
    print(f"\n  results written to artifacts/tutorial_3_results.json")
    print(f"  per-query scores written to artifacts/tutorial_3_per_query.csv")


if __name__ == "__main__":
    main()
