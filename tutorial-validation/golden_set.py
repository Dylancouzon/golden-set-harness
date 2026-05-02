"""Build the labeled eval set from FiQA's native qrels.

Why use FiQA's qrels rather than synthesise queries: the qrels are
human-labeled relevance judgments, which gives us a real signal for
context_precision / context_recall. Synthesising queries would mean grading
the system against text that came from the same model — circular.

Pipeline:
  1. Load BeIR/fiqa-qrels (test split: 1,706 (query, doc, score) tuples)
  2. Load BeIR/fiqa queries (6,648 questions)
  3. Sample 300 queries deterministically (seed 42)
  4. For each, generate a 1-2 sentence reference answer using Claude,
     constrained to ONLY the labeled-relevant passages
  5. Drop any query whose reference comes back as "NO_ANSWER"
  6. Write results/golden_set.jsonl with the tutorial's entry shape:
     {"query_id", "query_text", "labels": {doc_id: score, ...}, "ground_truth"}

The reference answers are LLM-generated — that's the weakest link in the
methodology and is called out explicitly in the comparison report. context_*
metrics are scored against text that itself wasn't human-validated.

One-shot. The demo and batch eval read the cached file; we never regenerate
references on a config tweak.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import anthropic
from datasets import load_dataset
from tqdm import tqdm

from config import settings


REF_PROMPT = """You are creating a short reference answer for an evaluation set.
Use ONLY the source passages below. Answer in 1-2 sentences.
If the sources don't contain the answer, write "NO_ANSWER".

Sources:
{relevant_passages}

Question:
{query_text}
"""

OUT_PATH = Path(__file__).parent / "results" / "golden_set.jsonl"


def _load_qrels_and_queries() -> tuple[dict[str, dict[str, int]], dict[str, str], dict[str, str]]:
    """Load the three FiQA tables we need.

    Returns:
      qrels:    {query_id: {doc_id: relevance_score, ...}}  (only positive scores)
      queries:  {query_id: question text}
      corpus:   {doc_id: passage text}
    """
    qrels_ds = load_dataset("BeIR/fiqa-qrels", split="test")
    queries_ds = load_dataset("BeIR/fiqa", "queries", split="queries")
    corpus_ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row.get("score", 1))
        # Skip non-relevant labels — we only want positive examples.
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[did] = score

    queries = {str(row["_id"]): row["text"] for row in queries_ds}
    corpus = {str(row["_id"]): (row.get("text") or "").strip() for row in corpus_ds}
    return qrels, queries, corpus


def _generate_reference(client: anthropic.Anthropic, query_text: str, passages: list[str]) -> str:
    """Generate a 1-2 sentence answer constrained to the relevant passages.

    Capped at 5 passages to keep the prompt focused and the cost down.
    The prompt explicitly tells the model to write 'NO_ANSWER' when the
    passages don't cover the question — those entries get dropped.
    """
    body = REF_PROMPT.format(
        relevant_passages="\n\n".join(f"- {p}" for p in passages[:5]),
        query_text=query_text,
    )
    resp = client.messages.create(
        model=settings.generator_model,
        max_tokens=200,
        messages=[{"role": "user", "content": body}],
    )
    return resp.content[0].text.strip()


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        existing = sum(1 for _ in OUT_PATH.open())
        print(f"{OUT_PATH} already exists with {existing} entries. Delete to regenerate.")
        return 0

    print("Loading FiQA qrels + queries + corpus...")
    qrels, queries, corpus = _load_qrels_and_queries()
    print(f"  {len(qrels):,} test queries with qrels; {len(queries):,} queries; {len(corpus):,} passages.")

    qids_with_qrels = sorted(qrels.keys())
    rng = random.Random(settings.random_seed)
    rng.shuffle(qids_with_qrels)

    target = settings.eval_sample_size
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    written = 0
    skipped_no_answer = 0
    skipped_missing_text = 0
    pbar = tqdm(total=target, desc="golden-set")

    with OUT_PATH.open("w") as out:
        for qid in qids_with_qrels:
            if written >= target:
                break
            query_text = queries.get(qid)
            if not query_text:
                continue
            labels = qrels[qid]
            passages = [corpus[did] for did in labels if did in corpus and corpus[did]]
            if not passages:
                skipped_missing_text += 1
                continue
            try:
                reference = _generate_reference(client, query_text, passages)
            except Exception as exc:
                print(f"  reference generation failed for qid={qid}: {exc}", file=sys.stderr)
                continue
            if reference == "NO_ANSWER" or not reference:
                skipped_no_answer += 1
                continue
            entry = {
                "query_id": qid,
                "query_text": query_text,
                "labels": labels,
                "ground_truth": reference,
            }
            out.write(json.dumps(entry) + "\n")
            out.flush()
            written += 1
            pbar.update(1)
    pbar.close()

    print(
        f"Wrote {written} entries to {OUT_PATH}. "
        f"Skipped {skipped_no_answer} NO_ANSWER, {skipped_missing_text} missing-corpus."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
