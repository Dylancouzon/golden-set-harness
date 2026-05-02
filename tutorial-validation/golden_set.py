"""Build the labeled eval set from FiQA's qrels.

Reads BeIR/fiqa qrels and queries from HuggingFace, samples 300 queries
deterministically (seed 42), generates a 1-2 sentence reference answer per query
constrained to the relevant passages, and caches the result to
results/golden_set.jsonl.

Entries follow the tutorial shape:
    {"query_id": str, "query_text": str,
     "labels": {doc_id: relevance, ...}, "ground_truth": str}

This is a one-shot batch script — the demo always reads from the cached
golden_set.jsonl and never regenerates references on the fly.
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
    """Return (qrels_by_query, query_text_by_id, corpus_text_by_id)."""
    qrels_ds = load_dataset("BeIR/fiqa-qrels", split="test")
    queries_ds = load_dataset("BeIR/fiqa", "queries", split="queries")
    corpus_ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row.get("score", 1))
        if score <= 0:
            continue
        qrels.setdefault(qid, {})[did] = score

    queries = {str(row["_id"]): row["text"] for row in queries_ds}
    corpus = {str(row["_id"]): (row.get("text") or "").strip() for row in corpus_ds}
    return qrels, queries, corpus


def _generate_reference(client: anthropic.Anthropic, query_text: str, passages: list[str]) -> str:
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
