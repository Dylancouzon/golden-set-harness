# golden-set-harness

Internal harness that validates the code blocks in [retrieval-quality-golden-set.md](https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/documentation/tutorials-search-engineering/retrieval-quality-golden-set.md) actually run against a real Qdrant instance and the Anthropic API. Use this to sanity-check the tutorial before merging changes to it.

## What it tests

Every code block from the tutorial, in order:

1. `generate_queries_for_doc` — Anthropic call, parses one-query-per-line output.
2. `recall_at_k` / `mrr` — pure functions, checked against hand-computed values.
3. `ndcg_at_k` — ideal vs. reversed ordering.
4. `evaluate` — runs `client.query_points` through a labeled golden set and reports `mean_recall` / `mean_mrr`.

The harness builds a toy 8-doc corpus and a 5-query golden set where each query's relevant doc is hand-labeled, so correctness is verifiable (not just "it ran").

## Setup

```bash
cp .env.example .env          # paste QDRANT_URL, QDRANT_API_KEY, ANTHROPIC_API_KEY
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python harness.py                # full end-to-end (creates + deletes a test collection)
python generate_queries_only.py  # just the Anthropic block
```

`harness.py` creates a collection named `golden_set_tutorial_test`, upserts 8 points, runs the evaluation, and deletes the collection at the end.

## Expected output

If the tutorial is accurate, `harness.py` prints:

- `recall@3=1.0  mrr=0.5`
- `ndcg@3=1.000000` and `ndcg@3 (reversed)=0.586883`
- `{'mean_recall': 1.0, 'mean_mrr': 0.9}` on the golden set
- 3 queries from `generate_queries_for_doc`
- `ALL TESTS PASSED`

If any assertion fails or the Qdrant / Anthropic call shape has drifted from what the tutorial shows, the run exits non-zero.

## When to re-run

- Before merging any change to `retrieval-quality-golden-set.md`.
- After a Qdrant Python client major version bump (API shape changes).
- After changing the Anthropic model default in the tutorial.
