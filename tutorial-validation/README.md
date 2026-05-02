# Ragas vs DeepEval — Live Demo

A production-shaped RAG pipeline (FiQA corpus, hybrid Qdrant retrieval, cross-encoder rerank, Claude generator) scored side-by-side by Ragas and DeepEval. Built as a live Streamlit demo plus a batch comparison report that ends with a data-backed lead-library recommendation.

This project also validates the `pipeline-output-quality.md` tutorial by following its patterns and logging every gap to `results/tutorial_gaps.md`.

## Setup

```bash
cp .env.example .env   # fill in QDRANT_URL, QDRANT_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## One-shot batch scripts

```bash
python ingest.py        # ~15-25 min, ~$1 OpenAI embedding cost; idempotent
python golden_set.py    # ~5 min; produces results/golden_set.jsonl
```

## Live demo (centerpiece)

```bash
streamlit run app.py
```

Sidebar knobs let the presenter tweak the pipeline (prompt, k, hybrid/rerank, models, thresholds, custom G-Eval criterion). Both eval libraries score the same `(question, retrieved_context, generated_answer, reference)` samples, scores stream into the table per-query, and the drill-down panel surfaces the judge's reasoning for every metric.

## Batch path (static report)

```bash
python eval_ragas.py          # ~10-20 min
python eval_deepeval.py       # ~10-20 min
python manual_judge.py        # ~10 min interactive (rate 10 queries)
pytest tests/                 # CI-gate demo (DeepEval pytest-native API)
python compare.py             # → results/comparison_report.md
```

## Cost expectations

- Live demo run (20 queries): ~$1–2
- Full batch run (300 queries): ~$15–25
- Ingest: one-shot ~$1, never re-run on tweak

## Files

- `app.py` — Streamlit live demo
- `ingest.py`, `golden_set.py` — one-shot offline scripts
- `pipeline/` — retrieval, generation, end-to-end pipeline, parallel scoring helpers
- `eval_ragas.py`, `eval_deepeval.py` — batch eval scripts
- `compare.py` — comparison report + lead-library recommendation rubric
- `manual_judge.py` — capture human ratings on 10 queries
- `tests/test_pipeline_deepeval.py` — pytest-native CI gate demo
- `results/` — produced artifacts (CSV/JSON/JSONL, comparison report, tutorial gap log)
