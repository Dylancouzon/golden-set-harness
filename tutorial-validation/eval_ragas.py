"""Batch Ragas run over the full golden set.

Caches results/eval_records.jsonl (the generated answers) so eval_deepeval.py
scores the same artifacts. Writes:
  results/ragas_per_query.csv
  results/ragas_scores.json   (aggregates)

Usage: python eval_ragas.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DEFAULT_PROMPT_TEMPLATE, settings
from pipeline.pipeline import run_one
from pipeline.scoring import _resolve_judge_model, _score_ragas_async


GOLDEN_PATH = Path(__file__).parent / "results" / "golden_set.jsonl"
RECORDS_PATH = Path(__file__).parent / "results" / "eval_records.jsonl"
PER_QUERY_PATH = Path(__file__).parent / "results" / "ragas_per_query.csv"
SCORES_PATH = Path(__file__).parent / "results" / "ragas_scores.json"


def _generate_records(golden: list[dict]) -> list[dict]:
    """Generate-and-cache eval records. Reuses cached file if it exists with matching qids."""
    cached: dict[str, dict] = {}
    if RECORDS_PATH.exists():
        for line in RECORDS_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cached[r["query_id"]] = r

    out = []
    new_count = 0
    pbar = tqdm(golden, desc="generate", unit="q")
    with RECORDS_PATH.open("a") as fout:
        for q in pbar:
            qid = q["query_id"]
            if qid in cached:
                out.append(cached[qid])
                continue
            try:
                rec = run_one(
                    q["query_text"],
                    k_retrieve=settings.top_k_retrieve,
                    k_rerank=settings.top_k_rerank,
                    hybrid=True,
                    rerank=True,
                    generator_model=settings.generator_model,
                    prompt_template=DEFAULT_PROMPT_TEMPLATE,
                )
            except Exception as exc:
                print(f"  generation failed qid={qid}: {exc}", file=sys.stderr)
                continue
            rec["query_id"] = qid
            rec["ground_truth"] = q["ground_truth"]
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            out.append(rec)
            new_count += 1
    print(f"Generated {new_count} new, used {len(out) - new_count} cached.")
    return out


async def _score_all(records: list[dict], judge: str) -> list[dict]:
    rows = []
    for rec in tqdm(records, desc="ragas-score", unit="q"):
        try:
            scored = await _score_ragas_async(rec, judge)
        except Exception as exc:
            print(f"  ragas failed qid={rec['query_id']}: {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "query_id": rec["query_id"],
                "query_text": rec["query_text"],
                **{f"ragas_{k}": v for k, v in scored["scores"].items()},
            }
        )
    return rows


def main() -> int:
    if not GOLDEN_PATH.exists():
        print(f"No golden set at {GOLDEN_PATH}; run python golden_set.py first.")
        return 1
    golden = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(golden)} golden entries.")

    records = _generate_records(golden)
    if not records:
        print("No records produced; aborting.")
        return 1

    judge = _resolve_judge_model(settings.judge_model)
    print(f"Scoring with Ragas (judge={judge})…")
    t0 = time.time()
    rows = asyncio.run(_score_all(records, judge))
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(PER_QUERY_PATH, index=False)

    aggregates = {
        col: float(df[col].dropna().mean())
        for col in df.columns
        if col.startswith("ragas_")
    }
    aggregates["wall_seconds"] = elapsed
    aggregates["n_queries"] = len(df)
    aggregates["judge_model_used"] = judge
    SCORES_PATH.write_text(json.dumps(aggregates, indent=2))

    print(f"Wrote {PER_QUERY_PATH} and {SCORES_PATH}.")
    print(f"Wall: {elapsed/60:.1f} min, judge={judge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
