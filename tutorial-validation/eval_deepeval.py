"""Batch DeepEval run over the cached eval_records.jsonl produced by eval_ragas.py.

Reads results/eval_records.jsonl (must exist), scores the same generations with
DeepEval, writes:
  results/deepeval_per_query.csv
  results/deepeval_scores.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import settings
from pipeline.scoring import _resolve_judge_model, _score_deepeval_sync


RECORDS_PATH = Path(__file__).parent / "results" / "eval_records.jsonl"
PER_QUERY_PATH = Path(__file__).parent / "results" / "deepeval_per_query.csv"
SCORES_PATH = Path(__file__).parent / "results" / "deepeval_scores.json"


def main() -> int:
    if not RECORDS_PATH.exists():
        print(
            f"No cached records at {RECORDS_PATH}. Run eval_ragas.py first "
            "(it produces the shared eval_records.jsonl)."
        )
        return 1

    records = [json.loads(l) for l in RECORDS_PATH.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records)} cached records.")

    judge = _resolve_judge_model(settings.judge_model)
    print(f"Scoring with DeepEval (judge={judge})…")

    t0 = time.time()
    rows = []
    for rec in tqdm(records, desc="deepeval-score", unit="q"):
        try:
            scored = _score_deepeval_sync(rec, judge)
        except Exception as exc:
            print(f"  deepeval failed qid={rec['query_id']}: {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "query_id": rec["query_id"],
                "query_text": rec["query_text"],
                **{f"deepeval_{k}": v for k, v in scored["scores"].items()},
            }
        )
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(PER_QUERY_PATH, index=False)

    aggregates = {
        col: float(df[col].dropna().mean())
        for col in df.columns
        if col.startswith("deepeval_")
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
