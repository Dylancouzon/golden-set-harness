"""Capture human ratings on a deterministic 10-query slice.

Without human-anchored ratings, the comparison report can only say "the
libraries disagree by X" — not "library Y was right". This script captures
a small human ground truth (~10 minutes of work) so the recommendation
rubric can answer the lead-library question with evidence rather than vibes.

Picks the same 10 queries every run (seeded), prints each, prompts for
faithfulness + answer_relevancy on a 0-1 scale. 's' to skip.

Output: results/manual_ratings.jsonl, one JSON object per line:
  {"query_id", "faithfulness_human", "answer_relevancy_human"}
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from config import settings


RECORDS_PATH = Path(__file__).parent / "results" / "eval_records.jsonl"
OUT_PATH = Path(__file__).parent / "results" / "manual_ratings.jsonl"
N_QUERIES = 10


def _pick_ten(records: list[dict]) -> list[dict]:
    rng = random.Random(settings.random_seed)
    pool = list(records)
    rng.shuffle(pool)
    return pool[: min(N_QUERIES, len(pool))]


def _ask(prompt: str) -> float | None:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"s", "skip", ""}:
            return None
        try:
            v = float(raw)
        except ValueError:
            print("  parse error — enter a number 0..1 or 's' to skip.")
            continue
        if not 0.0 <= v <= 1.0:
            print("  out of range — must be between 0 and 1.")
            continue
        return v


def _print_record(rec: dict, n: int, total: int) -> None:
    print("\n" + "=" * 70)
    print(f"[{n}/{total}] qid={rec['query_id']}")
    print("=" * 70)
    print(f"\nQUESTION:\n  {rec['query_text']}")
    print(f"\nRETRIEVED CONTEXTS ({len(rec['contexts'])}):")
    for i, c in enumerate(rec["contexts"], 1):
        snippet = (c[:280] + "…") if len(c) > 280 else c
        print(f"  [{i}] {snippet}")
    print(f"\nGENERATED ANSWER:\n  {rec['answer']}")
    print(f"\nREFERENCE ANSWER:\n  {rec['ground_truth']}")
    print()


def main() -> int:
    if not RECORDS_PATH.exists():
        print(
            f"No cached records at {RECORDS_PATH}. Run eval_ragas.py first "
            "to produce eval_records.jsonl."
        )
        return 1

    records = [json.loads(l) for l in RECORDS_PATH.read_text().splitlines() if l.strip()]
    chosen = _pick_ten(records)
    print(f"Rating {len(chosen)} queries. Enter 0..1 or 's' to skip.\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fout:
        for n, rec in enumerate(chosen, start=1):
            _print_record(rec, n, len(chosen))
            f = _ask("  faithfulness 0..1 (s to skip): ")
            a = _ask("  answer_relevancy 0..1 (s to skip): ")
            if f is None and a is None:
                continue
            entry = {
                "query_id": rec["query_id"],
                "faithfulness_human": f,
                "answer_relevancy_human": a,
            }
            fout.write(json.dumps(entry) + "\n")
            fout.flush()
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
