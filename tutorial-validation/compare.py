"""Static comparison report + lead-library recommendation.

Reads:
  results/ragas_per_query.csv     (from eval_ragas.py)
  results/deepeval_per_query.csv  (from eval_deepeval.py)
  results/ragas_scores.json       (aggregates with wall_seconds)
  results/deepeval_scores.json    (aggregates with wall_seconds)
  results/manual_ratings.jsonl    (optional; from manual_judge.py)
  results/tutorial_gaps.md        (for the tutorial-update suggestions section)

Writes:
  results/comparison_report.md

The recommendation rubric is explicit: each criterion scores +1 to a winning
library; the lead-library line is "Lead with {winner} in the tutorial — won
{n}/6 criteria."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent
RES = ROOT / "results"

RAGAS_CSV = RES / "ragas_per_query.csv"
DE_CSV = RES / "deepeval_per_query.csv"
RAGAS_SCORES = RES / "ragas_scores.json"
DE_SCORES = RES / "deepeval_scores.json"
MANUAL = RES / "manual_ratings.jsonl"
GAPS = RES / "tutorial_gaps.md"
OUT = RES / "comparison_report.md"

OVERLAPPING = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _safe_corr(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(df) < 3 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return float("nan"), float("nan")
    pear = pearsonr(df["a"], df["b"]).statistic
    spear = spearmanr(df["a"], df["b"]).statistic
    return float(pear), float(spear)


def _md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _load_inputs() -> dict:
    if not (RAGAS_CSV.exists() and DE_CSV.exists()):
        print(f"Missing per-query CSVs. Need {RAGAS_CSV} and {DE_CSV}.")
        sys.exit(1)
    ragas = pd.read_csv(RAGAS_CSV)
    de = pd.read_csv(DE_CSV)
    merged = ragas.merge(de, on="query_id", suffixes=("", "_de"), how="inner")

    ragas_aggs = json.loads(RAGAS_SCORES.read_text()) if RAGAS_SCORES.exists() else {}
    de_aggs = json.loads(DE_SCORES.read_text()) if DE_SCORES.exists() else {}

    manual = pd.DataFrame()
    if MANUAL.exists():
        rows = [json.loads(l) for l in MANUAL.read_text().splitlines() if l.strip()]
        if rows:
            manual = pd.DataFrame(rows)
    return {
        "merged": merged,
        "ragas": ragas,
        "deepeval": de,
        "ragas_aggs": ragas_aggs,
        "de_aggs": de_aggs,
        "manual": manual,
    }


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------

def section_aggregate_table(merged: pd.DataFrame) -> str:
    rows = []
    for m in OVERLAPPING:
        rcol = f"ragas_{m}"
        dcol = f"deepeval_{m}"
        r = merged[rcol].mean() if rcol in merged.columns else float("nan")
        d = merged[dcol].mean() if dcol in merged.columns else float("nan")
        delta = (r or 0) - (d or 0) if not (pd.isna(r) or pd.isna(d)) else float("nan")
        rows.append([m, _fmt(r), _fmt(d), _fmt(delta)])
    return "## 1. Aggregate scores\n\n" + _md_table(rows, ["Metric", "Ragas", "DeepEval", "Δ (R-D)"])


def section_per_query_correlation(merged: pd.DataFrame) -> str:
    rows = []
    for m in OVERLAPPING:
        rcol = f"ragas_{m}"
        dcol = f"deepeval_{m}"
        if rcol not in merged.columns or dcol not in merged.columns:
            rows.append([m, "—", "—"])
            continue
        pear, spear = _safe_corr(merged[rcol], merged[dcol])
        rows.append([m, _fmt(pear), _fmt(spear)])
    table = _md_table(rows, ["Metric", "Pearson r", "Spearman ρ"])
    return (
        "## 2. Per-query correlation between libraries\n\n"
        "Both libraries should agree on directional quality. Pearson > 0.6 is a healthy floor "
        "for faithfulness and answer_relevancy; below that, the libraries are measuring different things.\n\n"
        + table
    )


def section_manual_correlation(merged: pd.DataFrame, manual: pd.DataFrame) -> tuple[str, dict]:
    """Returns (markdown, {library: {metric: pearson_r}}). The dict feeds the rubric."""
    if manual.empty:
        body = (
            "## 3. Correlation against manual ratings\n\n"
            "_No `results/manual_ratings.jsonl` found. Run `python manual_judge.py` to enable "
            "this section. Without it, the rubric falls back to per-query correlation as a "
            "weaker proxy for trustworthiness._"
        )
        return body, {"ragas": {}, "deepeval": {}}

    joined = merged.merge(manual, on="query_id", how="inner")
    rows = []
    correlations: dict = {"ragas": {}, "deepeval": {}}
    for human_metric, lib_suffix in [
        ("faithfulness_human", "faithfulness"),
        ("answer_relevancy_human", "answer_relevancy"),
    ]:
        if human_metric not in joined.columns:
            continue
        for lib in ("ragas", "deepeval"):
            col = f"{lib}_{lib_suffix}"
            pear, spear = _safe_corr(joined[col], joined[human_metric])
            rows.append([lib_suffix, lib, _fmt(pear), _fmt(spear)])
            correlations[lib][lib_suffix] = pear

    table = _md_table(rows, ["Metric", "Library", "Pearson r", "Spearman ρ"])
    body = (
        "## 3. Correlation against manual ratings\n\n"
        f"Computed on {len(joined)} hand-rated queries. The library with higher Pearson "
        "is the more trustworthy judge on this corpus.\n\n"
        + table
    )
    return body, correlations


def section_disagreement_audit(merged: pd.DataFrame) -> str:
    if "ragas_faithfulness" not in merged.columns or "deepeval_faithfulness" not in merged.columns:
        return "## 4. Disagreement audit\n\n_Faithfulness columns missing._"
    diffs = (merged["ragas_faithfulness"] - merged["deepeval_faithfulness"]).abs()
    top = merged.assign(abs_delta=diffs).nlargest(10, "abs_delta")
    rows = []
    for _, r in top.iterrows():
        rows.append([
            r["query_id"],
            (r.get("query_text") or r.get("query_text_de") or "")[:60].replace("\n", " "),
            _fmt(r.get("ragas_faithfulness")),
            _fmt(r.get("deepeval_faithfulness")),
            _fmt(r["abs_delta"]),
        ])
    table = _md_table(rows, ["query_id", "query (truncated)", "Ragas", "DeepEval", "|Δ|"])
    return (
        "## 4. Top-10 faithfulness disagreements\n\n"
        "Eyeballing 3 of these answers the question 'when they disagree, which one is right?' — "
        "leave a 1-line note per inspection.\n\n"
        + table
    )


def section_runtime_cost(ragas_aggs: dict, de_aggs: dict, n_queries: int) -> str:
    rwall = ragas_aggs.get("wall_seconds", float("nan"))
    dwall = de_aggs.get("wall_seconds", float("nan"))
    rper = (rwall / n_queries) if (rwall and n_queries) else float("nan")
    dper = (dwall / n_queries) if (dwall and n_queries) else float("nan")
    rows = [
        ["wall seconds (total)", _fmt(rwall), _fmt(dwall)],
        ["wall seconds / query", _fmt(rper), _fmt(dper)],
        ["judge model used", ragas_aggs.get("judge_model_used", "—"), de_aggs.get("judge_model_used", "—")],
    ]
    body = (
        "## 5. Runtime + judge model\n\n"
        "Token usage is not exposed by either library on its metric objects, so judge-token "
        "cost has to be estimated upstream (the Streamlit cost meter does this directionally). "
        "Wall-clock comparison below.\n\n"
        + _md_table(rows, ["Field", "Ragas", "DeepEval"])
    )
    return body, rper, dper


def section_feature_matrix(setup_loc: dict[str, int]) -> str:
    rows = [
        ["Pytest-native API", "No", "Yes (`assert_test`)"],
        ["Custom metrics via natural language", "Limited", "Yes (G-Eval)"],
        ["Reference-free metrics", "Yes", "Yes"],
        ["Built-in dataset/UI", "No", "Yes (Confident AI cloud)"],
        ["Agent-trajectory metrics", "Yes (newer)", "Yes"],
        ["Setup LOC for this pipeline", str(setup_loc.get("ragas", "—")), str(setup_loc.get("deepeval", "—"))],
        ["Judge reasoning ergonomics", "Available, but undocumented — walk `result.ragas_traces`", "Direct: `metric.reason`"],
        ["Reasoning granularity", "Per atomic claim / chunk", "Holistic (one string per metric)"],
    ]
    return (
        "## 6. Library feature matrix\n\n"
        + _md_table(rows, ["Capability", "Ragas", "DeepEval"])
    )


def section_recommendation(
    correlations: dict,
    rper: float,
    dper: float,
    setup_loc: dict[str, int],
    has_geval_signal: bool,
) -> str:
    rows = []
    score = {"ragas": 0, "deepeval": 0}

    def winner_or_tie(a: float, b: float, lo_better: bool = False) -> str:
        if pd.isna(a) and pd.isna(b):
            return "—"
        if pd.isna(a):
            return "deepeval"
        if pd.isna(b):
            return "ragas"
        if a == b:
            return "tie"
        if lo_better:
            return "ragas" if a < b else "deepeval"
        return "ragas" if a > b else "deepeval"

    # 1: faithfulness manual correlation
    f_r = correlations.get("ragas", {}).get("faithfulness", float("nan"))
    f_d = correlations.get("deepeval", {}).get("faithfulness", float("nan"))
    w = winner_or_tie(f_r, f_d, lo_better=False)
    rows.append(["Higher correlation with manual faithfulness", _fmt(f_r), _fmt(f_d), w])
    if w in score:
        score[w] += 1

    # 2: answer_relevancy manual correlation
    r_r = correlations.get("ragas", {}).get("answer_relevancy", float("nan"))
    r_d = correlations.get("deepeval", {}).get("answer_relevancy", float("nan"))
    w = winner_or_tie(r_r, r_d, lo_better=False)
    rows.append(["Higher correlation with manual answer_relevancy", _fmt(r_r), _fmt(r_d), w])
    if w in score:
        score[w] += 1

    # 3: setup LOC (lower wins)
    lr = setup_loc.get("ragas", float("nan"))
    ld = setup_loc.get("deepeval", float("nan"))
    w = winner_or_tie(lr, ld, lo_better=True)
    rows.append(["Lower setup LOC", _fmt(lr), _fmt(ld), w])
    if w in score:
        score[w] += 1

    # 4: wall seconds per query (lower wins)
    w = winner_or_tie(rper, dper, lo_better=True)
    rows.append(["Lower wall-clock per query", _fmt(rper), _fmt(dper), w])
    if w in score:
        score[w] += 1

    # 5: cost per query proxy = wall seconds (we don't have separate token counts).
    # Skip with explicit note rather than double-count.
    rows.append(["Lower judge-token cost per query (proxy: wall seconds)", "see #4", "see #4", "skipped"])

    # 6: catches a failure mode the other misses (G-Eval signal)
    w = "deepeval" if has_geval_signal else "tie"
    rows.append(["Catches a failure mode via G-Eval", "—", "✓" if has_geval_signal else "—", w])
    if w in score:
        score[w] += 1

    table = _md_table(rows, ["Criterion", "Ragas", "DeepEval", "Winner"])
    if score["ragas"] > score["deepeval"]:
        winner = "Ragas"
    elif score["deepeval"] > score["ragas"]:
        winner = "DeepEval"
    else:
        winner = "tie"
    n = max(score.values())
    total = 5  # 6 criteria, one is "skipped"

    headline = (
        f"**Lead with {winner} in the tutorial — won {n}/{total} scored criteria** "
        f"(Ragas: {score['ragas']}, DeepEval: {score['deepeval']})."
    )
    return (
        "## 7. Recommendation rubric\n\n"
        + headline
        + "\n\n"
        + table
        + "\n\n_The 5th criterion (judge-token cost) is folded into wall-clock since neither "
        "library exposes per-metric token counts. The 6th (G-Eval failure-mode catch) is "
        "DeepEval-only by definition; it scores +1 only when at least one query had a "
        "G-Eval custom score < 0.5 while standard metrics scored ≥ 0.7._"
    )


def section_caveat() -> str:
    return (
        "## 8. Reference-answer quality caveat\n\n"
        "The golden-set references were LLM-generated (constrained to the relevant FiQA "
        "passages, but still LLM output). This is the weakest link in the methodology — "
        "`context_precision` and `context_recall` are scored against text that itself "
        "wasn't human-validated. Scores can shift by a few points depending on the "
        "reference-generation prompt. **Treat the recommendation as directional, not "
        "definitive, and re-run with human-written references if the call is close** "
        "(margin ≤ 1 criterion)."
    )


def section_tutorial_gaps() -> str:
    if not GAPS.exists():
        return "## 9. Tutorial-update suggestions\n\n_results/tutorial_gaps.md not found._"
    raw = GAPS.read_text()
    # Pull out section headers as a quick punch list. We don't strip prose — full file is short.
    return (
        "## 9. Tutorial-update suggestions\n\n"
        "Top gaps logged during the build (full text in `results/tutorial_gaps.md`):\n\n"
        + raw.split("---", 1)[-1]
    )


# ----------------------------------------------------------------------------
# Setup-LOC counter — directional, counts non-blank lines per scoring helper.
# ----------------------------------------------------------------------------

def _count_loc_in_file(path: Path, between: tuple[str, str]) -> int:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return 0
    start_marker, end_marker = between
    start = text.find(start_marker)
    end = text.find(end_marker, start + 1) if start != -1 else -1
    if start == -1 or end == -1:
        return 0
    block = text[start:end]
    return sum(1 for line in block.splitlines() if line.strip() and not line.strip().startswith("#"))


def setup_loc_counts() -> dict[str, int]:
    scoring = ROOT / "pipeline" / "scoring.py"
    return {
        "ragas": _count_loc_in_file(
            scoring,
            ("# ----------------------------------------------------------------------------\n# Ragas",
             "# ----------------------------------------------------------------------------\n# DeepEval"),
        ),
        "deepeval": _count_loc_in_file(
            scoring,
            ("# ----------------------------------------------------------------------------\n# DeepEval",
             "# ----------------------------------------------------------------------------\n# Combined"),
        ),
    }


def has_geval_signal() -> bool:
    """G-Eval signal: at least one row with deepeval_g_eval_custom < 0.5 while
    deepeval_faithfulness >= 0.7. Uses deepeval_per_query.csv as the source."""
    if not DE_CSV.exists():
        return False
    df = pd.read_csv(DE_CSV)
    if "deepeval_g_eval_custom" not in df.columns:
        return False
    fcol = "deepeval_faithfulness"
    gcol = "deepeval_g_eval_custom"
    if fcol not in df.columns:
        return False
    mask = (df[gcol] < 0.5) & (df[fcol] >= 0.7)
    return bool(mask.any())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    data = _load_inputs()
    merged = data["merged"]
    n = len(merged)

    parts: list[str] = []
    parts.append(f"# Ragas vs DeepEval — comparison report\n\n_Generated from {n} queries._\n")

    parts.append(section_aggregate_table(merged))
    parts.append(section_per_query_correlation(merged))

    manual_md, correlations = section_manual_correlation(merged, data["manual"])
    parts.append(manual_md)

    parts.append(section_disagreement_audit(merged))

    runtime_md, rper, dper = section_runtime_cost(data["ragas_aggs"], data["de_aggs"], n)
    parts.append(runtime_md)

    locs = setup_loc_counts()
    parts.append(section_feature_matrix(locs))

    geval_signal = has_geval_signal()
    parts.append(section_recommendation(correlations, rper, dper, locs, geval_signal))

    parts.append(section_caveat())
    parts.append(section_tutorial_gaps())

    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
