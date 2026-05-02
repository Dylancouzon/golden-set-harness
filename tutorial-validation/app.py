"""Live demo: Ragas vs DeepEval on a production-shaped RAG pipeline.

Run: streamlit run app.py

Sidebar knobs let the presenter tweak the pipeline (prompt, k, hybrid/rerank,
generator/judge models, thresholds, custom G-Eval criterion). Per-query scores
stream into the table as they finish; threshold sliders re-bin without re-scoring;
the drill-down panel surfaces judge reasoning side by side.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DEFAULT_PROMPT_TEMPLATE,
    GENERATOR_CHOICES,
    JUDGE_CHOICES,
    model_family,
    settings,
)
from pipeline.pipeline import run_one
from pipeline.retrieval import collection_status, update_dense_hnsw
from pipeline.scoring import score_both_parallel

GOLDEN_PATH = Path(__file__).parent / "results" / "golden_set.jsonl"

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_golden_set() -> list[dict]:
    """Read results/golden_set.jsonl. Cached for the session so file IO doesn't
    happen on every Streamlit re-run (the script re-executes top-to-bottom on
    every interaction)."""
    if not GOLDEN_PATH.exists():
        return []
    return [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]


def sample_queries(golden: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministically pick `n` queries. Same seed + same n → same subset,
    which is what the slider depends on for stable demo runs."""
    rng = random.Random(seed)
    pool = list(golden)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


# ----------------------------------------------------------------------------
# Cost estimation (rough — for the live cost meter, not billing)
# ----------------------------------------------------------------------------
# Per-token rates in USD/1K tokens for each model the demo can use. Pulled
# from public price lists; treat as directional. The cost meter aggregates
# generator usage + an approximated judge usage per row.
#
# Order matters: longer prefixes must come BEFORE shorter ones because the
# fallback matcher uses str.startswith and picks the first hit. e.g.
# "claude-haiku-4-5" must precede "claude" so we don't accidentally apply
# Sonnet pricing to Haiku.
COST_PER_1K = {
    # Anthropic
    "claude-opus-4-7":     {"in": 0.015,   "out": 0.075},
    "claude-sonnet-4-6":   {"in": 0.003,   "out": 0.015},
    "claude-haiku-4-5":    {"in": 0.0008,  "out": 0.004},
    "claude-sonnet":       {"in": 0.003,   "out": 0.015},
    "claude-haiku":        {"in": 0.0008,  "out": 0.004},
    # OpenAI
    "gpt-5.4":             {"in": 0.005,   "out": 0.015},
    "gpt-5-mini":          {"in": 0.00025, "out": 0.002},
    "gpt-5-nano":          {"in": 0.00005, "out": 0.0004},
    "gpt-4o-mini":         {"in": 0.00015, "out": 0.0006},
    "gpt-4o":              {"in": 0.005,   "out": 0.015},
}


def _model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K.get(model)
    if rates is None:
        for prefix, r in COST_PER_1K.items():
            if model.startswith(prefix):
                rates = r
                break
    if rates is None:
        return 0.0
    return (input_tokens / 1000.0) * rates["in"] + (output_tokens / 1000.0) * rates["out"]


# ----------------------------------------------------------------------------
# Aggregate / pass-fail helpers (used live)
# ----------------------------------------------------------------------------

# The four overlapping metrics — both libs name them the same way.
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _safe_mean(series: pd.Series) -> float:
    """Mean ignoring NaN. Returns NaN when the series is fully empty."""
    s = series.dropna()
    return float(s.mean()) if len(s) else float("nan")


def _format_aggregates(df: pd.DataFrame, faith_thresh: float, relev_thresh: float) -> dict:
    """Per-metric averages + pass/total counts for the scorecard row."""
    if df.empty:
        return {}
    out = {}
    for m in RAGAS_METRICS:
        out[f"ragas_{m}"] = _safe_mean(df.get(f"ragas_{m}", pd.Series(dtype=float)))
        out[f"deepeval_{m}"] = _safe_mean(df.get(f"deepeval_{m}", pd.Series(dtype=float)))
    if "deepeval_g_eval_custom" in df.columns:
        out["deepeval_g_eval_custom"] = _safe_mean(df["deepeval_g_eval_custom"])
    out["pass_count"] = int(_pass_mask(df, faith_thresh, relev_thresh).sum())
    out["total"] = int(len(df))
    return out


def _pass_mask(df: pd.DataFrame, faith_thresh: float, relev_thresh: float) -> pd.Series:
    """Pass = both libs' faithfulness AND both libs' answer_relevancy clear the
    threshold. The strictest interpretation — a row only passes if every
    relevant signal agrees. Re-run on every Streamlit interaction so threshold
    sliders flip pass/fail badges instantly without re-scoring."""
    if df.empty:
        return pd.Series([], dtype=bool)
    f_ok = (
        (df.get("ragas_faithfulness", 0).fillna(0) >= faith_thresh)
        & (df.get("deepeval_faithfulness", 0).fillna(0) >= faith_thresh)
    )
    r_ok = (
        (df.get("ragas_answer_relevancy", 0).fillna(0) >= relev_thresh)
        & (df.get("deepeval_answer_relevancy", 0).fillna(0) >= relev_thresh)
    )
    return f_ok & r_ok


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Ragas vs DeepEval — live", layout="wide")
st.title("Ragas vs DeepEval — live RAG eval")
st.caption(
    "Hybrid Qdrant retrieval (dense + sparse + RRF) → cross-encoder rerank → "
    "Claude generator → both eval libraries score the same samples in parallel."
)

# Sidebar
with st.sidebar:
    st.header("Pipeline knobs")
    sample_size = st.slider(
        "Sample size", min_value=10, max_value=300, value=20, step=5,
        help=(
            "How many queries to draw from the cached golden set. Sampling is "
            "deterministic (seed 42), so the same value always picks the same "
            "subset. 20 ≈ a $1–2 / 10–15 min run; 300 ≈ a $15–25 / 30–45 min run."
        ),
    )

    generator_model = st.selectbox(
        "Generator model", GENERATOR_CHOICES,
        index=GENERATOR_CHOICES.index(settings.generator_model)
        if settings.generator_model in GENERATOR_CHOICES else 0,
        help=(
            "The LLM that produces the RAG answer (the system under test). "
            "Mini/nano models cut cost ~10x and roughly halve generator wall-clock "
            "but may be less faithful on hard queries."
        ),
    )
    judge_model = st.selectbox(
        "Judge model", JUDGE_CHOICES,
        index=JUDGE_CHOICES.index(settings.judge_model)
        if settings.judge_model in JUDGE_CHOICES else 0,
        help=(
            "The LLM both eval libraries call to score each sample. Pick a "
            "DIFFERENT family from the generator to avoid self-judging bias. "
            "Bigger judges correlate better with humans; mini judges are cheaper "
            "but noisier. gpt-5-mini/nano are excluded because Ragas's internal "
            "temperature override breaks them."
        ),
    )
    if model_family(generator_model) == model_family(judge_model):
        st.warning(
            "Same model family — self-judging contamination risk. "
            "Pick a judge from a different family."
        )

    k_retrieve = st.slider(
        "top_k_retrieve", 10, 100, settings.top_k_retrieve, step=10,
        help=(
            "How many candidates Qdrant returns from hybrid search before "
            "reranking. Higher = better recall (rerank can rescue more), "
            "slower (cross-encoder pays more). Sweet spot: 50."
        ),
    )
    k_rerank = st.slider(
        "top_k_rerank", 1, 20, settings.top_k_rerank, step=1,
        help=(
            "How many passages survive the cross-encoder and end up in the LLM "
            "prompt. Lower = cheaper generator, risks 'context doesn't contain "
            "the answer'. Higher = more grounding, but signal-to-noise drops "
            "and faithfulness can decrease."
        ),
    )
    hybrid = st.toggle(
        "Hybrid (dense + sparse RRF)", value=True,
        help=(
            "On = dense (OpenAI embedding) + sparse (BM25) prefetches fused with "
            "Reciprocal Rank Fusion. Off = dense-only. Hybrid usually wins on "
            "queries with rare or proper-noun terms BM25 catches but the dense "
            "model misses."
        ),
    )
    rerank = st.toggle(
        "Cross-encoder rerank", value=True,
        help=(
            "On = ms-marco-MiniLM scores each (query, passage) pair directly and "
            "reorders. Slower but typically a big precision win. Off skips this "
            "stage and uses the hybrid order — cheaper and avoids the 120 MB "
            "model download if you're on a fresh checkout."
        ),
    )

    st.divider()
    st.subheader("HNSW tuning (dense vector)")
    st.caption(
        "search_hnsw_ef is per-query and free to flip. m and ef_construct "
        "are build-time — clicking Apply rebuilds the graph in place "
        "(no re-embed needed)."
    )
    search_hnsw_ef = st.slider(
        "search_hnsw_ef (per-query)", 16, 512, settings.search_hnsw_ef, step=16,
        help=(
            "Search-time HNSW candidate breadth — how many graph nodes to "
            "consider during a single query. Higher = better recall, slower "
            "queries. Free to change per call (no re-index)."
        ),
    )
    hnsw_m = st.slider(
        "m (graph degree)", 4, 64, settings.hnsw_m, step=2,
        help=(
            "HNSW edges per node. Higher = denser graph = better recall but "
            "more RAM and slower ingest/re-index. Build-time: changes require "
            "Apply (which triggers an in-place rebuild)."
        ),
    )
    hnsw_ef_construct = st.slider(
        "ef_construct (build breadth)", 32, 512, settings.hnsw_ef_construct, step=16,
        help=(
            "Candidate pool when adding nodes during graph construction. Higher "
            "= better graph quality, slower to build. Build-time: changes "
            "require Apply."
        ),
    )
    apply_hnsw = st.button(
        "Apply HNSW config", use_container_width=True,
        help=(
            "Sends the new m / ef_construct to Qdrant via update_collection. "
            "Qdrant rebuilds the dense vector's HNSW graph in place — no "
            "re-embed cost. Status panel below tracks the rebuild "
            "(green → yellow → green)."
        ),
    )
    hnsw_status_box = st.empty()

    st.divider()
    st.subheader("CI gate thresholds")
    st.caption("Sliders re-bin existing scores live — no re-scoring.")
    faith_thresh = st.slider(
        "Faithfulness threshold", 0.0, 1.0, 0.7, step=0.05,
        help=(
            "A row PASSES only if both Ragas faithfulness AND DeepEval "
            "faithfulness clear this bar. Moving the slider re-bins existing "
            "scores instantly — no LLM calls. This is the live CI-gate demo."
        ),
    )
    relev_thresh = st.slider(
        "Answer-relevancy threshold", 0.0, 1.0, 0.7, step=0.05,
        help=(
            "Same idea as faithfulness threshold but for answer_relevancy. "
            "Both metrics gate pass/fail; raising either tightens the gate."
        ),
    )

    st.divider()
    st.subheader("Prompt template")
    prompt_template = st.text_area(
        "PROMPT_TEMPLATE", value=DEFAULT_PROMPT_TEMPLATE, height=180,
        help=(
            "The grounding prompt sent to the generator. MUST contain the "
            "literal placeholders {retrieved_context} and {query_text} — "
            "those get filled in per query. Editing this changes what the "
            "generator sees, which changes the answer, which changes the "
            "scores. Watch faithfulness if you loosen the 'use only the "
            "context' instruction."
        ),
    )

    st.subheader("G-Eval custom criterion (DeepEval only)")
    geval_criterion = st.text_area(
        "When set, adds a g_eval_custom column on the next run.",
        value="",
        placeholder="Every numerical claim in the answer must be supported by a specific quoted span from the retrieved context.",
        height=120,
        help=(
            "DeepEval-only: write a natural-language pass/fail rule (e.g. 'no "
            "made-up dollar amounts'). DeepEval's G-Eval metric asks the judge "
            "to apply this criterion as a custom score on every sample. Leave "
            "blank to skip. This is the headline DeepEval differentiator — "
            "Ragas has no equivalent."
        ),
    )

    run_clicked = st.button(
        "▶ Run sample", type="primary", use_container_width=True,
        help=(
            "Execute the pipeline + scoring on N queries with the current "
            "config. Per-query scores stream into the table as they finish."
        ),
    )
    reset_clicked = st.button(
        "Reset / clear cache", use_container_width=True,
        help=(
            "Clears the in-session record cache, the cost meter, and "
            "Streamlit's @cache_data. Use this when you want to re-load the "
            "golden set after editing it on disk, or zero out the cost meter."
        ),
    )

# Mirror sidebar state into session_state. The drill-down + threshold-rebinning
# logic at the bottom of the script reads from session_state so it picks up
# fresh slider values on every re-run without forcing a re-score.
for k, v in {
    "sample_size": sample_size, "k_retrieve": k_retrieve, "k_rerank": k_rerank,
    "hybrid": hybrid, "rerank": rerank,
    "generator_model": generator_model, "judge_model": judge_model,
    "prompt_template": prompt_template, "geval_criterion": geval_criterion,
    "faith_thresh": faith_thresh, "relev_thresh": relev_thresh,
}.items():
    st.session_state[k] = v

if reset_clicked:
    for k in ["last_run", "records", "reasons", "cost_usd", "judge_active"]:
        st.session_state.pop(k, None)
    st.cache_data.clear()
    st.success("Caches cleared. Click Run sample.")

# ----------------------------------------------------------------------------
# HNSW: apply build-time changes + render live status
# ----------------------------------------------------------------------------
# Updating m / ef_construct is fire-and-forget — Qdrant rebuilds the dense
# vector's HNSW graph in the background. We surface the rebuild progress in
# the sidebar so the presenter can wait until status flips back to green
# before kicking off a sample.

if apply_hnsw:
    try:
        update_dense_hnsw(m=hnsw_m, ef_construct=hnsw_ef_construct)
        st.toast(
            f"HNSW update sent (m={hnsw_m}, ef_construct={hnsw_ef_construct}). "
            "Status panel below will track the rebuild."
        )
    except Exception as exc:
        st.error(f"HNSW update failed: {exc}")


def _render_hnsw_status() -> None:
    try:
        s = collection_status()
    except Exception as exc:
        hnsw_status_box.warning(f"Status unavailable: {exc}")
        return
    label = {
        "green": ("✅ ready (graph built)", "info"),
        "yellow": ("⚙️ optimising / re-indexing", "warning"),
        "grey": ("⏸ pending", "warning"),
        "red": ("❌ error", "error"),
    }.get(s["status"], (f"? {s['status']}", "info"))
    msg = (
        f"**Index status:** {label[0]}  \n"
        f"{s['points']:,} points · {s['indexed_vectors']:,} vectors indexed  \n"
        f"optimizer: {s['optimizer_status']}"
    )
    getattr(hnsw_status_box, label[1])(msg)


_render_hnsw_status()

# Sidebar cost meter (rendered after the run loop updates session state)
cost_placeholder = st.sidebar.empty()
judge_placeholder = st.sidebar.empty()


def _render_cost_meter():
    cost = st.session_state.get("cost_usd", 0.0)
    cost_placeholder.metric("Approx. cost (this session)", f"${cost:.3f}")
    judge = st.session_state.get("judge_active") or judge_model
    if judge != judge_model:
        judge_placeholder.warning(f"Active judge: **{judge}** (fell back from {judge_model})")
    else:
        judge_placeholder.info(f"Active judge: **{judge}**")


_render_cost_meter()

# ----------------------------------------------------------------------------
# Main run loop
# ----------------------------------------------------------------------------

main_placeholder = st.container()

if run_clicked:
    golden = load_golden_set()
    if not golden:
        st.error(
            f"No golden set at {GOLDEN_PATH}. Run `python golden_set.py` first."
        )
    else:
        # Deterministic sample so the same slider value always picks the same
        # queries — important for repeatable demos.
        sampled = sample_queries(golden, sample_size, seed=settings.random_seed)
        st.session_state.setdefault("records", {})
        st.session_state.setdefault("reasons", {})
        st.session_state["cost_usd"] = 0.0

        with main_placeholder:
            # st.empty() returns placeholder slots we'll keep overwriting from
            # the loop. This is what makes the table appear to "stream" — we
            # render the full DataFrame after every row.
            st.subheader(f"Running {len(sampled)} queries…")
            metrics_box = st.empty()
            table_box = st.empty()
            scatter_box = st.empty()
            progress = st.progress(0.0)

        rows: list[dict] = []
        for i, q in enumerate(sampled):
            try:
                record = run_one(
                    q["query_text"],
                    k_retrieve=k_retrieve,
                    k_rerank=k_rerank,
                    hybrid=hybrid,
                    rerank=rerank,
                    generator_model=generator_model,
                    prompt_template=prompt_template,
                    hnsw_ef=search_hnsw_ef,
                )
            except Exception as exc:
                st.error(f"Pipeline failed on qid={q['query_id']}: {exc}")
                continue

            record["ground_truth"] = q["ground_truth"]
            record["query_id"] = q["query_id"]

            try:
                scored = score_both_parallel(
                    record,
                    judge_model=judge_model,
                    custom_geval_criterion=(geval_criterion.strip() or None),
                )
            except Exception as exc:
                st.error(f"Scoring failed on qid={q['query_id']}: {exc}")
                continue

            # Surface the judge that was actually used (may have fallen back).
            st.session_state["judge_active"] = scored["judge_model_used"]

            # ---- Cost meter contribution ----
            # The generator's token usage is exact (returned by the SDK).
            # The judge's token usage is approximated because neither Ragas
            # nor DeepEval expose token counts on their metric objects. We
            # rule-of-thumb 1 char ≈ 0.25 tokens, multiplied by 4 metrics in
            # and ~150 tokens out per metric. Use this as a directional
            # signal for "is this run getting expensive", not for billing.
            gen_usage = record.get("generator_usage", {})
            cost_inc = _model_cost(
                generator_model,
                gen_usage.get("input_tokens", 0),
                gen_usage.get("output_tokens", 0),
            )
            input_chars = (
                sum(len(c) for c in record["contexts"])
                + len(record["answer"])
                + len(record["ground_truth"])
            )
            judge_in_est = int(input_chars / 4) * 4  # 4 metrics × per-metric input
            judge_out_est = 600  # ~150 output tokens × 4 metrics
            cost_inc += _model_cost(scored["judge_model_used"], judge_in_est, judge_out_est)
            st.session_state["cost_usd"] = st.session_state.get("cost_usd", 0.0) + cost_inc

            row = {
                "query_id": q["query_id"],
                "query": q["query_text"][:80],
                **{f"ragas_{k}": v for k, v in scored["ragas"]["scores"].items()},
                **{f"deepeval_{k}": v for k, v in scored["deepeval"]["scores"].items()},
                "wall_seconds": scored["wall_seconds"],
            }
            rows.append(row)

            st.session_state["reasons"][q["query_id"]] = {
                "ragas": scored["ragas"]["reasons"],
                "deepeval": scored["deepeval"]["reasons"],
            }
            st.session_state["records"][q["query_id"]] = record

            # Re-render scorecards + table after each row. The placeholders
            # (metrics_box, table_box, scatter_box) are overwritten in place,
            # which is what gives the page its streaming feel.
            df = pd.DataFrame(rows)
            agg = _format_aggregates(df, faith_thresh, relev_thresh)

            with metrics_box.container():
                cols = st.columns(min(5, len(RAGAS_METRICS) + 1))
                for j, m in enumerate(RAGAS_METRICS):
                    delta = (agg.get(f"ragas_{m}", 0) or 0) - (agg.get(f"deepeval_{m}", 0) or 0)
                    cols[j].metric(
                        m,
                        f"R {agg.get(f'ragas_{m}', 0):.2f} / D {agg.get(f'deepeval_{m}', 0):.2f}",
                        f"{delta:+.2f}",
                    )
                cols[-1].metric("pass / total", f"{agg.get('pass_count', 0)}/{agg.get('total', 0)}")

            mask = _pass_mask(df, faith_thresh, relev_thresh)
            df_display = df.copy()
            df_display.insert(2, "pass", mask.map({True: "✅", False: "❌"}))
            table_box.dataframe(df_display, use_container_width=True, hide_index=True)

            # Scatter visualises Ragas-vs-DeepEval agreement on faithfulness.
            # Diagonal dotted line = perfect agreement. Points far from the
            # line are the disagreement audit — the same queries compare.py's
            # report calls out as Top-10 disagreements.
            if len(df) >= 2 and "ragas_faithfulness" in df.columns and "deepeval_faithfulness" in df.columns:
                fig = px.scatter(
                    df,
                    x="ragas_faithfulness",
                    y="deepeval_faithfulness",
                    hover_data=["query_id", "query"],
                    title="Faithfulness: Ragas vs DeepEval (per query)",
                    range_x=[0, 1],
                    range_y=[0, 1],
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
                scatter_box.plotly_chart(fig, use_container_width=True)

            progress.progress((i + 1) / len(sampled))
            _render_cost_meter()

        st.session_state["last_run"] = pd.DataFrame(rows)
        st.success(f"Done. {len(rows)} queries scored.")

# ----------------------------------------------------------------------------
# Drill-down panel
# ----------------------------------------------------------------------------
# This block re-renders on every Streamlit interaction (slider move, dropdown
# select, etc.) using cached scores in st.session_state["last_run"]. That's
# the trick behind threshold sliders flipping pass/fail badges live without
# re-running any LLM calls — we just recompute the mask against the existing
# numeric scores.
if "last_run" in st.session_state and not st.session_state["last_run"].empty:
    st.divider()
    st.subheader("Per-query drill-down")
    df = st.session_state["last_run"].copy()
    mask = _pass_mask(df, faith_thresh, relev_thresh)
    df.insert(2, "pass", mask.map({True: "✅", False: "❌"}))
    st.dataframe(df, use_container_width=True, hide_index=True)

    qids = df["query_id"].tolist()
    selected = st.selectbox("Inspect query", qids, index=0)
    full_record = st.session_state["records"].get(selected)
    reasons = st.session_state["reasons"].get(selected, {"ragas": {}, "deepeval": {}})
    row = df.set_index("query_id").loc[selected]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Question")
        st.write(full_record["query_text"])
        st.subheader("Retrieved contexts")
        for i, c in enumerate(full_record["contexts"]):
            with st.expander(f"Chunk {i + 1}"):
                st.write(c)
        st.subheader("Generated answer")
        st.write(full_record["answer"])
        st.subheader("Reference answer")
        st.write(full_record["ground_truth"])

    with col2:
        st.subheader("Side-by-side scores + judge reasoning")
        for metric in RAGAS_METRICS:
            st.markdown(f"**{metric}**")
            mc1, mc2 = st.columns(2)
            with mc1:
                rval = row.get(f"ragas_{metric}", float("nan"))
                st.metric("Ragas", f"{rval:.2f}" if not (rval is None or math.isnan(rval)) else "—")
                with st.expander("Ragas judge reasoning"):
                    st.write(reasons.get("ragas", {}).get(metric) or "(none)")
            with mc2:
                dval = row.get(f"deepeval_{metric}", float("nan"))
                st.metric("DeepEval", f"{dval:.2f}" if not (dval is None or math.isnan(dval)) else "—")
                with st.expander("DeepEval judge reasoning"):
                    st.write(reasons.get("deepeval", {}).get(metric) or "(none)")

        if "deepeval_g_eval_custom" in df.columns:
            st.markdown("**G-Eval custom criterion (DeepEval only)**")
            gv = row.get("deepeval_g_eval_custom", float("nan"))
            st.metric("g_eval_custom", f"{gv:.2f}" if not (gv is None or math.isnan(gv)) else "—")
            with st.expander("G-Eval reasoning"):
                st.write(reasons.get("deepeval", {}).get("g_eval_custom") or "(none)")

elif not run_clicked:
    st.info(
        "Set knobs in the sidebar, then click **▶ Run sample**.\n\n"
        "The sample is drawn deterministically (seed 42) from the cached "
        "`results/golden_set.jsonl`."
    )
