"""Per-sample Ragas + DeepEval scoring helpers, run in parallel.

This is the core of the demo: for one (query, contexts, answer, reference)
record, we score it with **both** libraries on the **same** sample so the
comparison is apples-to-apples.

Why this matters:
  - The Streamlit app calls `score_both_parallel` per query so the table can
    stream live. Running the libs sequentially would double the wall-clock
    per row, which kills the live-demo feel.
  - Both libs' top-level scoring entry points are sync; we run them in
    parallel via a ThreadPoolExecutor so they overlap.

Three notable pragmas, each documented inline below and in the gap log:
  1. We use the deprecated `ragas.metrics` (module-style) classes instead of
     the new `ragas.metrics.collections` because the new ones require an
     InstructorLLM, not the LangchainLLMWrapper the tutorial uses.
  2. Ragas's per-metric reasoning lives on `result.ragas_traces` — a
     tree of ChainRun objects, not a `metric.reason` attribute. We walk
     the tree to extract per-claim reasoning for each metric. See
     `_extract_ragas_reasons` below.
  3. The judge model has a 404-safe fallback path. The tutorial pins
     `gpt-5.4`; if that name disappears, we fall back to `gpt-4o`.
"""
from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Suppress the deprecation noise that comes from intentionally importing the
# older module-style Ragas metrics — see results/tutorial_gaps.md. Without
# this filter, every Streamlit re-run dumps four DeprecationWarnings into the
# console.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
# Same idea for DeepEval's LLMTestCaseParams (preferred name is now
# SingleTurnParams). Kept for tutorial fidelity.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="deepeval")

# --- Ragas: deprecated module-style metrics ---------------------------------
# Why deprecated style: ragas 0.4.3's `ragas.metrics.collections` requires
# `InstructorLLM` (built via `llm_factory`). The tutorial — and most of the
# Ragas docs floating around the internet — use `LangchainLLMWrapper`, which
# is *only* compatible with the deprecated path. Switching to the new path
# would break tutorial fidelity. See results/tutorial_gaps.md for details.
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# --- DeepEval: stable API ---------------------------------------------------
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval,
)
from deepeval.models import GPTModel, AnthropicModel

from config import settings, JUDGE_FALLBACK


# Track which judge model was actually used by the most recent
# score_both_parallel call. The Streamlit cost meter reads this so the
# presenter can see whether the configured judge was honoured or whether the
# fallback kicked in.
LAST_JUDGE_USED: str = ""


# ----------------------------------------------------------------------------
# Judge construction with provider routing
# ----------------------------------------------------------------------------
# Each library has its own LLM-wrapper convention. We branch on the model
# name prefix because Anthropic's models start with "claude" and OpenAI's
# start with "gpt" or (for reasoning models) "o".

def _make_ragas_judge(model_name: str):
    """Return a Ragas-compatible judge wrapper.

    We pass temperature=0 for determinism. Note: this rules out OpenAI's
    reasoning-class minis (gpt-5-mini, gpt-5-nano) which only accept the
    default temperature — Ragas itself forces a low temperature on its
    internal calls regardless of what we set here, so those models
    return NaN scores when used as Ragas judges. They are excluded from
    JUDGE_CHOICES on purpose; see config.py.
    """
    if model_name.startswith("gpt") or model_name.startswith("o"):
        return LangchainLLMWrapper(
            ChatOpenAI(model=model_name, temperature=0, api_key=settings.openai_api_key)
        )
    return LangchainLLMWrapper(
        ChatAnthropic(model=model_name, temperature=0, api_key=settings.anthropic_api_key)
    )


def _make_deepeval_judge(model_name: str):
    """Return a DeepEval-compatible judge wrapper.

    DeepEval reads its API keys from the environment, hence the
    `os.environ.setdefault` calls — they're a no-op when the env is already
    populated (which is the normal case via .env), and a safety net for
    tests that construct the judge without a full env.
    """
    if model_name.startswith("gpt") or model_name.startswith("o"):
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        return GPTModel(model=model_name)
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    return AnthropicModel(model=model_name)


def _is_model_unavailable_error(exc: BaseException) -> bool:
    """Heuristic: did this exception say the model doesn't exist?

    We catch a model-not-found and substitute the fallback. We do NOT catch
    auth errors, network errors, rate limits, or parameter errors —
    those should surface so the user can fix them.
    """
    msg = (str(exc) or "").lower()
    cls = type(exc).__name__.lower()
    return any(
        s in msg
        for s in (
            "model_not_found",
            "not_found_error",
            "404",
            "does not exist",
            "is not a known",
            "no such model",
        )
    ) or cls == "notfounderror"


def _resolve_judge_model(requested: str) -> str:
    """Resolve the judge model, falling back to JUDGE_FALLBACK on a 404.

    Caches the resolution in the module-level LAST_JUDGE_USED so subsequent
    calls within the same process don't re-probe. If the user changes the
    judge in the Streamlit sidebar, the cache misses on the new name and we
    re-probe — that's the correct behaviour.

    The probe sends a tiny "ok" message with no token cap because modern
    OpenAI reasoning models reject `max_tokens` and require
    `max_completion_tokens`. Skipping the cap entirely sidesteps the issue.
    """
    global LAST_JUDGE_USED
    if LAST_JUDGE_USED:
        # Reuse cached resolution if it matches the current request.
        if LAST_JUDGE_USED == requested or (
            LAST_JUDGE_USED == JUDGE_FALLBACK and requested == JUDGE_FALLBACK
        ):
            return LAST_JUDGE_USED

    try:
        if requested.startswith("gpt") or requested.startswith("o"):
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=settings.openai_api_key)
            client.chat.completions.create(
                model=requested,
                messages=[{"role": "user", "content": "ok"}],
            )
        else:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=settings.anthropic_api_key)
            client.messages.create(
                model=requested,
                max_tokens=1,
                messages=[{"role": "user", "content": "ok"}],
            )
        LAST_JUDGE_USED = requested
        return requested
    except Exception as exc:
        if _is_model_unavailable_error(exc):
            print(
                f"[scoring] judge model '{requested}' unavailable "
                f"({type(exc).__name__}); falling back to '{JUDGE_FALLBACK}'."
            )
            LAST_JUDGE_USED = JUDGE_FALLBACK
            return JUDGE_FALLBACK
        # Auth, network, parameter errors etc. propagate.
        raise


# ----------------------------------------------------------------------------
# Ragas
# ----------------------------------------------------------------------------
# Map from a metric name to the ChainRun trace nodes that hold its reasoning.
# Each entry maps the metric to a "leaf prompt name" inside the trace tree
# that contains the structured judge output. Discovered by probing
# `result.ragas_traces` against ragas 0.4.3.
_RAGAS_REASON_LEAVES: dict[str, str] = {
    "faithfulness": "n_l_i_statement_prompt",
    "answer_relevancy": "response_relevance_prompt",
    "context_precision": "context_precision_prompt",
    "context_recall": "context_recall_classification_prompt",
}


def _format_faithfulness_reason(leaf_outputs: list) -> str:
    """Faithfulness leaves carry per-statement verdicts with reasons."""
    parts: list[str] = []
    for block in leaf_outputs:
        statements = (block or {}).get("statements", []) if isinstance(block, dict) else []
        for s in statements:
            verdict = s.get("verdict", "?")
            mark = "✓" if verdict == 1 else ("✗" if verdict == 0 else "?")
            parts.append(f"{mark} {s.get('statement', '')}\n   reason: {s.get('reason', '').strip()}")
    return "\n".join(parts) if parts else "(no per-statement output)"


def _format_answer_relevancy_reason(leaf_outputs: list) -> str:
    """AnswerRelevancy back-translates the answer into questions; the score
    is the average cosine similarity to the original. The traces give us
    the back-translated questions plus a 'noncommittal' flag."""
    questions: list[str] = []
    noncommittal = False
    for block in leaf_outputs:
        if not isinstance(block, dict):
            continue
        q = block.get("question")
        if q:
            questions.append(q)
        if block.get("noncommittal"):
            noncommittal = True
    parts = [f"Back-translated questions from the answer (cosine-similar to original = score):"]
    parts += [f"  • {q}" for q in questions]
    if noncommittal:
        parts.append("Judge flagged the answer as non-committal (auto-zero).")
    return "\n".join(parts) if questions else "(no back-translated questions)"


def _format_context_precision_reason(leaf_outputs: list) -> str:
    """ContextPrecision asks per-context: was this chunk useful for the
    answer? Returns one (verdict, reason) per chunk."""
    parts: list[str] = []
    for i, block in enumerate(leaf_outputs, start=1):
        if not isinstance(block, dict):
            continue
        verdict = block.get("verdict", "?")
        mark = "✓ useful" if verdict == 1 else ("✗ not useful" if verdict == 0 else "?")
        parts.append(f"Chunk {i}: {mark}\n   reason: {block.get('reason', '').strip()}")
    return "\n".join(parts) if parts else "(no per-chunk verdicts)"


def _format_context_recall_reason(leaf_outputs: list) -> str:
    """ContextRecall splits the reference into atomic statements and asks
    whether each is attributable to the retrieved contexts."""
    parts: list[str] = []
    for block in leaf_outputs:
        classifications = (block or {}).get("classifications", []) if isinstance(block, dict) else []
        for c in classifications:
            attr = c.get("attributed", "?")
            mark = "✓ attributed" if attr == 1 else ("✗ not attributed" if attr == 0 else "?")
            parts.append(f"{mark}: {c.get('statement', '')}\n   reason: {c.get('reason', '').strip()}")
    return "\n".join(parts) if parts else "(no per-statement classifications)"


_RAGAS_REASON_FORMATTERS = {
    "faithfulness": _format_faithfulness_reason,
    "answer_relevancy": _format_answer_relevancy_reason,
    "context_precision": _format_context_precision_reason,
    "context_recall": _format_context_recall_reason,
}


def _to_plain_dict(obj: Any) -> Any:
    """Recursively convert Pydantic objects to plain dicts so the
    per-metric formatters (which assume dict access) can read them.
    Falls through plain types unchanged."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, list):
        return [_to_plain_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return obj


def _extract_ragas_reasons(ragas_traces: dict) -> dict[str, str]:
    """Walk a `result.ragas_traces` tree and return human-readable reasoning
    text per metric.

    The trace tree is structured as:
        evaluation → row 0 → <metric_name> → <prompt_name>(s)
    where the prompt_name leaves carry the structured judge output (as
    Pydantic objects like NLIStatementOutput, ContextRecallClassifications,
    etc.). Multiple leaf nodes can share the same prompt_name (e.g.
    context_precision runs once per chunk), so we collect ALL leaves with
    the matching name, dump them to plain dicts, and pass the list to the
    per-metric formatter.
    """
    reasons: dict[str, str] = {}
    for metric_name, leaf_name in _RAGAS_REASON_LEAVES.items():
        leaf_outputs: list = []
        for run in ragas_traces.values():
            if getattr(run, "name", None) != leaf_name:
                continue
            outputs = run.outputs or {}
            payload = outputs.get("output") if hasattr(outputs, "get") else None
            if payload is None:
                payload = outputs
            payload = _to_plain_dict(payload)
            if isinstance(payload, list):
                leaf_outputs.extend(payload)
            elif payload:
                leaf_outputs.append(payload)
        formatter = _RAGAS_REASON_FORMATTERS.get(metric_name)
        if formatter is None or not leaf_outputs:
            reasons[metric_name] = "(no reasoning captured)"
        else:
            try:
                reasons[metric_name] = formatter(leaf_outputs)
            except Exception as exc:
                reasons[metric_name] = f"(reason format failed: {type(exc).__name__}: {exc})"
    return reasons


def _score_ragas_sync(record: dict[str, Any], judge_model: str) -> dict[str, Any]:
    """Score one record with Ragas's four standard metrics + capture reasoning.

    Uses `evaluate()` rather than per-metric `single_turn_ascore` because
    only `evaluate()` returns an `EvaluationResult` with `ragas_traces` —
    the tree of ChainRun objects that carries the judge's structured
    output (the 'reasoning'). See _extract_ragas_reasons for the tree walk.

    Tutorial fidelity: still builds SingleTurnSample with the exact field
    names the tutorial uses.
    """
    judge = _make_ragas_judge(judge_model)
    # AnswerRelevancy needs an embeddings model. We hard-code text-embedding-
    # 3-small to match the dense embedding used at ingest time — keeps the
    # eval consistent with the index.
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    )

    sample = SingleTurnSample(
        user_input=record["query_text"],
        retrieved_contexts=record["contexts"],
        response=record["answer"],
        reference=record["ground_truth"],
    )

    metric_objs = [
        ("faithfulness", Faithfulness()),
        ("answer_relevancy", AnswerRelevancy()),
        ("context_precision", ContextPrecision()),
        ("context_recall", ContextRecall()),
    ]

    scores: dict[str, float] = {n: float("nan") for n, _ in metric_objs}
    reasons: dict[str, str] = {n: "(no reasoning captured)" for n, _ in metric_objs}

    try:
        ds = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=ds,
            metrics=[m for _, m in metric_objs],
            llm=judge,
            embeddings=embeddings,
            show_progress=False,
        )
        # Pull per-metric scores from the dataframe (one row, one column per metric).
        df = result.to_pandas()
        for name, _ in metric_objs:
            if name in df.columns and len(df) > 0:
                v = df.iloc[0][name]
                try:
                    scores[name] = float(v)
                except (TypeError, ValueError):
                    scores[name] = float("nan")
        # Walk the trace tree to extract reasoning.
        traces = getattr(result, "ragas_traces", {}) or {}
        reasons.update(_extract_ragas_reasons(traces))
    except Exception as exc:
        for name in scores:
            scores[name] = float("nan")
            reasons[name] = f"(scoring failed: {type(exc).__name__}: {exc})"

    return {"scores": scores, "reasons": reasons}


# ----------------------------------------------------------------------------
# DeepEval
# ----------------------------------------------------------------------------

def _score_deepeval_sync(
    record: dict[str, Any],
    judge_model: str,
    custom_geval_criterion: str | None = None,
) -> dict[str, Any]:
    """Score one record with DeepEval's four standard metrics + optional G-Eval.

    G-Eval is the headline DeepEval differentiator: a natural-language
    criterion the judge applies as if it were a custom metric. We only add
    it when `custom_geval_criterion` is non-empty.
    """
    judge = _make_deepeval_judge(judge_model)
    case = LLMTestCase(
        input=record["query_text"],
        actual_output=record["answer"],
        retrieval_context=record["contexts"],
        expected_output=record["ground_truth"],
    )

    # async_mode=False because we're already running this whole function
    # inside a worker thread (see score_both_parallel). Letting DeepEval
    # spin up its own asyncio loop here would conflict.
    metrics: list[tuple[str, Any]] = [
        ("faithfulness", FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
        ("answer_relevancy", AnswerRelevancyMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
        ("context_precision", ContextualPrecisionMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
        ("context_recall", ContextualRecallMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
    ]

    if custom_geval_criterion:
        metrics.append(
            (
                "g_eval_custom",
                GEval(
                    name="custom_criterion",
                    criteria=custom_geval_criterion,
                    # These params tell G-Eval which fields of the test case
                    # to feed the judge. INPUT + ACTUAL_OUTPUT + RETRIEVAL_CONTEXT
                    # is the right combo for criteria like "every numerical
                    # claim must be supported by a quoted span from context".
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.RETRIEVAL_CONTEXT,
                    ],
                    model=judge,
                    threshold=0.5,
                    async_mode=False,
                ),
            )
        )

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for name, metric in metrics:
        try:
            metric.measure(case)
            scores[name] = float(metric.score) if metric.score is not None else float("nan")
            # `metric.reason` is the killer feature — DeepEval surfaces the
            # judge's natural-language explanation per metric. The drill-down
            # panel renders this in expander widgets next to each score.
            reasons[name] = (getattr(metric, "reason", "") or "").strip()
        except Exception as exc:
            scores[name] = float("nan")
            reasons[name] = f"(scoring failed: {type(exc).__name__}: {exc})"

    return {"scores": scores, "reasons": reasons}


# ----------------------------------------------------------------------------
# Combined parallel call
# ----------------------------------------------------------------------------

def score_both_parallel(
    record: dict[str, Any],
    judge_model: str,
    custom_geval_criterion: str | None = None,
) -> dict[str, Any]:
    """Run Ragas and DeepEval in parallel for one record.

    Both lib entry points are sync; we hand each to its own worker thread
    and join. The two libs do roughly the same amount of LLM work per
    record, so running them concurrently roughly halves the per-row
    wall-clock vs running sequentially.

    Returns:
      {
        "ragas":     {"scores": {...}, "reasons": {...}},
        "deepeval":  {"scores": {...}, "reasons": {...}},
        "wall_seconds":     float,
        "judge_model_used": str,   # may differ from `judge_model` on fallback
      }
    """
    resolved_judge = _resolve_judge_model(judge_model)
    start = time.time()

    with ThreadPoolExecutor(max_workers=2) as pool:
        ragas_future = pool.submit(_score_ragas_sync, record, resolved_judge)
        de_future = pool.submit(
            _score_deepeval_sync, record, resolved_judge, custom_geval_criterion
        )
        ragas_result = ragas_future.result()
        de_result = de_future.result()

    return {
        "ragas": ragas_result,
        "deepeval": de_result,
        "wall_seconds": time.time() - start,
        "judge_model_used": resolved_judge,
    }
