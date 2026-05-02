"""Per-sample Ragas + DeepEval scoring helpers, run in parallel.

This is the core of the demo: for one (query, contexts, answer, reference)
record, we score it with **both** libraries on the **same** sample so the
comparison is apples-to-apples.

Why this matters:
  - The Streamlit app calls `score_both_parallel` per query so the table can
    stream live. Running the libs sequentially would double the wall-clock
    per row, which kills the live-demo feel.
  - Ragas is async-native; DeepEval is sync-native. We run Ragas in the main
    event loop and DeepEval in a worker thread so the two overlap.

Three notable pragmas, each documented inline below and in the gap log:
  1. We use the deprecated `ragas.metrics` (module-style) classes instead of
     the new `ragas.metrics.collections` because the new ones require an
     InstructorLLM, not the LangchainLLMWrapper the tutorial uses.
  2. Ragas in 0.4.x does not expose per-metric reasoning. We surface a
     placeholder string so the drill-down panel still has a slot for it,
     keeping the layout symmetric with DeepEval.
  3. The judge model has a 404-safe fallback path. The tutorial pins
     `gpt-5.4`; if that name disappears, we fall back to `gpt-4o`.
"""
from __future__ import annotations

import asyncio
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
from ragas import SingleTurnSample
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
    """Return a Ragas-compatible judge wrapper."""
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

async def _score_ragas_async(record: dict[str, Any], judge_model: str) -> dict[str, Any]:
    """Score one record with Ragas's four standard metrics.

    Tutorial fidelity: builds a SingleTurnSample with exactly the field names
    the tutorial uses (`user_input`, `retrieved_contexts`, `response`,
    `reference`) and assigns the judge LLM via `metric.llm = ...` per the
    tutorial's pattern.
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

    # Each metric is a stateless object with a couple of attributes set
    # before we call `single_turn_ascore`. Building them fresh per record is
    # cheap (the heavy work is the LLM call inside ascore).
    metrics = [
        ("faithfulness", Faithfulness()),
        ("answer_relevancy", AnswerRelevancy()),
        ("context_precision", ContextPrecision()),
        ("context_recall", ContextRecall()),
    ]

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}

    for name, metric in metrics:
        metric.llm = judge
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
        # Ragas 0.4.x does not expose per-metric reasoning. The tutorial
        # speculates a `single_turn_ascore_with_reason` API that doesn't
        # exist — see results/tutorial_gaps.md. We fall back to scores-only.
        try:
            score = await metric.single_turn_ascore(sample)
        except Exception as exc:
            scores[name] = float("nan")
            reasons[name] = f"(scoring failed: {type(exc).__name__}: {exc})"
            continue
        try:
            scores[name] = float(score)
        except (TypeError, ValueError):
            scores[name] = float("nan")
        reasons[name] = "(reason not exposed by this Ragas metric version)"

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
    """Run Ragas (async) and DeepEval (sync, in a thread) in parallel.

    The two libs do roughly the same amount of LLM work per record, so
    running them concurrently roughly halves the per-row wall-clock vs
    running sequentially.

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

    # Fresh event loop per call so we don't conflict with any caller's loop
    # (e.g. Streamlit doesn't have one). The single-worker pool is enough
    # because we only need one DeepEval call to overlap with one Ragas call.
    loop = asyncio.new_event_loop()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            de_future = pool.submit(
                _score_deepeval_sync, record, resolved_judge, custom_geval_criterion
            )
            ragas_result = loop.run_until_complete(
                _score_ragas_async(record, resolved_judge)
            )
            de_result = de_future.result()
    finally:
        loop.close()

    return {
        "ragas": ragas_result,
        "deepeval": de_result,
        "wall_seconds": time.time() - start,
        "judge_model_used": resolved_judge,
    }
