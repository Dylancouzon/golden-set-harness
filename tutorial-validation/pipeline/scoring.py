"""Per-sample Ragas + DeepEval scoring helpers, run in parallel.

Two helpers, then one combined entry point:

  - `_score_ragas_async`  — async, scores 4 metrics via the deprecated module-style
                             Ragas metrics (collections-style requires an InstructorLLM
                             from llm_factory, see results/tutorial_gaps.md).
  - `_score_deepeval_sync` — sync, scores 4 standard metrics + optional G-Eval.
                             Surfaces `metric.reason` for the drill-down.
  - `score_both_parallel`  — runs both libs in parallel for one record.

Returns shape: {"ragas": {scores, reasons}, "deepeval": {scores, reasons, judge_usage},
                "wall_seconds": float, "judge_model_used": str}
"""
from __future__ import annotations

import asyncio
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Suppress Ragas deprecation noise — see tutorial_gaps.md for the reason.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
# DeepEval also flags LLMTestCaseParams; preserved for tutorial fidelity.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="deepeval")

# Ragas (deprecated module-style — see tutorial_gaps.md)
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

# DeepEval
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


# Track which judge model the most recent call actually used. The Streamlit
# cost meter reads this to surface fallback events to the presenter.
LAST_JUDGE_USED: str = ""

# Track approximate judge token usage. DeepEval doesn't expose token counts on
# its metric objects, so we estimate via tiktoken-equivalent character heuristics
# at the boundary. We only count what we can — used as directional, not exact.
_JUDGE_TOKEN_TOTALS = {"input_tokens": 0, "output_tokens": 0}


def reset_judge_token_counter() -> None:
    _JUDGE_TOKEN_TOTALS["input_tokens"] = 0
    _JUDGE_TOKEN_TOTALS["output_tokens"] = 0


def get_judge_token_totals() -> dict[str, int]:
    return dict(_JUDGE_TOKEN_TOTALS)


# ----------------------------------------------------------------------------
# Judge construction with model fallback
# ----------------------------------------------------------------------------

def _make_ragas_judge(model_name: str):
    if model_name.startswith("gpt") or model_name.startswith("o"):
        return LangchainLLMWrapper(
            ChatOpenAI(model=model_name, temperature=0, api_key=settings.openai_api_key)
        )
    return LangchainLLMWrapper(
        ChatAnthropic(model=model_name, temperature=0, api_key=settings.anthropic_api_key)
    )


def _make_deepeval_judge(model_name: str):
    if model_name.startswith("gpt") or model_name.startswith("o"):
        # DeepEval accepts `model=<name>` for OpenAI; api key from env.
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        return GPTModel(model=model_name)
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    return AnthropicModel(model=model_name)


def _is_model_unavailable_error(exc: BaseException) -> bool:
    """Heuristic: judge model 404 / not-found / unsupported."""
    msg = (str(exc) or "").lower()
    cls = type(exc).__name__.lower()
    return any(
        s in msg
        for s in ("model_not_found", "not_found_error", "404", "does not exist", "is not a known", "no such model")
    ) or cls == "notfounderror"


def _resolve_judge_model(requested: str) -> str:
    """Cache-aware judge resolution. Performs a tiny probe call once; if the
    requested model 404s, swap to JUDGE_FALLBACK and remember the choice."""
    global LAST_JUDGE_USED
    if LAST_JUDGE_USED:
        # Already resolved this session; reuse.
        # If the env-requested model differs from what we resolved, attempt the
        # new one (the user may have switched in the UI).
        if LAST_JUDGE_USED == requested or LAST_JUDGE_USED == JUDGE_FALLBACK and requested == JUDGE_FALLBACK:
            return LAST_JUDGE_USED

    # Probe with a one-token completion to see if the model exists.
    try:
        if requested.startswith("gpt") or requested.startswith("o"):
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=settings.openai_api_key)
            client.chat.completions.create(
                model=requested,
                max_tokens=1,
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
                f"[scoring] judge model '{requested}' unavailable ({type(exc).__name__}); "
                f"falling back to '{JUDGE_FALLBACK}'."
            )
            LAST_JUDGE_USED = JUDGE_FALLBACK
            return JUDGE_FALLBACK
        # Other errors (network, auth) propagate.
        raise


# ----------------------------------------------------------------------------
# Ragas
# ----------------------------------------------------------------------------

async def _score_ragas_async(record: dict[str, Any], judge_model: str) -> dict[str, Any]:
    judge = _make_ragas_judge(judge_model)
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    )

    sample = SingleTurnSample(
        user_input=record["query_text"],
        retrieved_contexts=record["contexts"],
        response=record["answer"],
        reference=record["ground_truth"],
    )

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
        # Ragas 0.4.x: no per-metric reason API. See tutorial_gaps.md.
        try:
            score = await metric.single_turn_ascore(sample)
        except Exception as exc:
            score = float("nan")
            reasons[name] = f"(scoring failed: {type(exc).__name__}: {exc})"
            scores[name] = score
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
    judge = _make_deepeval_judge(judge_model)
    case = LLMTestCase(
        input=record["query_text"],
        actual_output=record["answer"],
        retrieval_context=record["contexts"],
        expected_output=record["ground_truth"],
    )

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
    """Run Ragas (async) and DeepEval (sync, in a thread) in parallel for one record.

    Returns:
      {"ragas": {scores, reasons},
       "deepeval": {scores, reasons},
       "wall_seconds": float,
       "judge_model_used": str}
    """
    resolved_judge = _resolve_judge_model(judge_model)
    start = time.time()

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
