# Ragas vs DeepEval — comparison report

_Generated from 10 queries._


## 1. Aggregate scores

| Metric | Ragas | DeepEval | Δ (R-D) |
|---|---|---|---|
| faithfulness | 0.865 | 0.959 | -0.094 |
| answer_relevancy | 0.584 | 0.938 | -0.354 |
| context_precision | 0.619 | 0.630 | -0.011 |
| context_recall | 0.533 | 0.800 | -0.267 |

## 2. Per-query correlation between libraries

Both libraries should agree on directional quality. Pearson > 0.6 is a healthy floor for faithfulness and answer_relevancy; below that, the libraries are measuring different things.

| Metric | Pearson r | Spearman ρ |
|---|---|---|
| faithfulness | -0.028 | 0.130 |
| answer_relevancy | -0.053 | 0.309 |
| context_precision | 0.831 | 0.815 |
| context_recall | 0.634 | 0.639 |

## 3. Correlation against manual ratings

_No `results/manual_ratings.jsonl` found. Run `python manual_judge.py` to enable this section. Without it, the rubric falls back to per-query correlation as a weaker proxy for trustworthiness._

## 4. Top-10 faithfulness disagreements

Eyeballing 3 of these answers the question 'when they disagree, which one is right?' — leave a 1-line note per inspection.

| query_id | query (truncated) | Ragas | DeepEval | |Δ| |
|---|---|---|---|---|
| 8271 | Income in zero-interest environment | 0.727 | 1.000 | 0.273 |
| 2568 | How to pay with cash when car shopping? | 0.789 | 1.000 | 0.211 |
| 2388 | Do financial advisors get better deals on mortgages? | 0.923 | 0.714 | 0.209 |
| 3149 | Tips for insurance coverage for one-man-teams | 0.846 | 1.000 | 0.154 |
| 3512 | As an employee, when is it inappropriate to request to see y | 0.850 | 1.000 | 0.150 |
| 1281 | How FTB and IRS find mistakes in amended tax returns? Are th | 0.864 | 1.000 | 0.136 |
| 4105 | As an investor what are side effects of Quantitative Easing  | 0.750 | 0.875 | 0.125 |
| 7311 | Finance, Social Capital IPOA.U | 0.941 | 1.000 | 0.059 |
| 4179 | Why could the serious financial woes of some EU member state | 0.960 | 1.000 | 0.040 |
| 6867 | Will there always be somebody selling/buying in every stock? | 1.000 | 1.000 | 0.000 |

## 5. Runtime + judge model

Token usage is not exposed by either library on its metric objects, so judge-token cost has to be estimated upstream (the Streamlit cost meter does this directionally). Wall-clock comparison below.

| Field | Ragas | DeepEval |
|---|---|---|
| wall seconds (total) | 459.597 | 536.579 |
| wall seconds / query | 45.960 | 53.658 |
| judge model used | gpt-5.4 | gpt-5.4 |

## 6. Library feature matrix

| Capability | Ragas | DeepEval |
|---|---|---|
| Pytest-native API | No | Yes (`assert_test`) |
| Custom metrics via natural language | Limited | Yes (G-Eval) |
| Reference-free metrics | Yes | Yes |
| Built-in dataset/UI | No | Yes (Confident AI cloud) |
| Agent-trajectory metrics | Yes (newer) | Yes |
| Setup LOC for this pipeline | 41 | 52 |
| Judge reasoning exposed per metric | No (not in 0.4.x) | Yes (`metric.reason`) |

## 7. Recommendation rubric

**Lead with Ragas in the tutorial — won 2/5 scored criteria** (Ragas: 2, DeepEval: 0).

| Criterion | Ragas | DeepEval | Winner |
|---|---|---|---|
| Higher correlation with manual faithfulness | — | — | — |
| Higher correlation with manual answer_relevancy | — | — | — |
| Lower setup LOC | 41 | 52 | ragas |
| Lower wall-clock per query | 45.960 | 53.658 | ragas |
| Lower judge-token cost per query (proxy: wall seconds) | see #4 | see #4 | skipped |
| Catches a failure mode via G-Eval | — | — | tie |

_The 5th criterion (judge-token cost) is folded into wall-clock since neither library exposes per-metric token counts. The 6th (G-Eval failure-mode catch) is DeepEval-only by definition; it scores +1 only when at least one query had a G-Eval custom score < 0.5 while standard metrics scored ≥ 0.7._

## 8. Reference-answer quality caveat

The golden-set references were LLM-generated (constrained to the relevant FiQA passages, but still LLM output). This is the weakest link in the methodology — `context_precision` and `context_recall` are scored against text that itself wasn't human-validated. Scores can shift by a few points depending on the reference-generation prompt. **Treat the recommendation as directional, not definitive, and re-run with human-written references if the call is close** (margin ≤ 1 criterion).

## 9. Tutorial-update suggestions

Top gaps logged during the build (full text in `results/tutorial_gaps.md`):



### Tutorial pins `gpt-5.4` as the judge model with no fallback
**Where:** config block (`JUDGE_MODEL=gpt-5.4`)
**What was missing:** The tutorial pins a model ID that may not be available at OpenAI today. The reader has no signal that a fallback is needed if the API 404s.
**What I had to invent:** A `JUDGE_FALLBACK = "gpt-4o"` constant in `config.py` and a try/except in `pipeline/scoring.py` that catches NotFoundError on the first judge call and re-runs with the fallback. The active judge is surfaced in the Streamlit cost meter.
**Tutorial fix suggestion:** Note that judge model IDs churn — recommend `gpt-4o` as a stable default and explain how to swap.

### `from ragas.metrics.collections import faithfulness, answer_relevancy, ...` doesn't exist in ragas 0.4.3
**Where:** `pipeline/scoring.py` import block in instructions
**What was missing:** The tutorial imports lowercase singletons from `ragas.metrics.collections`, but in ragas 0.4.3 the `collections` submodule exports *classes* (`Faithfulness`, `AnswerRelevancy`, ...) that must be instantiated with `llm=...` at construction time. The lowercase singleton form only exists at the deprecated `ragas.metrics` module path.
**What I had to invent:** Use the deprecated module-style (`from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall`) which still accepts `metric.llm = LangchainLLMWrapper(...)` after construction. Suppressed the deprecation warning at the import boundary.
**Tutorial fix suggestion:** Either move to the modern collections classes with `llm_factory(...)` (and `InstructorLLM`) or pin to the deprecated module-style path explicitly with a version note.

### Ragas 0.4.3 collections metrics require `InstructorLLM`, not `LangchainLLMWrapper`
**Where:** `pipeline/scoring.py` — `_ragas_judge` returns `LangchainLLMWrapper(ChatOpenAI(...))`
**What was missing:** Tutorial assumes `LangchainLLMWrapper` is the canonical way to wrap a judge LLM for any Ragas metric. In ragas 0.4.3, the new `collections` metrics raise `ValueError: Collections metrics only support modern InstructorLLM` and direct you to `llm_factory('gpt-4o-mini', client=openai_client)`.
**What I had to invent:** Stick with the deprecated module-style metrics, which still accept `LangchainLLMWrapper`. The trade-off: lose the new collections-style features (none of which we need for this demo).
**Tutorial fix suggestion:** Show both forms side-by-side, explain when each applies, and explicitly call out the LangchainLLMWrapper-vs-InstructorLLM split.

### `single_turn_ascore_with_reason` is not exposed by any Ragas metric in 0.4.3
**Where:** `pipeline/scoring.py` async loop in instructions
**What was missing:** The instructions speculate the method may exist (`try ... except AttributeError`). It does not exist on either deprecated or new collections metrics. There is no public API to retrieve per-metric reasoning out of Ragas in this version.
**What I had to invent:** Permanent fallback path: call `single_turn_ascore` and store `"(reason not exposed by this Ragas metric version)"` as the reason. The DeepEval drill-down panel still surfaces real judge reasoning, which is the demo's main asymmetry pitch.
**Tutorial fix suggestion:** Don't claim Ragas reasoning is available. Either drop the reason column on the Ragas side or document that reasoning is DeepEval-only on the live demo.

### Qdrant rejects empty `points=[]` upserts as "Bad request: Empty update request"
**Where:** `ingest.py` end-of-run flush
**What was missing:** Tutorial recommends a final `client.upsert(points=[], wait=True)` as a "flush" to ensure all pending writes have landed before reporting the count. Qdrant 1.17+ rejects this with `400 Bad request: Empty update request`. Ingest completes correctly but exits with a 400 stack trace, which looks like failure to a casual reader.
**What I had to invent:** Replace the empty-points flush with a `client.count(..., exact=True)` call — that forces a sync round-trip and waits for pending writes without sending an empty body.
**Tutorial fix suggestion:** Either drop the explicit flush (the synchronous count check at the end is enough), or document that the Qdrant API requires non-empty point lists.

### Modern OpenAI models reject `max_tokens` — needs `max_completion_tokens`
**Where:** `pipeline/generation.py` — OpenAI generator branch
**What was missing:** Tutorial pins gpt-5.4 as a generator option and uses `max_tokens=512` against the OpenAI Chat Completions API. Reasoning-class models (gpt-5+, o-series) reject `max_tokens` with `BadRequestError: Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.` LangChain/DeepEval handle this internally, but the raw OpenAI client used in the generator does not.
**What I had to invent:** Try/except in `generate()` that retries with `max_completion_tokens=512` when the API rejects `max_tokens`. Also dropped the token cap from the judge-model probe in `scoring.py` since it doesn't need one.
**Tutorial fix suggestion:** Either default to a model that accepts `max_tokens` (e.g. gpt-4o), or branch on model name and use `max_completion_tokens` for gpt-5+/o-series.

### Tutorial's `query_vector` field on golden-set entries can't survive an embedding-model switch
**Where:** Eval entry shape (`query_id` / `query_text` / `query_vector` / `labels` / `ground_truth`)
**What was missing:** Tutorial pre-computes a `query_vector` and stores it on each golden-set entry. That ties the golden set to a specific embedding model — switching embedding models invalidates every cached vector and forces a re-embed. The demo's design intent is to let the presenter try different models cheaply, which is fundamentally incompatible with a per-entry pinned vector.
**What I had to invent:** Drop the `query_vector` field from golden-set entries. The retrieval module computes the dense vector at query time using whatever embedding model the active config specifies. Cost: one extra OpenAI call per query (negligible at demo scale).
**Tutorial fix suggestion:** Either drop the `query_vector` field (recommended), or document that golden-set entries are tied to one embedding model and must be regenerated on switch.

### `LLMTestCaseParams` is deprecated in DeepEval 3.9.9 (use `SingleTurnParams`)
**Where:** `pipeline/scoring.py` G-Eval block in instructions
**What was missing:** The tutorial imports `LLMTestCaseParams` from `deepeval.test_case`. DeepEval emits a `DeprecationWarning` recommending `SingleTurnParams` instead.
**What I had to invent:** Kept `LLMTestCaseParams` to preserve tutorial fidelity but warn-suppressed the deprecation at import time so the demo console isn't noisy.
**Tutorial fix suggestion:** Update to `from deepeval.test_case import SingleTurnParams` (same usage) before the next DeepEval major.


