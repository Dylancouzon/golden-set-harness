# Tutorial gaps — pipeline-output-quality.md

Each entry records a place where the tutorial was missing detail, ambiguous, or wrong while building this Ragas vs DeepEval comparison demo. The gap log is the actual verification artifact for the tutorial.

Format per entry:

```
### <one-line gap title>
**Where:** <tutorial section / line range>
**What was missing:** <concrete description>
**What I had to invent:** <code or decision>
**Tutorial fix suggestion:** <one sentence>
```

---

### Ragas's internal temperature override blocks reasoning-class minis as judges
**Where:** `pipeline/scoring.py` — `_make_ragas_judge` returns a `LangchainLLMWrapper(ChatOpenAI(...))`
**What was missing:** Tutorial assumes any OpenAI chat model can be a Ragas judge. In Ragas 0.4.x, the metric layer forces `temperature=0.01` on its internal LLM calls regardless of what's set on the underlying ChatOpenAI. OpenAI's reasoning-class minis (`gpt-5-mini`, `gpt-5-nano`) reject any non-default temperature with `BadRequestError: Unsupported value: 'temperature' does not support 0.01`. Net effect: Ragas returns NaN for every metric when one of these models is picked as judge. DeepEval works fine because it uses the OpenAI SDK directly and doesn't force a temperature.
**What I had to invent:** Removed `gpt-5-mini` and `gpt-5-nano` from `JUDGE_CHOICES` in config.py (they remain in `GENERATOR_CHOICES`, where we control the call). The cheap-judge path goes through `gpt-4o-mini` instead, which accepts arbitrary temperatures and is ~30x cheaper than gpt-5.4.
**Tutorial fix suggestion:** Document that not every chat model can serve as a Ragas judge and call out reasoning-class minis as known-incompatible until Ragas exposes a way to skip the temperature override.

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

### Ragas reasoning is available but undiscoverable via the per-sample API
**Where:** `pipeline/scoring.py` Ragas scoring path
**What was missing:** The instructions imply Ragas may expose reasoning via `single_turn_ascore_with_reason`. It does not — there is no such method, and the metric instances themselves carry no `reason` attribute or `score_with_reason` method after `single_turn_ascore`. **However**, when scoring via `evaluate()` (not the per-sample API), Ragas populates `EvaluationResult.ragas_traces` with a tree of `ChainRun` objects whose leaves carry the structured judge output as Pydantic models (`NLIStatementOutput`, `ContextRecallClassifications`, etc.). The reasoning is rich — for Faithfulness, you get verdict + reason per atomic claim; for ContextPrecision, per chunk; for ContextRecall, per reference statement.
**What I had to invent:** Switched the Ragas scoring path to `evaluate()` instead of `single_turn_ascore`, then walk `ragas_traces` and dump the Pydantic leaf payloads to plain dicts so per-metric formatters can render them. See `_extract_ragas_reasons` in `pipeline/scoring.py`. The drill-down panel now shows real Ragas reasoning (per-claim or per-chunk) alongside DeepEval's (per-metric holistic).
**Tutorial fix suggestion:** Document that to surface reasoning out of Ragas, you must use `evaluate()` (not the per-sample API) and walk `result.ragas_traces`. This is undocumented in the official Ragas docs as of 0.4.3 — discoverability is the gap, not capability.

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

