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

### `LLMTestCaseParams` is deprecated in DeepEval 3.9.9 (use `SingleTurnParams`)
**Where:** `pipeline/scoring.py` G-Eval block in instructions
**What was missing:** The tutorial imports `LLMTestCaseParams` from `deepeval.test_case`. DeepEval emits a `DeprecationWarning` recommending `SingleTurnParams` instead.
**What I had to invent:** Kept `LLMTestCaseParams` to preserve tutorial fidelity but warn-suppressed the deprecation at import time so the demo console isn't noisy.
**Tutorial fix suggestion:** Update to `from deepeval.test_case import SingleTurnParams` (same usage) before the next DeepEval major.

