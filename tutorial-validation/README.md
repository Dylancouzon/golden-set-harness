# Tutorial Validation

End-to-end test of the three retrieval-quality tutorials against a live
Qdrant collection. Every code block from each tutorial was exercised; every
result is persisted in `artifacts/` for inspection.

## What's in this directory

| File | Purpose |
|---|---|
| `00_load_corpus.py` | Load 500 SQuAD v1 validation paragraphs, embed with FastEmbed (`BAAI/bge-small-en-v1.5`, 384d cosine), upload to Qdrant as collection `tutorial_validation_wiki`. |
| `01_test_tutorial_1.py` | Exercise Tutorial 1's `avg_precision_at_k` helper with 50 real SQuAD questions as test vectors. |
| `02_test_tutorial_2.py` | Generate synthetic queries via Anthropic, assemble the golden set (tutorial's verbatim assembly loop), build `Qrels` and `Run`, score with `ranx.evaluate()`. |
| `03_test_tutorial_3.py` | Extend the golden set with ground-truth answers, run retrieve-generate-record with Anthropic, score with Ragas (`faithfulness`, `answer_relevancy`, `context_precision`), drill per-query. |
| `artifacts/corpus.json` | The 500-doc corpus indexed into Qdrant. |
| `artifacts/golden_set.json` | 150 synthetic queries with `query_id`, `query_text`, `labels` (`source_doc_id: 1`), `source_doc_title`. Vectors omitted for file-size sanity; re-embed as needed. |
| `artifacts/tutorial_1_results.json` | Average `precision@10` for Tutorial 1. |
| `artifacts/tutorial_2_results.json` | `recall@10`, `MRR`, `NDCG@10` for Tutorial 2. |
| `artifacts/tutorial_3_results.json` | Aggregate Ragas scores + per-query preview for Tutorial 3. |
| `artifacts/tutorial_3_samples.json` | Human-readable dump of the 30 `SingleTurnSample` records Ragas scored. |
| `artifacts/tutorial_3_per_query.csv` | Full per-query Ragas scores for drill-down. |

## Setup

Reuses the harness's existing venv at `../venv/` with these additions installed for the tutorials:

```
ranx==0.3.21
ragas==0.4.3
datasets==4.5.0
eval_type_backport==0.3.1         # ragas 0.4.3 uses `str | Path` syntax, needs backport on Python 3.9
langchain-anthropic==0.3.22       # wrapping Anthropic as Ragas's judge LLM
langchain-huggingface==0.3.1      # HuggingFaceEmbeddings for Ragas's answer_relevancy embeddings
sentence-transformers==5.1.2      # dependency of HuggingFaceEmbeddings
```

Env vars (read from `../.env`): `QDRANT_URL`, `QDRANT_API_KEY`, `ANTHROPIC_API_KEY`.

Run order:

```
python 00_load_corpus.py          # one-time index load
python 01_test_tutorial_1.py
python 02_test_tutorial_2.py      # ~50 Anthropic calls for query generation
python 03_test_tutorial_3.py      # ~30 gen + ~30 gt + ~270 Ragas judge calls
```

---

## Corpus

- Source: `rajpurkar/squad` validation split
- 500 unique paragraph contexts (deduplicated)
- Fields: `doc_id` (0..499), `text`, `title`
- Embedded with `BAAI/bge-small-en-v1.5` (384 dims, cosine)
- Indexed into Qdrant Cloud collection `tutorial_validation_wiki`

---

## Tutorial 1 — Measuring ANN Precision

### What was tested

The tutorial's `avg_precision_at_k` helper (Automate-in-CI-with-Python section), run with 50 distinct SQuAD questions embedded as test vectors. No modifications to the tutorial's code.

### Result

`avg precision@10 = 1.0000`

### Interpretation

On a 500-point index, HNSW with default parameters matches exact kNN exactly across the test set. The code path is technically validated; the score is saturated because the corpus is small enough that approximate search isn't losing anything. In a multi-million-point production index the expected score would be in the 0.95–0.99 range the tutorial describes.

### Tutorial's Web UI section

Not exercised here (requires manual interaction with the dashboard). The prerequisites for the Web UI section are satisfied: the Qdrant collection exists, points are loaded, and the Search Quality tab is the correct path to run the same comparison interactively.

### Observations

- The tutorial's code ran end-to-end without modification.
- `SearchParams(exact=True)` is the correct API and produces the expected exact-kNN result.
- Small-corpus validation doesn't prove tuning behavior (HNSW parameters weren't varied here). A meaningful `m` / `ef_construct` sweep needs a larger index where approximation actually drops precision.

---

## Tutorial 2 — Measuring Retrieval Relevance

### What was tested

The full Generating Queries + Using the Golden Set flow. Steps executed:

1. **LLM-Based Synthetic Generation** — 50 source docs sampled from the corpus, Anthropic called with the tutorial's prompt verbatim, generating 3 queries per doc. Result: 150 synthetic queries with `labels = {source_doc_id: 1}`.
2. **Step 1 (Load and assemble)** — the tutorial's verbatim assembly loop produces `golden_set` entries with `query_id`, `query_text`, `query_vector`, `labels`.
3. **Step 2 (Build Qrels and Run)** — the tutorial's verbatim `retrieval_run` function + `Qrels` construction.
4. **Step 3 (Compute metrics)** — `evaluate(qrels, run, ["recall@10", "mrr", "ndcg@10"])`.

### Results

```
recall@10 = 0.9600
mrr       = 0.7871
ndcg@10   = 0.8299
```

Written to `artifacts/tutorial_2_results.json`.

### Interpretation

- `recall@10 = 0.96` — for 96% of synthetic queries, the source document appears in the top-10. Expected-high for synthetic queries (they're generated from the source, so they carry strong lexical/semantic signal back to it).
- `mrr = 0.79` — the source document lands at position 1 or 2 for most queries; lower-rank hits pull the harmonic mean down.
- `ndcg@10 = 0.83` — consistent with a mix of top-1 and lower-rank correct retrievals.

The tutorial's pitfall "Synthetic-query unrealism" applies here: these scores are inflated relative to what real user queries would produce because the LLM sees the source doc while writing queries. The tutorial's mitigation advice (compare synthetic vs real-query distributions) is accurate.

### Observations

- `ranx` API (`Qrels`, `Run`, `evaluate`) worked exactly as the tutorial shows.
- ID type coercion: `ranx` requires consistent key types across `Qrels` and `Run`. Our labels use `str` doc IDs while `qdrant-client` returns integer point IDs — needed to cast `p.id` to `str` before building the `Run` dict. Worth a note in the tutorial (or standardize to one type).
- One Numba `NumbaTypeSafetyWarning` from ranx about `uint64 → int64` cast. Harmless, expected.

---

## Tutorial 3 — Evaluating Pipeline Output Quality

### What was tested

The full Wiring + Scoring flow:

1. **Step 1 (Prepare the evaluation data)** — subsampled 30 entries from Tutorial 2's golden set, re-embedded `query_text`, synthesized `ground_truth` answers via Anthropic (one answer per query, constrained to the source document).
2. **Step 2 (Define the grounding prompt)** — the tutorial's `PROMPT_TEMPLATE` verbatim.
3. **Step 3 (Run retrieval and generation)** — the tutorial's verbatim `generate_answer` + `build_eval_set` functions.
4. **Scoring with Ragas** — `evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])`, with an Anthropic judge configured explicitly (see "Tutorial gap" below).
5. **Per-query drill** — `scores.to_pandas().nsmallest(10, "faithfulness")`.

### Results

```
faithfulness       = 0.9820
answer_relevancy   = 0.7525
context_precision  = 0.7516
```

Aggregate over 30 samples. Written to `artifacts/tutorial_3_results.json`; full per-query scores in `artifacts/tutorial_3_per_query.csv`.

### Interpretation

- **`faithfulness = 0.98`** — almost every generated answer's claims are supported by the retrieved context. Expected-high: the generator prompt explicitly restricts the model to "only the provided context" and asks it to say so when the context is insufficient, so hallucinations are rare.
- **`answer_relevancy = 0.75`** — answers usually address the question, with some padding or off-topic drift on a subset. The bottom-five-by-faithfulness table shows one query (`what happened in the second half of Super Bowl 50`) hitting 0 on relevancy — the generator responded with "context does not contain the answer", which `answer_relevancy` correctly scores as unrelated to the question.
- **`context_precision = 0.75`** — retrieval is surfacing useful chunks most of the time, with some noise in the top-k. The per-query scores show 0.0 for a few queries where retrieval's top chunks don't match the reference answer.

### Per-query drill worked as advertised

`scores.to_pandas().nsmallest(10, "faithfulness")` returned a tidy DataFrame with one row per query and all three metric columns, exactly as the tutorial promises. Worst-scoring queries are inspectable immediately.

### Tutorial gap — Ragas judge LLM configuration

Ragas 0.4.3 defaults to OpenAI for its judge LLM. The tutorial's verbatim code:

```python
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
```

…raises an authentication error unless `OPENAI_API_KEY` is set. A reader following the tutorial with an Anthropic-only setup needs to wire the judge explicitly:

```python
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper

llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(...))  # answer_relevancy needs embeddings

scores = evaluate(dataset, metrics=[...], llm=llm, embeddings=embeddings)
```

Worth adding to the tutorial as a short note.

### Other observations

- `ragas.metrics import faithfulness, answer_relevancy, context_precision` works but emits a `DeprecationWarning` in Ragas 0.4.3: "will be removed in v1.0. Use `ragas.metrics.collections` instead." The tutorial's import path is on borrowed time.
- Python 3.9 compatibility: Ragas 0.4.3 uses `str | Path` union syntax in some type hints, which Python 3.9's typing evaluator rejects. `pip install eval_type_backport` resolves it. Worth a note that Ragas targets Python 3.10+.
- `max_tokens=512` (as in tutorial) is fine for the generator but the Ragas judge sometimes needs more — occasional `LLMDidNotFinishException` on the first run at 512. Bumping the judge LLM to `max_tokens=2048` eliminated them entirely.
- Embedding wrapper for `answer_relevancy`: `FastEmbedEmbeddings` from `langchain_community` fails a Ragas telemetry validation (its `model` attribute is a `TextEmbedding` object, not a string; Ragas's `EmbeddingUsageEvent` pydantic model rejects it and drops `answer_relevancy` to `NaN` on every sample). Switching to `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) works. Worth flagging as a FastEmbed-specific incompatibility to avoid.
- Ragas's generation+judging took ~2:20 for 30 samples × 3 metrics × ~3 judge calls per metric ≈ 90 evaluation jobs (not counting internal retries). That matches the tutorial's "thousands of judge calls for a 500-query golden set" cost warning.

---

## Recurring points worth feeding back into the tutorials

1. **Tutorial 2:** ID type mismatch — tutorial's synthetic-generation labels use arbitrary doc IDs (strings in most examples) while Qdrant returns the types you uploaded. `ranx` silently misaligns if types differ. A one-line "use consistent ID types across Qrels and Run" note would help.

2. **Tutorial 3:** Explicit judge-LLM configuration. The tutorial assumes the reader either has `OPENAI_API_KEY` set or will configure Ragas themselves. Adding a short "Configuring a non-OpenAI judge" block (Anthropic via `LangchainLLMWrapper` + embeddings via `LangchainEmbeddingsWrapper`) closes the gap.

3. **Tutorial 3:** The `ragas.metrics` import path emits a `DeprecationWarning` in 0.4.3. Not urgent, but worth pinning the Ragas version or updating to `ragas.metrics.collections` when the tutorial gets refreshed.

4. **Tutorial 3:** `max_tokens` on the judge LLM sometimes needs to be higher than the tutorial's `512` default (which is appropriate for the generator). If the tutorial shows a full Ragas config, recommending 1024–2048 on the judge avoids the transient `LLMDidNotFinishException`.

---

## Collection state

The Qdrant collection `tutorial_validation_wiki` is left intact for inspection.

- 500 points
- 384-d cosine
- Payload: `text`, `title`
- IDs: 0..499
