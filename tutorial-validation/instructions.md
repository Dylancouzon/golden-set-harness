 Ragas vs DeepEval — Live Demo Comparing Both Eval Libraries on a Production-Shaped RAG Pipeline

 Context

 This project has two purposes:

 1. Validate the pipeline-output-quality.md tutorial. Build the project by following the tutorial's patterns where possible (PROMPT_TEMPLATE, SingleTurnSample, build_eval_set loop, threshold-based CI gating). Every place the tutorial is
 missing detail, ambiguous, or wrong, log it to results/tutorial_gaps.md. That gap log is the actual verification artifact.
 2. Decide whether the tutorial should lead with Ragas or DeepEval. Run a live, production-shaped RAG pipeline through both libraries, score the same samples, correlate against a small manual-judgment set, and apply an explicit
 recommendation rubric (in compare.py) so the answer is data-backed rather than vibes.

 To do both, build a live Streamlit demo that runs Ragas and DeepEval against the same realistic RAG pipeline, streams per-query scores into a table as they finish, and lets a presenter tweak the pipeline (prompt, k, hybrid/rerank
 toggles, models, thresholds, custom G-Eval criteria) and see scores update.

 The pipeline mimics what a real Qdrant customer ships: hybrid retrieval (dense + sparse), cross-encoder reranking, and an LLM generator grounded by a versioned prompt. Both eval libraries score the same (question, retrieved_context,
 generated_answer, reference) samples so the comparison is apples-to-apples.

 The demo has two modes:
 - Live mode (Streamlit app.py) — small sample (default 20 queries), per-query streaming, sidebar knobs, drill-down on any row. The presenter-facing path.
 - Batch mode (eval_ragas.py, eval_deepeval.py, compare.py) — full 300-query run that produces a static markdown comparison report carrying the lead-library recommendation.

 Ingest and golden-set generation are offline scripts run once. The demo loads a warm Qdrant collection and a cached golden set so it can start scoring immediately.

 The executing Claude Code instance has access to QDRANT_URL, QDRANT_API_KEY, ANTHROPIC_API_KEY, and OPENAI_API_KEY. It does not have access to the tutorial — every detail needed to build the project is included below.

 ---
 High-level decisions (locked)

 ┌─────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
 │      Decision       │                                      Choice                                       │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Corpus              │ BeIR/fiqa (~57k finance Q&A passages, ~6.6k labeled queries with relevance qrels) │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Dense embedding     │ OpenAI text-embedding-3-small (1536 dim)                                          │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Sparse embedding    │ BM25 via fastembed (Qdrant/bm25)                                                  │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Reranker            │ cross-encoder/ms-marco-MiniLM-L-12-v2 (sentence-transformers)                     │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Generator (default) │ claude-sonnet-4-6 (Anthropic) — switchable in the demo                            │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Judge LLM (default) │ gpt-5.4 (OpenAI) — different family from generator; switchable in the demo        │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Live demo sample    │ Slider, default 20 queries (range 10–300), sampled deterministically with seed 42 │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Batch eval sample   │ 300 queries (cost-controlled vs the 6.6k full test split)                         │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ UI                  │ Streamlit                                                                         │
 ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
 │ Demo scope          │ retrieve + generate + score only; ingest and golden set are batch scripts         │
 └─────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

 ---
 Tutorial fidelity + gap log

 Build the project by following the tutorial's patterns. The tutorial's canonical shapes — listed below — must be preserved verbatim. Deviating means the tutorial gave insufficient guidance, which is the gap log's job to capture.

 Required fidelity points:
 - The grounding prompt is exactly the PROMPT_TEMPLATE shown in the tutorial (already pinned in config.DEFAULT_PROMPT_TEMPLATE).
 - Ragas samples are constructed via SingleTurnSample(user_input=..., retrieved_contexts=..., response=..., reference=...).
 - Eval entries follow the query_id / query_text / query_vector / labels / ground_truth shape from the tutorial.
 - Ragas batch scoring uses EvaluationDataset + evaluate(metrics=[faithfulness, answer_relevancy, context_precision]).
 - The CI gate pattern is "set a threshold per metric, fail when any drops below" (the tutorial's wording).

 As you build, keep results/tutorial_gaps.md — append an entry every time the tutorial was insufficient. Format:

 ### <one-line gap title>
 **Where:** <tutorial section / line range>
 **What was missing:** <concrete description>
 **What I had to invent:** <code or decision>
 **Tutorial fix suggestion:** <one sentence>

 Examples of likely gaps to watch for:
 - Tutorial shows client.query_points with a single dense vector; doesn't cover hybrid prefetch syntax with models.Prefetch + Fusion.RRF.
 - Tutorial doesn't cover passing a non-OpenAI judge LLM to Ragas (the in-line comment says "see Ragas docs" — that's a gap).
 - Tutorial doesn't cover the DeepEval setup at all beyond a one-line mention.
 - Tutorial doesn't say what judge_model to use or why family-separation matters for the code (it's in pitfalls but not in the code).
 - Tutorial's query_vector is pre-computed and stored on the entry — doesn't address how to switch embedding models (you'd have to re-embed).

 This gap log is read by the user to decide whether the tutorial needs updates before publishing.

 ---
 Project layout

 .
 ├── .env.example
 ├── README.md
 ├── requirements.txt
 ├── config.py
 ├── ingest.py                     # one-shot: corpus → Qdrant
 ├── golden_set.py                 # one-shot: build golden_set.jsonl
 ├── pipeline/
 │   ├── __init__.py
 │   ├── retrieval.py              # hybrid + rerank, with toggles
 │   ├── generation.py             # Claude/GPT generator with versioned prompt
 │   ├── pipeline.py               # retrieve + generate → eval record
 │   └── scoring.py                # per-sample Ragas + DeepEval helpers (parallel, with reasoning)
 ├── app.py                        # 🎯 Streamlit live demo (the centerpiece)
 ├── manual_judge.py               # CLI to capture human ratings on a 10-query slice
 ├── eval_ragas.py                 # batch Ragas run → JSON + CSV
 ├── eval_deepeval.py              # batch DeepEval run → JSON + CSV
 ├── tests/
 │   └── test_pipeline_deepeval.py # pytest-native CI gate demo
 ├── compare.py                    # batch comparison report + lead-library recommendation
 └── results/
     ├── golden_set.jsonl          # produced by golden_set.py
     ├── eval_records.jsonl        # cached generations (batch)
     ├── manual_ratings.jsonl      # human ratings on 10 queries
     ├── ragas_per_query.csv
     ├── deepeval_per_query.csv
     ├── ragas_scores.json
     ├── deepeval_scores.json
     ├── tutorial_gaps.md          # accumulated tutorial-validation findings
     └── comparison_report.md

 ---
 requirements.txt

 streamlit>=1.40
 qdrant-client>=1.12
 fastembed>=0.4
 openai>=1.50
 anthropic>=0.40
 sentence-transformers>=3.0
 datasets>=3.0
 ragas>=0.2
 deepeval>=2.0
 langchain-openai>=0.2
 langchain-anthropic>=0.2
 pandas>=2.2
 numpy>=2.0
 plotly>=5.24
 python-dotenv>=1.0
 tqdm>=4.66
 pytest>=8.0
 tabulate>=0.9
 scipy>=1.14

 ---
 .env.example

 QDRANT_URL=
 QDRANT_API_KEY=
 ANTHROPIC_API_KEY=
 OPENAI_API_KEY=

 QDRANT_COLLECTION=fiqa_eval
 DENSE_MODEL=text-embedding-3-small
 DENSE_DIM=1536
 SPARSE_MODEL=Qdrant/bm25
 RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
 GENERATOR_MODEL=claude-sonnet-4-6
 JUDGE_MODEL=gpt-5.4

 EVAL_SAMPLE_SIZE=300
 RANDOM_SEED=42
 TOP_K_RETRIEVE=50
 TOP_K_RERANK=10

 ---
 config.py

 import os
 from dataclasses import dataclass
 from dotenv import load_dotenv

 load_dotenv()

 @dataclass(frozen=True)
 class Settings:
     qdrant_url: str = os.environ["QDRANT_URL"]
     qdrant_api_key: str = os.environ["QDRANT_API_KEY"]
     anthropic_api_key: str = os.environ["ANTHROPIC_API_KEY"]
     openai_api_key: str = os.environ["OPENAI_API_KEY"]

     collection: str = os.getenv("QDRANT_COLLECTION", "fiqa_eval")
     dense_model: str = os.getenv("DENSE_MODEL", "text-embedding-3-small")
     dense_dim: int = int(os.getenv("DENSE_DIM", "1536"))
     sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
     reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")

     generator_model: str = os.getenv("GENERATOR_MODEL", "claude-sonnet-4-6")
     judge_model: str = os.getenv("JUDGE_MODEL", "gpt-5.4")

     eval_sample_size: int = int(os.getenv("EVAL_SAMPLE_SIZE", "300"))
     random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
     top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "50"))
     top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "10"))

 # Available models for live demo dropdowns
 GENERATOR_CHOICES = ["claude-sonnet-4-6", "claude-opus-4-7", "gpt-5.4"]
 JUDGE_CHOICES = ["gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-7"]

 DEFAULT_PROMPT_TEMPLATE = """You are answering questions using retrieved source material.

 Answer the question below using only the provided context.
 If the context does not contain the answer, say so explicitly.
 Do not rely on outside knowledge.

 Context:
 {retrieved_context}

 Question:
 {query_text}
 """

 settings = Settings()

 ---
 ingest.py — corpus → Qdrant (one-shot batch)

 Builds a single Qdrant collection with named vectors: one dense, one sparse. Each point's payload carries text, title, and doc_id.

 Steps:
 1. Load BeIR/fiqa corpus from HuggingFace datasets.
 2. Create the Qdrant collection with named vectors:
   - dense: 1536-dim, cosine distance
   - sparse: sparse vector config with Modifier.IDF (BM25 needs IDF aggregation across the corpus)
 3. Embed in batches of 256:
   - Dense: OpenAI text-embedding-3-small
   - Sparse: fastembed.SparseTextEmbedding("Qdrant/bm25")
 4. Upsert in batches of 256 using client.upsert with models.PointStruct. wait=False for throughput, final flush with wait=True.

 Collection creation:

 from qdrant_client import QdrantClient, models

 client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60)

 client.create_collection(
     collection_name=settings.collection,
     vectors_config={
         "dense": models.VectorParams(size=settings.dense_dim, distance=models.Distance.COSINE),
     },
     sparse_vectors_config={
         "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
     },
 )

 Upsert:

 points = []
 for doc_id, text, title, dense_vec, sparse_vec in batch:
     points.append(models.PointStruct(
         id=doc_id,
         vector={
             "dense": dense_vec,
             "sparse": models.SparseVector(indices=sparse_vec.indices.tolist(),
                                           values=sparse_vec.values.tolist()),
         },
         payload={"text": text, "title": title, "doc_id": doc_id},
     ))
 client.upsert(collection_name=settings.collection, points=points, wait=False)

 Notes:
 - FiQA passages are already short — do not chunk further.
 - Use the corpus's _id field as the point ID. Hash to UUID5 if non-integer.
 - tqdm progress bar. Expect 15–25 min ingest (mostly OpenAI latency).
 - Make ingest idempotent: if collection exists with expected count, skip.

 ---
 golden_set.py — labeled eval set from FiQA qrels (one-shot batch)

 Use FiQA's native qrels (query → relevant doc IDs) instead of synthesizing.

 Steps:
 1. Load BeIR/fiqa-qrels and BeIR/fiqa queries from HuggingFace.
 2. Filter to the test split.
 3. Sample 300 queries deterministically using RANDOM_SEED=42.
 4. Build entries:
 {
     "query_id": str,
     "query_text": str,
     "labels": {doc_id: relevance_score, ...},
     "ground_truth": str,
 }
 5. Generate a 1–2 sentence reference answer per query using claude-sonnet-4-6, constrained to the relevant docs from the labels. Required by context_precision (Ragas) and ContextualPrecisionMetric (DeepEval). Skip queries where the
 reference comes back as NO_ANSWER.
 6. Cache to results/golden_set.jsonl. The demo always reads from this file — never regenerates references on the fly.

 Reference-generation prompt:

 You are creating a short reference answer for an evaluation set.
 Use ONLY the source passages below. Answer in 1–2 sentences.
 If the sources don't contain the answer, write "NO_ANSWER".

 Sources:
 {relevant_passages}

 Question:
 {query_text}

 ---
 pipeline/retrieval.py — hybrid retrieve + rerank, with toggles

 Single function retrieve(query, k_retrieve, k_rerank, *, hybrid=True, rerank=True):

 from qdrant_client import models
 from sentence_transformers import CrossEncoder

 _reranker = None
 def _get_reranker():
     global _reranker
     if _reranker is None:
         _reranker = CrossEncoder(settings.reranker_model)
     return _reranker

 def retrieve(query, k_retrieve=50, k_rerank=10, *, hybrid=True, rerank=True):
     dense_vec = embed_dense(query)            # OpenAI embedding call
     prefetch = [models.Prefetch(query=dense_vec, using="dense", limit=k_retrieve)]
     if hybrid:
         sparse_vec = embed_sparse(query)      # fastembed BM25
         prefetch.append(models.Prefetch(
             query=models.SparseVector(
                 indices=sparse_vec.indices.tolist(),
                 values=sparse_vec.values.tolist(),
             ),
             using="sparse",
             limit=k_retrieve,
         ))

     results = client.query_points(
         collection_name=settings.collection,
         prefetch=prefetch,
         query=models.FusionQuery(fusion=models.Fusion.RRF) if hybrid else dense_vec,
         using=None if hybrid else "dense",
         limit=k_retrieve,
         with_payload=True,
     ).points

     if rerank:
         pairs = [(query, p.payload["text"]) for p in results]
         scores = _get_reranker().predict(pairs)
         results = [p for p, _ in sorted(zip(results, scores),
                                         key=lambda x: x[1], reverse=True)]

     top = results[:k_rerank]
     return [{"doc_id": p.id, "text": p.payload["text"], "score": float(p.score or 0)}
             for p in top]

 Lazy-load the reranker — it's ~120MB and the demo shouldn't pay startup cost when reranking is toggled off.

 ---
 pipeline/generation.py — generator with selectable model + custom prompt

 import anthropic
 from openai import OpenAI

 _anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)
 _openai = OpenAI(api_key=settings.openai_api_key)

 def generate(query_text: str, contexts: list[str], *,
              model: str, prompt_template: str) -> str:
     prompt = prompt_template.format(
         retrieved_context="\n\n".join(contexts),
         query_text=query_text,
     )
     if model.startswith("claude"):
         resp = _anthropic.messages.create(
             model=model,
             max_tokens=512,
             messages=[{"role": "user", "content": prompt}],
         )
         return resp.content[0].text
     else:  # gpt-*
         resp = _openai.chat.completions.create(
             model=model,
             max_tokens=512,
             messages=[{"role": "user", "content": prompt}],
         )
         return resp.choices[0].message.content

 Both providers are needed because the demo lets the presenter swap generators.

 ---
 pipeline/pipeline.py — end-to-end one-query function

 def run_one(query_text, *, k_retrieve, k_rerank, hybrid, rerank,
             generator_model, prompt_template):
     hits = retrieve(query_text, k_retrieve, k_rerank,
                     hybrid=hybrid, rerank=rerank)
     contexts = [h["text"] for h in hits]
     answer = generate(query_text, contexts,
                       model=generator_model, prompt_template=prompt_template)
     return {"query_text": query_text, "contexts": contexts,
             "answer": answer, "hits": hits}

 ---
 pipeline/scoring.py — per-sample Ragas + DeepEval helpers (parallel, with reasoning)

 This is the key new file. The demo scores one sample at a time (so the table can stream live), and Ragas + DeepEval run in parallel on each sample (so demo wall-clock stays low). Both helpers also return the judge's reasoning per metric,
  which the drill-down panel surfaces.

 import asyncio
 from concurrent.futures import ThreadPoolExecutor

 from ragas import SingleTurnSample
 from ragas.metrics.collections import (
     faithfulness, answer_relevancy, context_precision, context_recall,
 )
 from ragas.llms import LangchainLLMWrapper
 from ragas.embeddings import LangchainEmbeddingsWrapper
 from langchain_openai import ChatOpenAI, OpenAIEmbeddings
 from langchain_anthropic import ChatAnthropic

 from deepeval.test_case import LLMTestCase
 from deepeval.metrics import (
     FaithfulnessMetric, AnswerRelevancyMetric,
     ContextualPrecisionMetric, ContextualRecallMetric, GEval,
 )
 from deepeval.test_case import LLMTestCaseParams
 from deepeval.models import GPTModel, AnthropicModel

 def _ragas_judge(model_name):
     if model_name.startswith("gpt"):
         return LangchainLLMWrapper(ChatOpenAI(model=model_name, temperature=0))
     return LangchainLLMWrapper(ChatAnthropic(model=model_name, temperature=0))

 def _deepeval_judge(model_name):
     if model_name.startswith("gpt"):
         return GPTModel(model=model_name)
     return AnthropicModel(model=model_name)

 # ---------- Ragas ----------

 async def _score_ragas_async(record, judge_model):
     judge = _ragas_judge(judge_model)
     emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
     sample = SingleTurnSample(
         user_input=record["query_text"],
         retrieved_contexts=record["contexts"],
         response=record["answer"],
         reference=record["ground_truth"],
     )
     out = {"scores": {}, "reasons": {}}
     for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
         metric.llm = judge
         metric.embeddings = emb
         # Ragas exposes per-metric reasoning via metric.score_with_reason where available;
         # fall back to single_turn_ascore for metrics that don't expose it.
         try:
             score, reason = await metric.single_turn_ascore_with_reason(sample)
         except AttributeError:
             score = await metric.single_turn_ascore(sample)
             reason = "(reason not exposed by this Ragas metric version)"
         out["scores"][metric.name] = float(score)
         out["reasons"][metric.name] = reason
     return out

 # ---------- DeepEval ----------

 def _score_deepeval_sync(record, judge_model, custom_geval_criterion=None):
     judge = _deepeval_judge(judge_model)
     case = LLMTestCase(
         input=record["query_text"],
         actual_output=record["answer"],
         retrieval_context=record["contexts"],
         expected_output=record["ground_truth"],
     )
     metrics = [
         ("faithfulness", FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
         ("answer_relevancy", AnswerRelevancyMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
         ("context_precision", ContextualPrecisionMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
         ("context_recall", ContextualRecallMetric(model=judge, threshold=0.5, async_mode=False, include_reason=True)),
     ]
     if custom_geval_criterion:
         metrics.append((
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
         ))
     out = {"scores": {}, "reasons": {}}
     for name, m in metrics:
         m.measure(case)
         out["scores"][name] = float(m.score)
         out["reasons"][name] = getattr(m, "reason", "") or ""
     return out

 # ---------- Combined parallel call ----------

 def score_both_parallel(record, judge_model, custom_geval_criterion=None):
     """Run Ragas (async) and DeepEval (sync, in a thread) in parallel for one record.

     Returns: {"ragas": {scores, reasons}, "deepeval": {scores, reasons}, "wall_seconds": float}
     """
     import time
     start = time.time()

     loop = asyncio.new_event_loop()
     try:
         with ThreadPoolExecutor(max_workers=1) as pool:
             de_future = pool.submit(_score_deepeval_sync, record, judge_model, custom_geval_criterion)
             ragas_result = loop.run_until_complete(_score_ragas_async(record, judge_model))
             de_result = de_future.result()
     finally:
         loop.close()

     return {"ragas": ragas_result, "deepeval": de_result, "wall_seconds": time.time() - start}

 The Streamlit app calls score_both_parallel. The batch eval scripts (eval_ragas.py, eval_deepeval.py) reuse _score_ragas_async / _score_deepeval_sync for parity with the demo. Note: if the Ragas API in the installed version doesn't
 expose single_turn_ascore_with_reason, log this in tutorial_gaps.md (it's a real friction point — Ragas's reasoning surface has churned across versions) and fall back to the regular score-only call.

 ---
 🎯 app.py — Streamlit live demo

 This is the centerpiece. Layout:

 Sidebar (knobs):
 - Sample size slider (10 → 300, default 20)
 - Generator model dropdown (GENERATOR_CHOICES)
 - Judge model dropdown (JUDGE_CHOICES) — show a warning if it matches the generator family
 - top_k_retrieve (10 → 100, default 50)
 - top_k_rerank (1 → 20, default 10)
 - Hybrid retrieval toggle (default on)
 - Reranker toggle (default on)
 - Faithfulness threshold slider (0–1, default 0.7)
 - Answer-relevancy threshold slider (0–1, default 0.7)
 - Prompt template textarea (default = DEFAULT_PROMPT_TEMPLATE)
 - G-Eval custom criterion textarea (DeepEval-only, optional). Default empty. Placeholder: "Every numerical claim in the answer must be supported by a specific quoted span from the retrieved context." When non-empty, score_both_parallel
 adds a g_eval_custom column to the live table, and this becomes the centerpiece of the "DeepEval can do things Ragas can't" pitch.
 - ▶ Run sample button (triggers the streaming run)
 - Reset / clear-cache button

 Main pane:
 - Header + brief description
 - Aggregate scorecards row: 8 standard metrics (4 Ragas + 4 DeepEval) + the G-Eval custom metric if active, updated live as queries finish
 - Live results table — one row per scored query, columns: query, ragas_faith, de_faith, ragas_relev, de_relev, ragas_cp, de_cp, ragas_cr, de_cr, optional g_eval_custom, pass/fail (green/red badge based on threshold sliders)
 - Below the table: a small Plotly scatter of ragas_faithfulness vs deepeval_faithfulness to visualize agreement between libraries
 - Per-query drill-down: clicking a row (or selecting in a st.selectbox of completed queries) expands a panel showing question, retrieved contexts (numbered), generated answer, both libs' per-metric scores side-by-side, and the judge's
 reasoning text under each score

 Streaming run loop:

 import streamlit as st
 import pandas as pd

 if st.sidebar.button("▶ Run sample"):
     golden = load_golden_set()  # cached with st.cache_data
     sampled = sample_queries(golden, n=st.session_state.sample_size,
                              seed=settings.random_seed)

     table_placeholder = st.empty()
     metrics_placeholder = st.empty()
     progress = st.progress(0)
     rows = []

     for i, q in enumerate(sampled):
         record = run_one(
             q["query_text"],
             k_retrieve=st.session_state.k_retrieve,
             k_rerank=st.session_state.k_rerank,
             hybrid=st.session_state.hybrid,
             rerank=st.session_state.rerank,
             generator_model=st.session_state.generator_model,
             prompt_template=st.session_state.prompt_template,
         )
         record["ground_truth"] = q["ground_truth"]
         record["query_id"] = q["query_id"]

         # Run Ragas and DeepEval in parallel for this record.
         scored = score_both_parallel(
             record,
             judge_model=st.session_state.judge_model,
             custom_geval_criterion=st.session_state.geval_criterion or None,
         )

         rows.append({
             "query_id": q["query_id"],
             "query": q["query_text"][:80],
             **{f"ragas_{k}": v for k, v in scored["ragas"]["scores"].items()},
             **{f"deepeval_{k}": v for k, v in scored["deepeval"]["scores"].items()},
             "wall_seconds": scored["wall_seconds"],
         })
         # Stash reasons separately for the drill-down panel.
         st.session_state.setdefault("reasons", {})[q["query_id"]] = {
             "ragas": scored["ragas"]["reasons"],
             "deepeval": scored["deepeval"]["reasons"],
         }
         st.session_state.setdefault("records", {})[q["query_id"]] = record

         df = pd.DataFrame(rows)
         table_placeholder.dataframe(df, use_container_width=True)
         metrics_placeholder.write(_format_aggregates(df,
                                   faith_thresh=st.session_state.faith_thresh,
                                   relev_thresh=st.session_state.relev_thresh))
         progress.progress((i + 1) / len(sampled))

     st.session_state.last_run = df

 Threshold sliders are special: they don't trigger re-scoring. They re-bin the existing per-query scores into pass/fail. Implement by recomputing the pass/fail badge whenever sliders move, using st.session_state.last_run. This is what
 sells the "CI gate state flips live" story.

 Drill-down panel (with judge reasoning):

 if "last_run" in st.session_state:
     selected = st.selectbox("Inspect query", st.session_state.last_run["query_id"])
     row = st.session_state.last_run.set_index("query_id").loc[selected]
     full_record = st.session_state.records[selected]
     reasons = st.session_state.reasons[selected]

     col1, col2 = st.columns([1, 1])
     with col1:
         st.subheader("Question")
         st.write(full_record["query_text"])
         st.subheader("Retrieved contexts")
         for i, c in enumerate(full_record["contexts"]):
             with st.expander(f"Chunk {i+1}"):
                 st.write(c)
         st.subheader("Generated answer")
         st.write(full_record["answer"])
         st.subheader("Reference answer")
         st.write(full_record["ground_truth"])

     with col2:
         st.subheader("Side-by-side scores + judge reasoning")
         for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
             st.markdown(f"**{metric}**")
             mc1, mc2 = st.columns(2)
             with mc1:
                 st.metric("Ragas", f"{row[f'ragas_{metric}']:.2f}")
                 with st.expander("Ragas judge reasoning"):
                     st.write(reasons["ragas"].get(metric, "(none)"))
             with mc2:
                 st.metric("DeepEval", f"{row[f'deepeval_{metric}']:.2f}")
                 with st.expander("DeepEval judge reasoning"):
                     st.write(reasons["deepeval"].get(metric, "(none)"))

         if "g_eval_custom" in row.index:
             st.markdown("**G-Eval custom criterion (DeepEval only)**")
             st.metric("g_eval_custom", f"{row['g_eval_custom']:.2f}")
             with st.expander("G-Eval reasoning"):
                 st.write(reasons["deepeval"].get("g_eval_custom", "(none)"))

 The reasoning expanders are the demo's most powerful feature: they turn each score from a number into a debuggable signal, and they let the audience see why one library scored a sample differently than the other.

 Caching to keep the demo snappy:
 - @st.cache_data on load_golden_set() (file read)
 - @st.cache_resource on the reranker and embedding clients
 - Cache (query_id, settings_hash) → (record, ragas_scores, de_scores) in st.session_state so re-running the same config doesn't re-pay
 - The settings hash should include: prompt template, k values, hybrid/rerank toggles, generator model, judge model

 Family-collision warning:
 If sidebar selects e.g. claude-sonnet-4-6 for both generator and judge, render st.warning("Same model family — self-judging contamination risk"). Don't block, just warn.

 Cost meter:
 Track approximate cost as queries complete by counting completion + judge tokens. Display a running total in the sidebar so the presenter knows what each tweak costs.

 ---
 manual_judge.py — capture human ratings on a 10-query slice

 Without human-anchored ratings, the comparison can only say "the libraries disagree by X" — not "library Y was right." This script captures a small human ground truth so the report can answer the lead-library question with evidence.

 CLI flow:
 1. Reads results/eval_records.jsonl, picks the same 10 deterministic queries every time.
 2. For each query, prints: question, retrieved contexts (numbered), generated answer, reference answer.
 3. Prompts the user for faithfulness and answer_relevancy ratings on a 0–1 scale (any number, e.g. 0.8). s to skip.
 4. Writes results/manual_ratings.jsonl with {query_id, faithfulness_human, answer_relevancy_human}.

 # pseudocode
 for r in pick_ten():
     print_record(r)
     f = float(input("faithfulness 0-1 (s to skip): "))
     a = float(input("answer_relevancy 0-1 (s to skip): "))
     append_jsonl(...)

 The user runs this once after batch generation completes. Total time ~10 minutes. The ratings feed into the recommendation rubric in compare.py.

 ---
 eval_ragas.py — batch run (used for the static report)

 Run the full 300-query golden set through the pipeline (with a fixed config — env defaults), score all samples with Ragas, write results/ragas_per_query.csv and results/ragas_scores.json. Reuse pipeline/scoring.py::score_with_ragas per
 sample for parity with the demo, OR call Ragas's batch evaluate() once for speed — pick whichever is faster, but ensure the same samples are scored as in eval_deepeval.py.

 Critical: cache results/eval_records.jsonl so Ragas and DeepEval score the same generated answers. Without this, score differences could be noise from re-generation.

 ---
 eval_deepeval.py — batch run

 Mirror of eval_ragas.py, scores the same eval_records.jsonl with DeepEval. Writes results/deepeval_per_query.csv and results/deepeval_scores.json.

 ---
 tests/test_pipeline_deepeval.py — pytest-native CI gate demo

 import json, pytest
 from deepeval import assert_test
 from deepeval.test_case import LLMTestCase
 from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
 from deepeval.models import GPTModel
 from config import settings

 with open("results/eval_records.jsonl") as f:
     EVAL_RECORDS = [json.loads(line) for line in f]

 judge = GPTModel(model=settings.judge_model)

 @pytest.mark.parametrize("record", EVAL_RECORDS, ids=[r["query_text"][:40] for r in EVAL_RECORDS])
 def test_rag_pipeline(record):
     case = LLMTestCase(
         input=record["query_text"],
         actual_output=record["answer"],
         retrieval_context=record["contexts"],
         expected_output=record["ground_truth"],
     )
     assert_test(case, [
         FaithfulnessMetric(model=judge, threshold=0.7),
         AnswerRelevancyMetric(model=judge, threshold=0.7),
     ])

 Document in the README that pytest tests/ is the CI gate path and shows where DeepEval's pytest-native API earns its keep.

 ---
 compare.py — static comparison report + lead-library recommendation

 Reads ragas_per_query.csv, deepeval_per_query.csv, and manual_ratings.jsonl, joins on query_id, and writes results/comparison_report.md with:

 1. Aggregate scores table for the four overlapping metrics (faithfulness, answer_relevancy, context_precision, context_recall):

 | Metric | Ragas | DeepEval | Δ   |
 |--------|-------|----------|-----|

 2. Per-query score correlation (Pearson + Spearman) for each overlapping metric. If both libraries are measuring the same thing, faithfulness and answer_relevancy should correlate > 0.6.
 3. Manual-judgment correlation (the validation step). For the 10 queries with human ratings, compute Pearson + Spearman between each library's faithfulness and answer_relevancy and the human ratings. The library with higher correlation
 is the more trustworthy judge on this corpus.
 4. Disagreement audit — top 10 queries where Ragas and DeepEval disagree most on faithfulness. Eyeball 3 of them and write a 1-line note on which (if either) was right.
 5. Runtime + cost table — wall-clock seconds and judge-token usage per library. Track via OpenAI/Anthropic API responses.
 6. Library feature matrix:

 | Capability                          | Ragas                        | DeepEval                 |
 |-------------------------------------|------------------------------|--------------------------|
 | Pytest-native API                   | No                           | Yes                      |
 | Custom metrics via natural language | Limited                      | Yes (G-Eval)             |
 | Reference-free metrics              | Yes                          | Yes                      |
 | Built-in dataset/UI                 | No                           | Yes (Confident AI cloud) |
 | Agent-trajectory metrics            | Yes (newer)                  | Yes                      |
 | Setup LOC for this pipeline         | (count)                      | (count)                  |
 | Judge reasoning exposed per metric  | Inconsistent across versions | Yes (metric.reason)      |

 7. Recommendation rubric — explicit, machine-applied criteria so the report ends with a recommendation rather than vibes. Score each criterion +1 to the winning library:

 | Criterion                                                  | Winner                                                                                                                         |
 |------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
 | Higher correlation with manual ratings on faithfulness     | (computed)                                                                                                                     |
 | Higher correlation with manual ratings on answer_relevancy | (computed)                                                                                                                     |
 | Lower setup LOC for this pipeline                          | (computed)                                                                                                                     |
 | Lower wall-clock per query                                 | (computed)                                                                                                                     |
 | Lower judge-token cost per query                           | (computed)                                                                                                                     |
 | Catches a failure mode the other misses (G-Eval test)      | DeepEval if the custom criterion surfaced ≥1 sample where it scored <0.5 while standard metrics scored ≥0.7; otherwise neither |

 7. Output: "Lead with {winner} in the tutorial — won {n}/6 criteria." Include the table so the reader can audit the call.
 8. Reference-answer quality caveat. The golden-set references were LLM-generated (constrained to the relevant passages, but still LLM output). This is the weakest link in the methodology — context_precision and context_recall are scored
 against text that itself wasn't human-validated. The report should call this out explicitly: scores can shift by a few points depending on the reference-generation prompt. Treat the recommendation as directional, not definitive, and
 re-run with human-written references if the call is close.
 9. Tutorial-update suggestions. Reference results/tutorial_gaps.md. Top 5 most impactful gaps + a one-line patch each. This is what the user actually pastes back into the tutorial.

 ---
 README.md

 Walk through:
 1. cp .env.example .env and fill in keys
 2. pip install -r requirements.txt
 3. python ingest.py (15–25 min, ~$1 OpenAI embedding cost)
 4. python golden_set.py (~5 min)
 5. Live demo: streamlit run app.py — main path for showing this off
 6. Batch path (for the static report):
   - python eval_ragas.py (~10–20 min)
   - python eval_deepeval.py (~10–20 min)
   - python manual_judge.py (~10 min, interactive)
   - pytest tests/ (CI demo)
   - python compare.py → results/comparison_report.md

 Document expected total cost: ~$15–25 for the full batch run. Live demo runs (20 queries) are ~$1–2 each.

 ---
 Git workflow

 The user runs this for personal use, so commits don't need to be polished, but split the work into a few logical commits as you go (rather than one giant blob). Suggested milestones:

 1. init: scaffolding — .env.example, requirements.txt, config.py, project layout, empty pipeline/ package, README.md skeleton.
 2. ingest + golden set — ingest.py, golden_set.py working end-to-end against Qdrant.
 3. pipeline modules — pipeline/retrieval.py, pipeline/generation.py, pipeline/pipeline.py, pipeline/scoring.py with score_both_parallel and judge-reasoning extraction.
 4. streamlit demo — app.py with all sidebar knobs, streaming table, drill-down with reasoning, G-Eval criterion textarea.
 5. batch eval + manual judge — eval_ragas.py, eval_deepeval.py, manual_judge.py, tests/test_pipeline_deepeval.py.
 6. comparison report — compare.py with the recommendation rubric, plus first pass of results/tutorial_gaps.md.

 git init at the start, .gitignore for results/ (large CSVs and JSONL), .env, __pycache__/, and the Hugging Face datasets cache. Commit messages can be terse — single-line is fine.

 ---
 Verification

 After implementation:

 1. Smoke test ingest:
 python -c "from qdrant_client import QdrantClient; from config import settings; c = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key); print(c.count(settings.collection))" should report ~57k points.
 2. Smoke test retrieval: from pipeline.retrieval import retrieve; retrieve("how do dividends work?") — 10 hits with non-zero rerank scores.
 3. Smoke test generation: run pipeline.run_one(...) end-to-end on one query — answer is non-empty and grounded.
 4. Smoke test scoring: score_both_parallel(record, "gpt-5.4") returns {"ragas": {scores, reasons}, "deepeval": {scores, reasons}, "wall_seconds": ...} with non-empty reason strings.
 5. G-Eval smoke test: call score_both_parallel(record, "gpt-5.4", custom_geval_criterion="The answer must cite a source for every numerical claim") — DeepEval result includes a g_eval_custom score and reason.
 6. Demo smoke test: streamlit run app.py, set sample size to 5, click ▶ Run sample. All 5 rows appear in the table, threshold sliders re-color pass/fail badges instantly without re-scoring, the drill-down panel shows contexts + scores +
 judge reasoning side by side, and entering a G-Eval criterion adds a g_eval_custom column on the next run.
 7. Batch smoke test: with EVAL_SAMPLE_SIZE=10 in .env, run eval_ragas.py, eval_deepeval.py, manual_judge.py (rate 3 of the 10), then compare.py — comparison_report.md should have all nine sections populated, including a recommendation
 line of the form "Lead with {winner} in the tutorial — won {n}/6 criteria."
 8. Tutorial-gap log smoke test: results/tutorial_gaps.md should have at least 2–3 entries by the time the project runs end-to-end. If it's empty, the build deviated silently — go back and fill it in retroactively.

 ---
 Important details to preserve

 - Tutorial fidelity comes first. Reuse the tutorial's exact code shapes (PROMPT_TEMPLATE, SingleTurnSample, build_eval_set loop, evaluate() call). When you can't, log the deviation in results/tutorial_gaps.md.
 - Ragas + DeepEval run in parallel per record via score_both_parallel. Don't collapse to sequential — the demo's UX depends on it.
 - Surface judge reasoning in the drill-down. The demo's most defensible value is "you can see why the score is what it is."
 - G-Eval lives in DeepEval only. When the criterion textarea is non-empty, the live table grows a g_eval_custom column. This is the single most demo-able DeepEval differentiator — don't bury it.
 - Ingest and golden-set are one-shot scripts. The demo loads cached artifacts; never re-ingests on tweak.
 - Cache eval_records.jsonl so the batch Ragas and DeepEval runs score the same answers.
 - Threshold sliders re-bin existing scores — they must not trigger re-scoring (that's what makes the CI gate flip feel instant).
 - Different model family for judge vs generator — pass model= / llm= explicitly to both libraries; never let them default. Warn in the UI when the family collides.
 - Sample deterministically with RANDOM_SEED=42 — re-running with the same sample size + seed should yield the same query subset.
 - Don't chunk FiQA — passages are already short.
 - BM25 sparse vectors require Modifier.IDF in the collection config.
 - Drop golden-set entries where the reference is NO_ANSWER — context_precision and ContextualPrecisionMetric need a non-empty reference.
 - Track judge token usage during eval runs; the live cost meter and the batch report's cost section both need it.
 - Lazy-load the cross-encoder reranker — the demo shouldn't pay 120 MB of startup if the user toggles rerank off.
 - Reference answers are LLM-generated and so is the weakest link. The recommendation rubric explicitly notes this — don't oversell the result.

 ---
 Out of scope (do not build)

 - A UI for editing the golden-set labels themselves (separate problem).
 - Online metrics or A/B harness — this is offline eval against a frozen golden set.
 - Agentic-RAG metrics (tool-call accuracy, trajectory eval). Mention in the feature matrix but don't exercise — pipeline is single-turn RAG.
 - Continuous tuning loops or hyperparameter sweeps. The pipeline is fixed; the comparison is between eval libraries, not pipeline variants.