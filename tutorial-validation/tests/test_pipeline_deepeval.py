"""Pytest-native CI gate using DeepEval's assert_test.

This file is the most concrete demonstration of why the feature matrix in
the comparison report calls out 'Pytest-native API: Yes (assert_test)' for
DeepEval and 'No' for Ragas. With Ragas you'd write a custom test runner;
with DeepEval, parametrising over records and calling `assert_test` is all
the boilerplate you need for a real CI gate.

Run:    pytest tests/

Reads results/eval_records.jsonl. If the file doesn't exist, every
parametrised case is skipped (we hit `eval_ragas.py` to populate it).

Threshold = 0.7 on faithfulness + answer_relevancy. Matches the demo's
default CI-gate slider so the same record passing/failing in the live demo
also passes/fails the test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from config import settings
from pipeline.scoring import _make_deepeval_judge, _resolve_judge_model


_RECORDS_PATH = Path(__file__).resolve().parent.parent / "results" / "eval_records.jsonl"


def _load_records() -> list[dict]:
    if not _RECORDS_PATH.exists():
        return []
    return [json.loads(l) for l in _RECORDS_PATH.read_text().splitlines() if l.strip()]


_RECORDS = _load_records()
_JUDGE_MODEL = _resolve_judge_model(settings.judge_model) if _RECORDS else None
_JUDGE = _make_deepeval_judge(_JUDGE_MODEL) if _JUDGE_MODEL else None


@pytest.mark.skipif(not _RECORDS, reason="no eval_records.jsonl — run eval_ragas.py first")
@pytest.mark.parametrize(
    "record",
    _RECORDS,
    ids=[r["query_text"][:40] for r in _RECORDS] if _RECORDS else [],
)
def test_rag_pipeline(record):
    case = LLMTestCase(
        input=record["query_text"],
        actual_output=record["answer"],
        retrieval_context=record["contexts"],
        expected_output=record["ground_truth"],
    )
    assert_test(
        case,
        [
            FaithfulnessMetric(model=_JUDGE, threshold=0.7),
            AnswerRelevancyMetric(model=_JUDGE, threshold=0.7),
        ],
    )
