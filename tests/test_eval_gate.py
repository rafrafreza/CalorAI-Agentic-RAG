"""Retrieval quality is a tested property: CI fails if the held-out metrics regress."""

from evals.retrieval_eval import GATE, build_retriever, evaluate, load_cases


def test_held_out_retrieval_quality_gate():
    cases = [c for c in load_cases() if c.split == "test"]
    result = evaluate(build_retriever(with_dense=False), cases, channels=("bm25", "fuzzy"), rerank=True)
    for metric, threshold in GATE.items():
        assert result[metric] >= threshold, (metric, result[metric], result["failures"])
    assert result["false_grounding"] <= 0.15


def test_dataset_is_split_and_labelled():
    cases = load_cases()
    assert len(cases) >= 200
    assert {c.split for c in cases} == {"dev", "test"}
    assert sum(1 for c in cases if c.expected is None) >= 25  # out-of-KB queries that must abstain
