"""
Retrieval evaluation for the nutrition RAG engine.

    python -m evals.retrieval_eval                    # held-out test split, markdown report
    python -m evals.retrieval_eval --split dev        # the split used for tuning
    python -m evals.retrieval_eval --split all --json evals/results/retrieval.json
    python -m evals.retrieval_eval --gate             # CI: fail if quality regresses

Methodology
-----------
* 221 labelled queries across 10 categories (canonical names, regional synonyms,
  unseen transliterations, typos, morphology, descriptive paraphrases, compound
  dishes, profile-conditioned queries and out-of-KB queries that must abstain).
* Deterministic 50/50 dev/test split by query hash. Reranker weights and decision
  thresholds were tuned on *dev* only; the README reports *test*.
* Rank metrics (Hit@1, Hit@3, MRR@10) are computed on in-KB queries for every
  ablation. Decision metrics are computed for the full system:
    - grounding accuracy: in-KB query -> correct food AND not abstained
    - correct abstention: out-of-KB query -> ``no_match``
    - false-grounding rate: out-of-KB query confidently grounded to some food
* ``unseen`` marks queries whose spelling is NOT a surface form in the KB, i.e.
  the retriever had to generalise rather than look up an alias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from backend.rag.corpus import default_corpus
from backend.rag.profile import DietProfile
from backend.rag.retriever import HybridRetriever
from backend.rag.text import canonical_key

DATASET = Path(__file__).parent / "datasets" / "retrieval.jsonl"

ABLATIONS: list[tuple[str, dict]] = [
    ("BM25 only", {"channels": ("bm25",), "rerank": False}),
    ("Phonetic fuzzy only", {"channels": ("fuzzy",), "rerank": False}),
    ("BM25 + fuzzy (RRF)", {"channels": ("bm25", "fuzzy"), "rerank": False}),
    ("Hybrid + reranker (full, lexical)", {"channels": ("bm25", "fuzzy"), "rerank": True}),
]
DENSE_ABLATION = ("Hybrid + reranker + dense", {"channels": ("bm25", "fuzzy", "dense"), "rerank": True})

# CI quality gate on the held-out split (set a few points under current results).
GATE = {"hit1": 0.85, "grounding_accuracy": 0.80, "correct_abstention": 0.80}


@dataclass
class Case:
    query: str
    expected: str | None
    tag: str
    profile: DietProfile | None
    split: str
    unseen: bool


def load_cases(path: Path = DATASET) -> list[Case]:
    corpus = default_corpus()
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prof = row.get("profile")
        profile = DietProfile(diet=prof.get("diet"), allergens=frozenset(prof.get("allergens", []))) if prof else None
        split = "dev" if int(hashlib.md5(row["query"].encode()).hexdigest(), 16) % 2 == 0 else "test"
        exp = corpus.get(row["expected"])
        unseen = bool(exp) and canonical_key(row["query"]) not in {canonical_key(s) for s in exp.surface_forms}
        cases.append(Case(row["query"], row["expected"], row["tag"], profile, split, unseen))
    return cases


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def evaluate(retriever: HybridRetriever, cases: list[Case], channels, rerank) -> dict:
    in_kb = [c for c in cases if c.expected]
    ood = [c for c in cases if not c.expected]
    hit1 = hit3 = 0
    rr = 0.0
    grounded_ok = 0
    false_abstain = 0
    correct_abstain = 0
    false_ground = 0
    latencies: list[float] = []
    per_tag: dict[str, list[int]] = {}
    failures: list[dict] = []
    unseen_total = unseen_hit = 0

    for c in cases:
        res = retriever.search(c.query, k=10, profile=c.profile, channels=channels, rerank=rerank)
        latencies.append(res.latency_ms)
        ids = [cand.doc.id for cand in res.candidates]
        if c.expected:
            rank = ids.index(c.expected) + 1 if c.expected in ids else None
            ok1 = rank == 1
            hit1 += ok1
            hit3 += bool(rank and rank <= 3)
            rr += 1 / rank if rank else 0.0
            correct = ok1 and res.decision != "no_match"
            grounded_ok += correct
            false_abstain += res.decision == "no_match"
            if c.unseen:
                unseen_total += 1
                unseen_hit += ok1
            per_tag.setdefault(c.tag, []).append(int(correct if rerank else ok1))
            if not correct:
                failures.append(
                    {
                        "query": c.query,
                        "expected": c.expected,
                        "got": ids[:3],
                        "decision": res.decision,
                        "confidence": res.confidence,
                        "tag": c.tag,
                    }
                )
        else:
            abstained = res.decision == "no_match"
            correct_abstain += abstained
            false_ground += res.decision == "grounded"
            per_tag.setdefault(c.tag, []).append(int(abstained))
            if not abstained:
                failures.append(
                    {
                        "query": c.query,
                        "expected": None,
                        "got": ids[:1],
                        "decision": res.decision,
                        "confidence": res.confidence,
                        "tag": c.tag,
                    }
                )

    n, m = max(len(in_kb), 1), max(len(ood), 1)
    lat = sorted(latencies)
    return {
        "n_in_kb": len(in_kb),
        "n_out_of_kb": len(ood),
        "hit1": hit1 / n,
        "hit3": hit3 / n,
        "mrr10": rr / n,
        "unseen_hit1": unseen_hit / max(unseen_total, 1),
        "n_unseen": unseen_total,
        "grounding_accuracy": grounded_ok / n,
        "false_abstention": false_abstain / n,
        "correct_abstention": correct_abstain / m,
        "false_grounding": false_ground / m,
        "latency_p50_ms": statistics.median(lat),
        "latency_p95_ms": lat[int(0.95 * (len(lat) - 1))],
        "per_tag": {t: sum(v) / len(v) for t, v in sorted(per_tag.items())},
        "per_tag_n": {t: len(v) for t, v in sorted(per_tag.items())},
        "failures": failures,
    }


def build_retriever(with_dense: bool) -> HybridRetriever:
    corpus = default_corpus()
    dense = None
    if with_dense:
        from backend.core.config import get_settings
        from backend.rag.dense import DenseIndex, GeminiEmbedder

        s = get_settings()
        dense = DenseIndex(
            corpus, GeminiEmbedder(s.google_api_key, s.embedding_model, s.embedding_dims), s.cache_dir / "dense"
        )
        dense.build()
        if not dense.ready:
            print(f"[warn] dense channel unavailable: {dense.error}", file=sys.stderr)
            dense = None
    return HybridRetriever(corpus, dense=dense)


def render_markdown(split: str, rows: list[tuple[str, dict]], full: dict) -> str:
    lines = [
        f"### Retrieval quality — `{split}` split ({full['n_in_kb']} in-KB + {full['n_out_of_kb']} out-of-KB queries)",
        "",
    ]
    lines.append("| Configuration | Hit@1 | Hit@3 | MRR@10 | Hit@1 (unseen spellings) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, r in rows:
        lines.append(
            f"| {name} | {_pct(r['hit1'])} | {_pct(r['hit3'])} | {r['mrr10']:.3f} | {_pct(r['unseen_hit1'])} |"
        )
    lines += [
        "",
        "| Decision metric (full system) | Value |",
        "|---|---:|",
        f"| Grounding accuracy (correct food, not abstained) | {_pct(full['grounding_accuracy'])} |",
        f"| Correct abstention on out-of-KB queries | {_pct(full['correct_abstention'])} |",
        f"| False-grounding rate on out-of-KB queries | {_pct(full['false_grounding'])} |",
        f"| False abstention on in-KB queries | {_pct(full['false_abstention'])} |",
        f"| Latency p50 / p95 (per query, CPU) | {full['latency_p50_ms']:.2f} ms / {full['latency_p95_ms']:.2f} ms |",
        "",
        "| Category | n | Accuracy |",
        "|---|---:|---:|",
    ]
    for tag, acc in full["per_tag"].items():
        lines.append(f"| {tag} | {full['per_tag_n'][tag]} | {_pct(acc)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--dense", action="store_true", help="also evaluate the dense channel (needs GOOGLE_API_KEY)")
    ap.add_argument("--json", type=Path, help="write full results (incl. failures) to this file")
    ap.add_argument("--failures", action="store_true", help="print failing queries")
    ap.add_argument("--gate", action="store_true", help="exit non-zero if the full system is below the quality gate")
    args = ap.parse_args(argv)

    cases = [c for c in load_cases() if args.split == "all" or c.split == args.split]
    retriever = build_retriever(args.dense)
    ablations = list(ABLATIONS) + ([DENSE_ABLATION] if args.dense and retriever.dense_ready else [])
    rows = [(name, evaluate(retriever, cases, **cfg)) for name, cfg in ablations]
    full = rows[-1][1]
    print(render_markdown(args.split, rows, full))

    if args.failures:
        print("\nFailures:")
        for f in full["failures"]:
            got = f"got={f['got']} {f['decision']} ({f['confidence']})"
            print(f"  [{f['tag']}] {f['query']!r}: expected={f['expected']} {got}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"split": args.split, "results": dict(rows)}, indent=2), encoding="utf-8")
    if args.gate:
        failed = {k: full[k] for k, v in GATE.items() if full[k] < v}
        if failed:
            print(f"\nQUALITY GATE FAILED: {failed} (thresholds {GATE})", file=sys.stderr)
            return 1
        print(f"\nQuality gate passed {GATE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
