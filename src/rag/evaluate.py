"""Small, transparent RAG retrieval+abstention regression / challenge checks.

Real metrics require your locally installed sentence embedding model and FAISS.
No LLM API is used. These hand-authored, corpus-derived sets are NOT an
independently annotated generalization benchmark.

Examples:
  python -m src.rag.evaluate --compare
  python -m src.rag.evaluate --suite challenge --compare
  python -m src.rag.evaluate --suite both --compare --output reports/local_retrieval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from .evidence_gate import EvidenceDecision
from .legacy_evidence_gate import validate_legacy
from .pipeline import build_pipeline
from .retriever import RobustRetriever

ROOT = Path(__file__).resolve().parents[2]


def evaluate_retrieval(store, examples: list[dict], k: int = 5) -> dict:
    """Direct FAISS ranking without reranking or evidence gating."""
    if not examples:
        raise ValueError("Evaluation examples cannot be empty")
    hits = 0
    reciprocal_ranks = []
    for item in examples:
        found = store.search(item["question"], top_k=k, min_score=-1.0)
        rank = next((i for i, hit in enumerate(found, start=1)
                     if hit.chunk.document_id == item["expected_document_id"]), None)
        hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        print(f"{'PASS' if rank else 'MISS'} rank={rank} | {item['question']}")
    return {"count": len(examples), f"hit@{k}": hits / len(examples),
            f"mrr@{k}": sum(reciprocal_ranks) / len(examples)}


def evaluate_robust_retrieval(store, examples: list[dict], negatives: list[dict], k: int = 5,
                              evidence_gate: Callable | None = None) -> dict:
    if not examples or not negatives:
        raise ValueError("Positive and negative evaluation examples cannot be empty")
    retriever = RobustRetriever(store, evidence_gate=evidence_gate) if evidence_gate else RobustRetriever(store)
    found_count, accepted_count, reciprocal_rank = 0, 0, 0.0
    for item in examples:
        result = retriever.retrieve(item["question"], top_k=k)
        rank = next((i for i, h in enumerate(result.hits, start=1)
                     if h.chunk.document_id == item["expected_document_id"]), None)
        found_count += int(rank is not None)
        reciprocal_rank += 1.0 / rank if rank else 0.0
        accepted = result.status == "supported_candidates"
        accepted_count += int(accepted)
        print(f"{'PASS' if rank else 'MISS'} rank={rank}, gate={'ACCEPT' if accepted else 'REJECT'}, "
              f"support={result.supporting_document_id} | {item['question']}")
    rejected = 0
    for item in negatives:
        result = retriever.retrieve(item["question"], top_k=k)
        is_rejected = result.status != "supported_candidates"
        rejected += int(is_rejected)
        print(f"{'REJECT' if is_rejected else 'FALSE ACCEPT'} negative | {item['question']}")
    positive_accept = accepted_count / len(examples)
    negative_reject = rejected / len(negatives)
    return {
        "positive_count": len(examples), f"hit@{k}": found_count / len(examples),
        f"mrr@{k}": reciprocal_rank / len(examples),
        "positive_gate_accept_rate": positive_accept,
        "negative_count": len(negatives),
        "negative_gate_rejection_rate": negative_reject,
        "balanced_gate_accuracy": (positive_accept + negative_reject) / 2,
        "false_reject_count": len(examples) - accepted_count,
        "false_accept_count": len(negatives) - rejected,
    }


def load_suite(name: str) -> dict:
    filename = "retrieval_challenge.json" if name == "challenge" else "retrieval_regression.json"
    data = json.loads((ROOT / "data" / "evaluation" / filename).read_text(encoding="utf-8"))
    if not data.get("positive") or not data.get("negative"):
        raise ValueError("Evaluation suite needs positive and negative examples")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Compare current gate with a legacy union-of-passages gate and raw FAISS ranking")
    parser.add_argument("--suite", choices=("regression", "challenge", "both"), default="regression")
    parser.add_argument("--output", type=Path, help="Optionally write the metrics JSON to this local path")
    args = parser.parse_args()
    store = build_pipeline()
    report = {"methodology": "Small hand-authored corpus-derived checks, not independent validation or population-level accuracy"}
    suites = ("regression", "challenge") if args.suite == "both" else (args.suite,)
    for suite in suites:
        data = load_suite(suite)
        print(f"\n=== {suite.upper()} SET ===")
        print("\nCurrent passage-level, intent-aware gate:")
        values = {"current_gate": evaluate_robust_retrieval(store, data["positive"], data["negative"])}
        if args.compare:
            print("\nLegacy union-of-passages gate on current retrieval candidates (ablation only):")
            values["legacy_union_gate"] = evaluate_robust_retrieval(
                store, data["positive"], data["negative"], evidence_gate=validate_legacy)
            print("\nRaw FAISS ranking baseline (no evidence gate):")
            values["raw_faiss"] = evaluate_retrieval(store, data["positive"])
        report[suite] = values
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print("\n" + serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote report to: {args.output}")


if __name__ == "__main__":
    main()
