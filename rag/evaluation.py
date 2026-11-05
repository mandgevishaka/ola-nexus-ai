"""Document-level precision/recall comparison for both RAG collections."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path

from rag.indexer import RAGCore

# Each query has one expected source document. This is the small labelled set
# used consistently for both collections, as required by the capstone brief.
EVALUATION_QUERIES = [
    ("What is the initial response time for urgent tickets?", "sla_by_severity.md"),
    ("When does executive escalation happen?", "escalation_matrix.md"),
    ("What refund amount can be approved automatically?", "refund_compensation_policy.md"),
    ("Can agents use personal messaging applications?", "customer_communication_channels.md"),
    ("How long are support tickets retained?", "data_retention_policy.md"),
]


def compare_chunking_strategies(rag: RAGCore, k: int = 3) -> dict[str, Any]:
    """Score retrieved unique parent documents and preserve per-query arithmetic."""
    report: dict[str, Any] = {}
    for strategy in ("fixed", "sentence"):
        rows = []
        total_tp = total_fp = total_fn = 0
        for query, expected_source in EVALUATION_QUERIES:
            result = rag.retrieve(query, strategy=strategy, k=k)
            retrieved_sources = sorted({item["source"] for item in result["metadatas"]})
            relevant = {expected_source}
            retrieved = set(retrieved_sources)
            true_positive = len(relevant & retrieved)
            false_positive = len(retrieved - relevant)
            false_negative = len(relevant - retrieved)
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
            total_tp += true_positive
            total_fp += false_positive
            total_fn += false_negative
            rows.append(
                {
                    "query": query,
                    "expected_document": expected_source,
                    "retrieved_documents": retrieved_sources,
                    "arithmetic": f"TP={true_positive}, FP={false_positive}, FN={false_negative}",
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                }
            )
        report[strategy] = {
            "per_query": rows,
            "micro_precision": round(total_tp / (total_tp + total_fp), 4) if total_tp + total_fp else 0.0,
            "micro_recall": round(total_tp / (total_tp + total_fn), 4) if total_tp + total_fn else 0.0,
            "totals": {"TP": total_tp, "FP": total_fp, "FN": total_fn},
        }

    # Recommend based on measured recall first, then precision; this avoids a
    # recommendation based on a hard-coded preference for one chunking method.
    fixed = report["fixed"]
    sentence = report["sentence"]
    if (sentence["micro_recall"], sentence["micro_precision"]) >= (
        fixed["micro_recall"], fixed["micro_precision"]
    ):
        report["recommendation"] = "sentence"
    else:
        report["recommendation"] = "fixed"
    rag.recommended_strategy = report["recommendation"]
    chosen = report[report["recommendation"]]
    other_name = "fixed" if report["recommendation"] == "sentence" else "sentence"
    other = report[other_name]
    report["recommendation_explanation"] = (
        f"Deploy {report['recommendation']} chunking: it achieved micro precision "
        f"{chosen['micro_precision']:.3f} and micro recall {chosen['micro_recall']:.3f}. "
        f"{other_name.capitalize()} chunking achieved micro precision "
        f"{other['micro_precision']:.3f} and micro recall {other['micro_recall']:.3f}; "
        "the selected strategy ranks higher by recall, with precision used as the tie-breaker. "
        if (chosen["micro_recall"], chosen["micro_precision"]) != (other["micro_recall"], other["micro_precision"])
        else f"Both strategies tie at micro precision {chosen['micro_precision']:.3f} and micro recall {chosen['micro_recall']:.3f}; sentence chunking is selected as the simpler policy-sentence retrieval unit."
    )
    report_path = Path(__file__).resolve().parents[1] / "reports" / "chunking_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
