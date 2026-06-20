"""Scoring functions for comparing predicted output rows against labeled ground truth."""
from collections import defaultdict

COMPARABLE_FIELDS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]


def field_accuracy(predictions: list[dict], ground_truth: list[dict], field: str) -> float:
    """Exact-match accuracy for a single field across all rows."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, g in zip(predictions, ground_truth)
        if str(p.get(field, "")).strip().lower() == str(g.get(field, "")).strip().lower()
    )
    return correct / len(predictions)


def claim_status_classification_report(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """
    Per-class precision/recall/F1 for claim_status, plus macro-average F1.
    Classes: supported, contradicted, not_enough_information.
    """
    classes = ["supported", "contradicted", "not_enough_information"]
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for p, g in zip(predictions, ground_truth):
        pred_label = p.get("claim_status", "")
        true_label = g.get("claim_status", "")

        if pred_label == true_label and pred_label in classes:
            counts[pred_label]["tp"] += 1
        else:
            if pred_label in classes:
                counts[pred_label]["fp"] += 1
            if true_label in classes:
                counts[true_label]["fn"] += 1

    report = {}
    f1_scores = []
    for c in classes:
        tp, fp, fn = counts[c]["tp"], counts[c]["fp"], counts[c]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        report[c] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
        f1_scores.append(f1)

    report["macro_f1"] = round(sum(f1_scores) / len(f1_scores), 3)
    return report


def full_report(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """One combined report: per-field accuracy + claim_status classification breakdown."""
    accuracies = {f: round(field_accuracy(predictions, ground_truth, f), 3) for f in COMPARABLE_FIELDS}
    return {
        "n_rows": len(predictions),
        "field_accuracy": accuracies,
        "claim_status_report": claim_status_classification_report(predictions, ground_truth),
    }


def find_mismatches(predictions: list[dict], ground_truth: list[dict], field: str) -> list[dict]:
    """Returns rows where a specific field didn't match, for manual review."""
    mismatches = []
    for p, g in zip(predictions, ground_truth):
        pred_val = str(p.get(field, "")).strip().lower()
        true_val = str(g.get(field, "")).strip().lower()
        if pred_val != true_val:
            mismatches.append({
                "user_id": g.get("user_id"),
                "image_paths": g.get("image_paths"),
                "field": field,
                "predicted": p.get(field),
                "expected": g.get(field),
            })
    return mismatches
