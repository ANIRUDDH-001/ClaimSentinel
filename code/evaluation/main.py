"""Evaluation harness: runs the pipeline on sample_claims.csv and scores it against
the labeled ground truth columns already present in that file."""
import logging
import json
from dotenv import load_dotenv

load_dotenv()
from config import SAMPLE_CSV, HISTORY_CSV, EVIDENCE_CSV
from utils.csv_handler import read_claims, read_user_history, read_evidence_requirements
from pipeline.orchestrator import process_claim
from evaluation.metrics import full_report, find_mismatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INPUT_FIELDS = ["user_id", "image_paths", "user_claim", "claim_object"]


def main():
    rows = read_claims(SAMPLE_CSV)
    user_history_df = read_user_history(HISTORY_CSV)
    evidence_req_df = read_evidence_requirements(EVIDENCE_CSV)

    predictions, ground_truth = [], []

    for i, row in enumerate(rows, start=1):
        input_row = {k: row[k] for k in INPUT_FIELDS}
        pred = process_claim(input_row, user_history_df, evidence_req_df)
        predictions.append(pred)
        ground_truth.append(row)

    from utils import api_client
    print("\n--- Usage Summary ---")
    print(json.dumps(api_client.get_usage_summary(), indent=2))

    report = full_report(predictions, ground_truth)
    print(json.dumps(report, indent=2))

    overall_correct = sum(
        1 for p, g in zip(predictions, ground_truth)
        if p.get("claim_status") == g.get("claim_status")
    )
    print(f"\nclaim_status accuracy = {overall_correct}/{len(rows)} "
          f"({overall_correct/len(rows)*100:.1f}%)")

    for field in ["claim_status", "issue_type", "severity"]:
        mismatches = find_mismatches(predictions, ground_truth, field)
        if mismatches:
            print(f"\n--- Mismatches on {field} ({len(mismatches)}) ---")
            for m in mismatches:
                print(f"  {m['user_id']}: predicted={m['predicted']!r} expected={m['expected']!r}")


if __name__ == "__main__":
    main()
