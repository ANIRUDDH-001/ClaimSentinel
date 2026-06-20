import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import patch

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from config import CLAIMS_CSV, HISTORY_CSV, EVIDENCE_CSV
from utils.csv_handler import read_claims, read_user_history, read_evidence_requirements
from pipeline.orchestrator import process_claim

def run_tests():
    claims = read_claims(CLAIMS_CSV)
    user_history_df = read_user_history(HISTORY_CSV)
    evidence_req_df = read_evidence_requirements(EVIDENCE_CSV)

    # test cases: 001 is row 0 usually? Let's find it by case_id
    case_001_row = next((r for r in claims if "case_001" in r["image_paths"]), None)
    case_008_row = next((r for r in claims if "case_008" in r["image_paths"]), None)
    
    assert case_001_row is not None, "Could not find case_001"
    assert case_008_row is not None, "Could not find case_008"

    print("--- Test 1: case_001 ---")
    res1 = process_claim(case_001_row, user_history_df, evidence_req_df)
    print("case_001 output:")
    print(json.dumps(res1, indent=2))
    assert res1["claim_status"] == "supported" or res1["claim_status"] == "contradicted", "Test 1 might be different due to Judge, but it should be valid"
    assert "issue_type" in res1
    print("Test 1 Passed")

    print("\n--- Test 2: case_008 (injection) ---")
    res2 = process_claim(case_008_row, user_history_df, evidence_req_df)
    print("case_008 output:")
    print(json.dumps(res2, indent=2))
    assert "text_instruction_present" in res2["risk_flags"], "Test 2 Failed: missing injection flag"
    print("Test 2 Passed")

    print("\n--- Test 3: Crash resilience ---")
    with patch("pipeline.orchestrator.parse_claim", side_effect=ValueError("Boom")):
        res3 = process_claim(case_001_row, user_history_df, evidence_req_df)
        print("Crash output:")
        print(json.dumps(res3, indent=2))
        assert res3["evidence_standard_met"] == "false"
        assert res3["risk_flags"] == "manual_review_required"
    print("Test 3 Passed")

if __name__ == "__main__":
    run_tests()
