"""Entry point: reads claims.csv, runs the pipeline on every row, writes output.csv."""
import logging
import time
from dotenv import load_dotenv
load_dotenv()

from config import CLAIMS_CSV, HISTORY_CSV, EVIDENCE_CSV, OUTPUT_CSV
from utils.csv_handler import read_claims, read_user_history, read_evidence_requirements, write_output_csv
from pipeline.orchestrator import process_claim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    claims = read_claims(CLAIMS_CSV)
    user_history_df = read_user_history(HISTORY_CSV)
    evidence_req_df = read_evidence_requirements(EVIDENCE_CSV)

    logging.info(f"Processing {len(claims)} claims...")
    start = time.time()

    output_rows = []
    for i, row in enumerate(claims, start=1):
        logging.info(f"[{i}/{len(claims)}] {row['user_id']} — {row['claim_object']}")
        output_rows.append(process_claim(row, user_history_df, evidence_req_df))

    write_output_csv(output_rows, OUTPUT_CSV)

    elapsed = time.time() - start
    logging.info(f"Done. {len(output_rows)} rows written to {OUTPUT_CSV} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
