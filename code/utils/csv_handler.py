"""
All CSV reading and writing goes through here. QUOTE_ALL on output so that any
justification text containing commas, quotes, or semicolons never corrupts a row.
"""
import csv
import pandas as pd
from models.allowed_values import OUTPUT_CSV_COLUMNS


def read_claims(path: str) -> list[dict]:
    """Read claims.csv or sample_claims.csv into a list of plain dicts."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_user_history(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_evidence_requirements(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_output_csv(rows: list[dict], path: str) -> None:
    """
    Write final output rows. Enforces exact column order and QUOTE_ALL so every
    field is quoted regardless of content.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_CSV_COLUMNS})
