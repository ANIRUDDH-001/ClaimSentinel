"""Stage 0: pre-check. Pure Python, zero API calls."""
import logging
from pathlib import Path
from config import DATASET_DIR
from utils.injection_scanner import scan_injection_text


def precheck(row: dict, user_history_df, evidence_req_df) -> dict:
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    claim_text = row["user_claim"]
    image_paths = [p.strip() for p in row["image_paths"].split(";")]

    injection_result = scan_injection_text(claim_text)
    user_risk = lookup_user_risk(user_id, user_history_df)
    evidence_reqs = get_evidence_requirements(claim_object, evidence_req_df)

    valid_paths = [p for p in image_paths if (Path(DATASET_DIR) / p).exists()]
    missing_paths = [p for p in image_paths if p not in valid_paths]
    if missing_paths:
        logging.warning(f"[{user_id}] Missing images: {missing_paths}")

    return {
        "user_id": user_id,
        "claim_object": claim_object,
        "claim_text": claim_text,
        "image_paths": valid_paths,
        "missing_image_paths": missing_paths,
        "injection_in_text": injection_result["found"],
        "injection_phrases": injection_result["phrases"],
        "user_risk": user_risk,
        "evidence_requirements": evidence_reqs,
        "pre_flags": injection_result["flags"] + user_risk["recommended_flags"],
    }


def lookup_user_risk(user_id: str, df) -> dict:
    row = df[df["user_id"] == user_id]
    if row.empty:
        return {"found": False, "history_flags": "none",
                "history_summary": "No history found.", "recommended_flags": []}

    r = row.iloc[0]
    history_flags_raw = str(r["history_flags"])
    flags = [] if history_flags_raw == "none" else history_flags_raw.split(";")

    recommended = []
    if "user_history_risk" in flags:
        recommended.append("user_history_risk")
    if "manual_review_required" in flags:
        recommended.append("manual_review_required")

    return {
        "found": True,
        "past_claim_count": int(r["past_claim_count"]),
        "accepted": int(r["accept_claim"]),
        "rejected": int(r["rejected_claim"]),
        "last_90_days": int(r["last_90_days_claim_count"]),
        "history_flags": history_flags_raw,
        "history_summary": str(r["history_summary"]),
        "recommended_flags": recommended,
    }


def get_evidence_requirements(claim_object: str, df) -> list[dict]:
    """Returns rows applicable to this object: object-specific rows plus 'all' rows."""
    matches = df[(df["claim_object"] == claim_object) | (df["claim_object"] == "all")]
    return matches.to_dict(orient="records")
