"""Stage 7: final assembly, Pydantic validation, and safe fallback."""
import logging
from pydantic import ValidationError
from models.schemas import OutputRow


def finalize_row(input_row: dict, decision: dict, evidence_standard_met: bool,
                 evidence_standard_met_reason: str, risk_flags: str,
                 valid_image: bool) -> dict:
    """
    Assembles and validates the final output row. Never raises — falls back to
    emergency_fallback_row() on any validation error.
    """
    try:
        output = OutputRow(
            user_id=input_row["user_id"],
            image_paths=input_row["image_paths"],
            user_claim=input_row["user_claim"],
            claim_object=input_row["claim_object"],
            evidence_standard_met=evidence_standard_met,
            evidence_standard_met_reason=evidence_standard_met_reason,
            risk_flags=risk_flags,
            issue_type=decision.get("issue_type", "unknown"),
            object_part=decision.get("object_part", "unknown"),
            claim_status=decision.get("claim_status", "not_enough_information"),
            claim_status_justification=decision.get("claim_status_justification", ""),
            supporting_image_ids=decision.get("supporting_image_ids", "none"),
            valid_image=valid_image,
            severity=decision.get("severity", "unknown"),
        )
        return output.to_csv_row()

    except ValidationError as e:
        return emergency_fallback_row(input_row, str(e))

    except Exception as e:
        # Catch-all: anything unexpected still produces a valid, safe row.
        return emergency_fallback_row(input_row, f"Unexpected error: {e}")


def emergency_fallback_row(input_row: dict, error: str) -> dict:
    """
    Returns a safe, fully-valid output row when processing or validation fails
    completely. Logs the error. Never raises. Never leaves a field empty.
    """
    logging.error(f"Emergency fallback for {input_row.get('user_id', '?')}: {error}")
    return {
        "user_id": input_row.get("user_id", "unknown"),
        "image_paths": input_row.get("image_paths", ""),
        "user_claim": input_row.get("user_claim", ""),
        "claim_object": input_row.get("claim_object", "unknown"),
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Processing error; manual review required.",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Automated processing failed. Manual review required.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }
