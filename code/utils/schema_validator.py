"""
Three layers of validation:
  Layer 1 — validate_input_row():    before any LLM call
  Layer 2 — validate_stage_output(): after every LLM call, with repair_via_llm() as backstop
  Layer 3 — models.schemas.OutputRow (Pydantic): the final gate before CSV (built in Phase 1.3)
"""
import logging
from pathlib import Path
from config import DATASET_DIR
from models.allowed_values import CLAIM_OBJECTS


def validate_input_row(row: dict) -> tuple[bool, list[str]]:
    """
    Validate a claims.csv row before processing.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    required = ["user_id", "image_paths", "user_claim", "claim_object"]

    for field in required:
        if not row.get(field):
            errors.append(f"Missing required field: {field}")

    if row.get("claim_object") not in CLAIM_OBJECTS:
        errors.append(f"Invalid claim_object: {row.get('claim_object')}")

    if not errors:
        image_paths = [p.strip() for p in row["image_paths"].split(";")]
        for p in image_paths:
            if not (Path(DATASET_DIR) / p).exists():
                errors.append(f"Image not found: {p}")

    return len(errors) == 0, errors


STAGE_SCHEMAS = {
    "claim_parser": {
        "primary_issue_type": str,
        "primary_object_part": str,
        "secondary_claims": list,
        "claim_summary": str,
        "severity_claimed": str,
        "language_detected": str,
    },
    "image_analyzer": {
        "image_id": str,
        "object_detected": str,
        "correct_object_type": bool,
        "claimed_part_visible": bool,
        "damage_visible": bool,
        "visible_damage_type": str,
        "damage_matches_claim": str,
        "image_quality": str,
        "non_original_suspected": bool,
        "text_found_in_image": bool,
        "text_is_injection_attempt": bool,
        "findings_narrative": str,
    },
    "evidence": {
        "evidence_standard_met": bool,
        "evidence_standard_met_reason": str,
    },
    "decision": {
        "issue_type": str,
        "object_part": str,
        "claim_status": str,
        "claim_status_justification": str,
        "supporting_image_ids": str,
        "severity": str,
    },
}


def validate_stage_output(stage: str, output: dict) -> tuple[bool, dict]:
    """
    Validates raw LLM output for a stage.
    Returns (is_valid, repaired_output). Repairs fill missing optional fields with
    safe defaults. is_valid=False signals the caller that repair_via_llm() may be
    worth running for a stronger fix.
    """
    schema = STAGE_SCHEMAS.get(stage, {})
    repaired = dict(output)
    is_valid = True

    for field, expected_type in schema.items():
        if field not in repaired:
            is_valid = False
            if expected_type == bool:
                repaired[field] = False
            elif expected_type == list:
                repaired[field] = []
            elif expected_type == str:
                repaired[field] = "unknown"
        else:
            if expected_type == bool and isinstance(repaired[field], str):
                repaired[field] = repaired[field].strip().lower() == "true"

    return is_valid, repaired


def repair_via_llm(stage: str, broken_output: dict, original_prompt: str) -> dict:
    """
    When stage output is critically broken (missing core fields), run a lightweight
    repair call. Used sparingly — only when validate_stage_output() returns False on
    a field that actually matters for downstream logic.
    """
    from utils.api_client import call_text

    repair_prompt = f"""
The following JSON output is missing required fields or has invalid values.
Fix it to conform to the expected schema. Return ONLY valid JSON.

ORIGINAL PROMPT CONTEXT:
{original_prompt[:500]}...

BROKEN OUTPUT:
{broken_output}

REQUIRED FIELDS: {list(STAGE_SCHEMAS.get(stage, {}).keys())}

Return a complete, valid JSON with all required fields filled in.
If uncertain about a value, use "unknown" for strings or false for booleans.
"""
    logging.warning(f"[{stage}] Running LLM repair on broken output: {list(broken_output.keys())}")
    return call_text("claim_parser", repair_prompt)  # a reliable model, used purely for repair
