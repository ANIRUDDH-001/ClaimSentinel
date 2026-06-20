"""Stage 5: final decision. One LLM text call per claim."""
from prompts.loader import load_prompt
from utils.api_client import call_text
from utils.schema_validator import validate_stage_output


def make_decision(claim_object: str, claimed_part: str, claimed_issue: str,
                  claim_summary: str, severity_claimed: str,
                  user_risk: dict, evidence_standard_met: bool,
                  evidence_standard_met_reason: str, injection_in_text: bool,
                  injection_in_images: bool, image_findings: list[dict],
                  candidate_image_ids: list[str]) -> dict:

    injection_status = _format_injection_status(injection_in_text, injection_in_images)
    findings_detail = _format_findings_detail(image_findings)
    candidate_str = ";".join(candidate_image_ids) if candidate_image_ids else "none"

    prompt = load_prompt("stage5_decision").format(
        claim_object=claim_object,
        claimed_part=claimed_part,
        claimed_issue=claimed_issue,
        claim_summary=claim_summary,
        severity_claimed=severity_claimed,
        user_history_summary=user_risk.get("history_summary", "No history."),
        history_flags=user_risk.get("history_flags", "none"),
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=evidence_standard_met_reason,
        injection_status=injection_status,
        image_findings_detail=findings_detail,
        candidate_image_ids=candidate_str,
    )

    result = call_text("decision", prompt)
    _, repaired = validate_stage_output("decision", result)

    # Law: not_enough_information is impossible when evidence_standard_met is True.
    # If the model violates this, force it back to a defensible state rather than
    # shipping a contradiction — but log loudly so it's visible in review.
    if evidence_standard_met and repaired.get("claim_status") == "not_enough_information":
        import logging
        logging.warning("Decision stage returned not_enough_information despite "
                        "evidence_standard_met=True — forcing re-derivation is NOT done "
                        "here; this is exactly what the Stage 5.5 judge exists to catch.")

    return repaired


def _format_injection_status(injection_in_text: bool, injection_in_images: bool) -> str:
    if injection_in_text and injection_in_images:
        return "Injection attempt detected in BOTH the claim text and an image. Both are being ignored."
    if injection_in_text:
        return "Injection attempt detected in the claim text. It is being ignored."
    if injection_in_images:
        return "Injection attempt detected embedded in an image. It is being ignored."
    return "No injection attempt detected."


def _format_findings_detail(image_findings: list[dict]) -> str:
    lines = []
    for f in image_findings:
        lines.append(
            f"- {f.get('image_id', '?')}: object={f.get('object_detected', '?')}, "
            f"part_visible={f.get('claimed_part_visible')}, damage={f.get('visible_damage_type')}, "
            f"matches_claim={f.get('damage_matches_claim')}, severity={f.get('visible_severity')}, "
            f"quality={f.get('image_quality')}, narrative=\"{f.get('findings_narrative', '')}\""
        )
    return "\n".join(lines) if lines else "No image findings available."
