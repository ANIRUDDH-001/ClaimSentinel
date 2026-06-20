"""Stage 5.5: adversarial LLM-as-a-judge review of Stage 5's decision."""
import logging
import json
from prompts.loader import load_prompt
from utils.api_client import call_text

CORRECTABLE_FIELDS = {"claim_status", "issue_type", "severity"}


def run_judge(claim_object: str, claimed_part: str, claimed_issue: str, claim_summary: str,
             image_findings: list[dict], proposed_decision: dict,
             current_risk_flags: list[str]) -> tuple[dict, list[str], list[str]]:
    """
    Returns (corrected_decision, risk_flags_to_add, risk_flags_to_remove).
    If the judge approves, corrected_decision is proposed_decision unchanged and
    both flag-delta lists are empty.
    """
    findings_detail = _format_findings_detail(image_findings)
    flags_str = ";".join(current_risk_flags) if current_risk_flags else "none"

    prompt = load_prompt("stage55_judge").format(
        claim_object=claim_object,
        claimed_part=claimed_part,
        claimed_issue=claimed_issue,
        claim_summary=claim_summary,
        image_findings_detail=findings_detail,
        proposed_claim_status=proposed_decision.get("claim_status", ""),
        proposed_issue_type=proposed_decision.get("issue_type", ""),
        proposed_severity=proposed_decision.get("severity", ""),
        proposed_risk_flags=flags_str,
        proposed_justification=proposed_decision.get("claim_status_justification", ""),
    )

    judge_result = call_text("judge", prompt)

    if not judge_result or str(judge_result.get("verdict", "")).strip().lower() == "approved":
        return proposed_decision, [], []

    corrected = dict(proposed_decision)

    logging.debug(f"Judge raw response: {judge_result}")

    # --- claim_status ---
    VALID_STATUSES = {"supported", "contradicted", "not_enough_information"}
    corrected["claim_status"] = _extract_correction(
        judge_result, "claim_status", VALID_STATUSES,
        proposed_decision.get("claim_status", "")
    )

    # --- issue_type ---
    it_verdict = str(judge_result.get("issue_type_verdict", "ok")).strip().lower()
    if it_verdict != "ok":
        correction = judge_result.get("issue_type_correction")
        if correction and str(correction).strip():
            corrected["issue_type"] = str(correction).strip()
        elif it_verdict not in {"ok", "corrected", "wrong", "incorrect", "error"}:
            # The verdict field itself might contain the corrected value
            corrected["issue_type"] = it_verdict

    # --- severity ---
    VALID_SEVERITIES = {"none", "low", "medium", "high", "unknown"}
    corrected["severity"] = _extract_correction(
        judge_result, "severity", VALID_SEVERITIES,
        proposed_decision.get("severity", "")
    )

    flags_to_add = judge_result.get("risk_flags_to_add", []) or []
    flags_to_remove = judge_result.get("risk_flags_to_remove", []) or []

    logging.info(f"Judge verdict=corrected: {judge_result.get('judge_reasoning', '')}")

    return corrected, flags_to_add, flags_to_remove


def _extract_correction(judge_result: dict, field: str, valid_values: set,
                        fallback: str) -> str:
    """
    Robustly extract a correction from the judge response.
    Handles three patterns the LLM might use:
      1. verdict="corrected", correction="supported"
      2. verdict="wrong", correction="supported"
      3. verdict="supported", correction=null  (value in verdict field directly)
    """
    verdict_key = f"{field}_verdict"
    correction_key = f"{field}_correction"

    verdict = str(judge_result.get(verdict_key, "ok")).strip().lower()
    correction = judge_result.get(correction_key)

    if verdict == "ok":
        return fallback

    # Try the explicit correction field first
    if correction and str(correction).strip().lower() in valid_values:
        return str(correction).strip().lower()

    # Fall back: maybe the LLM put the corrected value in the verdict field itself
    if verdict in valid_values:
        return verdict

    # Verdict says "not ok" but no valid correction found — keep original
    logging.warning(f"Judge flagged {field} (verdict={verdict!r}) but no valid "
                    f"correction found (correction={correction!r}). Keeping original.")
    return fallback


def _format_findings_detail(image_findings: list[dict]) -> str:
    lines = []
    for f in image_findings:
        lines.append(
            f"- {f.get('image_id', '?')}: damage={f.get('visible_damage_type')}, "
            f"matches_claim={f.get('damage_matches_claim')}, severity={f.get('visible_severity')}, "
            f"narrative=\"{f.get('findings_narrative', '')}\""
        )
    return "\n".join(lines) if lines else "No image findings available."
