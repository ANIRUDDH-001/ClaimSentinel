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

    if not judge_result or judge_result.get("verdict") == "approved":
        return proposed_decision, [], []

    corrected = dict(proposed_decision)

    if judge_result.get("claim_status_verdict") == "corrected":
        correction = judge_result.get("claim_status_correction")
        if correction in {"supported", "contradicted", "not_enough_information"}:
            corrected["claim_status"] = correction

    if judge_result.get("issue_type_verdict") == "corrected":
        correction = judge_result.get("issue_type_correction")
        if correction:
            corrected["issue_type"] = correction

    if judge_result.get("severity_verdict") == "corrected":
        correction = judge_result.get("severity_correction")
        if correction in {"none", "low", "medium", "high", "unknown"}:
            corrected["severity"] = correction

    flags_to_add = judge_result.get("risk_flags_to_add", []) or []
    flags_to_remove = judge_result.get("risk_flags_to_remove", []) or []

    logging.info(f"Judge verdict=corrected: {judge_result.get('judge_reasoning', '')}")

    return corrected, flags_to_add, flags_to_remove


def _format_findings_detail(image_findings: list[dict]) -> str:
    lines = []
    for f in image_findings:
        lines.append(
            f"- {f.get('image_id', '?')}: damage={f.get('visible_damage_type')}, "
            f"matches_claim={f.get('damage_matches_claim')}, severity={f.get('visible_severity')}, "
            f"narrative=\"{f.get('findings_narrative', '')}\""
        )
    return "\n".join(lines) if lines else "No image findings available."
