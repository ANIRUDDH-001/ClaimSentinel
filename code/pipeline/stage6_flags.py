"""Stage 6: final flag assembly. Pure Python, no API calls."""
from models.allowed_values import ORDERED_FLAGS


def assemble_flags(precheck: dict, coherence: dict, image_findings: list[dict],
                   decision: dict, severity_claimed: str,
                   judge_flags_to_add: list[str] = None,
                   judge_flags_to_remove: list[str] = None) -> str:
    """
    Final flag union. Returns semicolon-separated string of flags, or "none".
    """
    judge_flags_to_add = judge_flags_to_add or []
    judge_flags_to_remove = judge_flags_to_remove or []

    flags = set()

    flags.update(precheck["pre_flags"])
    flags.update(coherence["additional_flags"])

    for f in image_findings:
        flags.update(f.get("image_risk_flags", []))
        if f.get("non_original_suspected"):
            flags.add("non_original_image")
        if f.get("text_is_injection_attempt"):
            flags.add("text_instruction_present")

    # Co-occurrence rule: history risk + any other concrete red flag → manual review
    if "user_history_risk" in flags and any(
        f in flags for f in ["claim_mismatch", "non_original_image",
                             "text_instruction_present", "damage_not_visible"]
    ):
        flags.add("manual_review_required")

    # Derived from decision: was the user's implied severity far above what's visible?
    claim_status = decision.get("claim_status", "")
    if claim_status == "contradicted":
        visible_sev = decision.get("severity", "unknown")
        sev_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}
        if sev_order.get(severity_claimed, -1) > sev_order.get(visible_sev, -1) + 1:
            flags.add("claim_mismatch")

    # Apply judge corrections last, so they have the final say
    flags.update(judge_flags_to_add)
    flags.difference_update(judge_flags_to_remove)

    flags.discard("none")

    if not flags:
        return "none"

    ordered = [f for f in ORDERED_FLAGS if f in flags]
    return ";".join(ordered)
