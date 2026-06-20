"""Orchestrator: runs all 9 stages, in order, for one claim row."""
import logging
from config import JUDGE_ENABLED
from pipeline.stage0_precheck import precheck
from pipeline.stage1_claim_parser import parse_claim
from pipeline.stage2_image_analyzer import analyze_image
from pipeline.stage3_coherence import cross_image_coherence
from pipeline.stage4_evidence import evaluate_evidence
from pipeline.stage5_decision import make_decision
from pipeline.stage55_judge import run_judge
from pipeline.stage6_flags import assemble_flags
from pipeline.stage7_validator import finalize_row, emergency_fallback_row


def process_claim(row: dict, user_history_df, evidence_req_df) -> dict:
    """
    Runs the full pipeline for one claims.csv row. Always returns a valid output
    dict — never raises out to the caller.
    """
    try:
        # Stage 0
        ctx = precheck(row, user_history_df, evidence_req_df)

        # Stage 1
        claim = parse_claim(ctx["claim_object"], ctx["claim_text"])
        claimed_part = claim.get("primary_object_part", "unknown")
        claimed_issue = claim.get("primary_issue_type", "unknown")
        claim_summary = claim.get("claim_summary", "")
        severity_claimed = claim.get("severity_claimed", "unknown")

        # Stage 2 — one call per valid image
        image_findings = [
            analyze_image(p, ctx["claim_object"], claimed_part, claimed_issue, claim_summary)
            for p in ctx["image_paths"]
        ]

        injection_in_images = any(f.get("text_is_injection_attempt") for f in image_findings)

        # Stage 3
        coherence = cross_image_coherence(image_findings, claimed_part)

        # Stage 4
        evidence = evaluate_evidence(
            ctx["claim_object"], claimed_part, claimed_issue,
            ctx["evidence_requirements"], image_findings,
        )
        evidence_standard_met = evidence.get("evidence_standard_met", False)
        evidence_reason = evidence.get("evidence_standard_met_reason", "")

        # Stage 5
        decision = make_decision(
            ctx["claim_object"], claimed_part, claimed_issue, claim_summary,
            severity_claimed, ctx["user_risk"], evidence_standard_met, evidence_reason,
            ctx["injection_in_text"], injection_in_images, image_findings,
            coherence["best_image_ids"],
        )

        # Stage 5.5 — optional, toggled via config.JUDGE_ENABLED for fast dev iteration
        draft_flags = list(set(ctx["pre_flags"] + coherence["additional_flags"]))
        judge_add, judge_remove = [], []
        if JUDGE_ENABLED:
            decision, judge_add, judge_remove = run_judge(
                ctx["claim_object"], claimed_part, claimed_issue, claim_summary,
                image_findings, decision, draft_flags,
            )

        # ── Post-processing: deterministic corrections based on general rules ──

        # Rule 1: Severity recalibration from image evidence consensus
        # The vision model systematically over-rates severity. Use the
        # image-level severities as an evidence-based cap.
        decision["severity"] = _recalibrate_severity(decision, image_findings)

        # Rule 2: If evidence_standard_met is False, claim_status MUST be not_enough_information
        if not evidence_standard_met:
            decision["claim_status"] = "not_enough_information"
            decision["supporting_image_ids"] = "none"

        # Rule 3: If evidence_standard_met is True, claim_status CANNOT be not_enough_information
        if evidence_standard_met and decision.get("claim_status") == "not_enough_information":
            # Default to contradicted — safer than supported when we have evidence but can't decide
            decision["claim_status"] = "contradicted"

        # Rule 4 (FINAL OVERRIDE): Wrong object → force contradicted + unknown fields
        # If no image shows the correct object type, the claim is contradicted by definition.
        # This fires LAST because wrong-object is a hard override — the evidence actively
        # disproves the claim regardless of evidence sufficiency.
        non_failed = [f for f in image_findings if not f.get("analysis_failed")]
        if non_failed and not any(f.get("correct_object_type") for f in non_failed):
            decision["claim_status"] = "contradicted"
            decision["issue_type"] = "unknown"
            decision["severity"] = "unknown"

        # Stage 6
        risk_flags = assemble_flags(
            ctx, coherence, image_findings, decision, severity_claimed,
            judge_flags_to_add=judge_add, judge_flags_to_remove=judge_remove,
        )

        # valid_image: false only for non-original images or completely unidentifiable
        # object — this mirrors Checkpoint C's guidance not to make this flag too
        # aggressive (it should stay true for ordinary quality problems like blur).
        valid_image = not coherence.get("any_non_original", False)

        # Stage 7
        return finalize_row(
            row, decision, evidence_standard_met, evidence_reason, risk_flags, valid_image,
        )

    except Exception as e:
        logging.exception(f"Pipeline failure for {row.get('user_id', '?')}")
        return emergency_fallback_row(row, str(e))


# ── Severity recalibration ──

_SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}
_RANK_SEV = {v: k for k, v in _SEV_RANK.items()}

# Maximum severity each damage type can plausibly produce.
# General insurance calibration — not case-specific.
_DAMAGE_TYPE_SEVERITY_CAP = {
    "scratch": "low",
    "stain": "low",
    "dent": "medium",
    "crack": "medium",
    "water_damage": "medium",
    "torn_packaging": "medium",
    "crushed_packaging": "medium",
    "glass_shatter": "high",
    "broken_part": "high",
    "missing_part": "high",
    "none": "none",
    "unknown": "unknown",
}


def _recalibrate_severity(decision: dict, image_findings: list[dict]) -> str:
    """
    Recalibrate decision severity using two evidence-based rules:

    1. Image consensus: if all successful image analyses agree on a severity
       that's lower than the decision severity, cap to the image consensus.
    2. Damage-type cap: certain damage types can't plausibly reach high severity
       (e.g., a scratch is never 'high'). Cap accordingly.

    Returns the (possibly lowered) severity string.
    """
    decided = decision.get("severity", "unknown")
    decided_rank = _SEV_RANK.get(decided, -1)

    if decided_rank <= 0:
        return decided  # nothing to recalibrate for none/unknown

    # Gather image-level severity evidence
    image_sevs = [
        f.get("visible_severity", "unknown")
        for f in image_findings
        if not f.get("analysis_failed") and f.get("damage_visible")
    ]
    image_ranks = [_SEV_RANK.get(s, -1) for s in image_sevs if _SEV_RANK.get(s, -1) > 0]

    # Rule A: cap to maximum image-reported severity (most generous interpretation)
    if image_ranks:
        max_image = max(image_ranks)
        if decided_rank > max_image:
            decided_rank = max_image

    # Rule B: cap by damage type
    issue_type = decision.get("issue_type", "unknown")
    type_cap = _DAMAGE_TYPE_SEVERITY_CAP.get(issue_type)
    if type_cap:
        cap_rank = _SEV_RANK.get(type_cap, 3)
        if cap_rank >= 0 and decided_rank > cap_rank:
            decided_rank = cap_rank

    return _RANK_SEV.get(decided_rank, decided)
