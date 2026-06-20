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
