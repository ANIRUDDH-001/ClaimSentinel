"""Stage 4: evidence sufficiency evaluator. One LLM text call per claim."""
from prompts.loader import load_prompt
from utils.api_client import call_text
from utils.schema_validator import validate_stage_output


def evaluate_evidence(claim_object: str, claimed_part: str, claimed_issue: str,
                      evidence_requirements: list[dict], image_findings: list[dict]) -> dict:
    requirements_text = _format_requirements(claim_object, evidence_requirements)
    findings_summary = _format_findings_summary(image_findings)
    n_with_part = sum(1 for f in image_findings if f.get("claimed_part_visible"))

    prompt = load_prompt("stage4_evidence").format(
        claim_object=claim_object,
        claimed_part=claimed_part,
        claimed_issue=claimed_issue,
        requirements_text=requirements_text,
        image_findings_summary=findings_summary,
        n_images=len(image_findings),
        n_with_part=n_with_part,
    )

    result = call_text("evidence", prompt)
    _, repaired = validate_stage_output("evidence", result)
    return repaired


def _format_requirements(claim_object: str, requirements: list[dict]) -> str:
    if not requirements:
        return "No specific evidence requirement on file for this object type."
    lines = [
        f"- {r['applies_to']}: {r['minimum_image_evidence']}"
        for r in requirements
    ]
    return "\n".join(lines)


def _format_findings_summary(image_findings: list[dict]) -> str:
    lines = []
    for f in image_findings:
        lines.append(
            f"- {f.get('image_id', '?')}: part_visible={f.get('claimed_part_visible')}, "
            f"damage_visible={f.get('damage_visible')}, quality={f.get('image_quality')}"
        )
    return "\n".join(lines) if lines else "No image findings available."
