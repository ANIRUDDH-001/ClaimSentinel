"""Stage 2: per-image vision analysis. One LLM vision call per image."""
from pathlib import Path
from prompts.loader import load_prompt
from utils.api_client import call_vision
from utils.schema_validator import validate_stage_output
from config import DATASET_DIR


def analyze_image(image_path: str, claim_object: str, claimed_part: str,
                  claimed_issue: str, claim_summary: str) -> dict:
    image_id = Path(image_path).stem

    prompt = load_prompt("stage2_image_analyzer").format(
        claim_object=claim_object,
        claimed_part=claimed_part,
        claimed_issue=claimed_issue,
        claim_summary=claim_summary,
        image_id=image_id,
    )

    result = call_vision(prompt, image_path, base_dir=DATASET_DIR,
                         claim_summary=claim_summary)

    if not result:
        return _empty_image_findings(image_path)

    _, repaired = validate_stage_output("image_analyzer", result)
    repaired["image_id"] = image_id  # Never trust the model's own echo of this field
    return repaired


def _empty_image_findings(image_path: str) -> dict:
    """
    Safe fallback when a vision call cannot complete at all — image unreadable, or
    every vision model exhausted. Every field defaults to the "can't confirm" side,
    so downstream stages never treat a failure as positive evidence. Stage 3 checks
    `analysis_failed` to exclude this image from coherence comparisons.
    """
    image_id = Path(image_path).stem
    return {
        "image_id": image_id,
        "object_detected": "unknown — analysis failed",
        "correct_object_type": False,
        "claimed_part_visible": False,
        "part_visibility_reason": "unclear",
        "damage_visible": False,
        "visible_damage_type": "unknown",
        "damage_matches_claim": "unknown",
        "visible_severity": "unknown",
        "image_quality": "cropped_or_obstructed",
        "non_original_suspected": False,
        "non_original_reason": None,
        "text_found_in_image": False,
        "text_quoted": None,
        "text_is_injection_attempt": False,
        "findings_narrative": "Image analysis failed; treat as insufficient evidence.",
        "image_risk_flags": ["damage_not_visible"],
        "analysis_failed": True,
    }
