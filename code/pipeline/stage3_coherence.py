"""Stage 3: cross-image coherence check. Pure Python, no API calls."""


def cross_image_coherence(image_findings: list[dict], claimed_part: str) -> dict:
    if len(image_findings) <= 1:
        return {"additional_flags": [], "best_image_ids": _select_best(image_findings)}

    additional_flags = []

    objects_correct = [f["correct_object_type"] for f in image_findings if not f.get("analysis_failed")]
    if objects_correct and not all(objects_correct) and any(objects_correct):
        additional_flags.append("possible_manipulation")

    if any(f.get("text_is_injection_attempt") for f in image_findings):
        additional_flags.append("text_instruction_present")

    for f in image_findings:
        for flag in f.get("image_risk_flags", []):
            if flag not in additional_flags:
                additional_flags.append(flag)

    return {
        "additional_flags": additional_flags,
        "best_image_ids": _select_best(image_findings),
        "any_part_visible": any(f.get("claimed_part_visible") for f in image_findings),
        "any_damage_matching": any(f.get("damage_matches_claim") == "yes" for f in image_findings),
        "any_non_original": any(f.get("non_original_suspected") for f in image_findings),
        "injection_in_images": any(f.get("text_is_injection_attempt") for f in image_findings),
    }


def _select_best(image_findings: list[dict]) -> list[str]:
    """
    Prefer clear images. Exclude pure context shots where neither the claimed part
    nor damage nor injected text is present.
    """
    quality_rank = {"clear": 0, "wrong_angle": 1, "cropped_or_obstructed": 2,
                    "low_light_or_glare": 3, "blurry": 4, "unknown": 5}

    relevant = [
        f for f in image_findings
        if f.get("claimed_part_visible") or f.get("damage_visible") or f.get("text_is_injection_attempt")
    ]

    if not relevant:
        return []

    sorted_relevant = sorted(relevant, key=lambda f: quality_rank.get(f.get("image_quality", "unknown"), 5))
    return [f["image_id"] for f in sorted_relevant]
