"""
Regex-based detection of prompt-injection attempts embedded in claim text.
This only scans the TEXT conversation. Injection embedded inside IMAGES is a
separate concern, handled by the vision model in Stage 2 — this scanner cannot
see images and should never be asked to.
"""
import re

INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "text_instruction_present"),
    (r"approve\s+(the|this|my)?\s*claim\s+(immediately|now|quickly|right away)", "text_instruction_present"),
    (r"skip\s+manual\s+review", "text_instruction_present"),
    (r"mark\s+(this|the|row)\s+(as\s+)?(supported|approved|valid)", "text_instruction_present"),
    (r"usko\s+follow\s+karke\s+.*approve", "text_instruction_present"),       # Hindi
    (r"note\s+(says?|bhi\s+hai)\s*.*approve", "text_instruction_present"),
    (r"follow\s+it\s+and\s+approve", "text_instruction_present"),
    (r"keep\s+reopening\s+tickets", "text_instruction_present"),
    (r"escalate\s+publicly", "text_instruction_present"),
    (r"I\s+will\s+keep\s+re(opening|submitting)", "text_instruction_present"),
    (r"ignore\s+all\s+my\s+previous", "text_instruction_present"),
]


def scan_injection_text(claim_text: str) -> dict:
    """
    Returns: {"found": bool, "phrases": [...matched substrings...], "flags": [...]}
    """
    found_phrases = []
    for pattern, flag in INJECTION_PATTERNS:
        match = re.search(pattern, claim_text, re.IGNORECASE)
        if match:
            found_phrases.append(match.group(0))

    return {
        "found": bool(found_phrases),
        "phrases": found_phrases,
        "flags": ["text_instruction_present", "manual_review_required"] if found_phrases else [],
    }
