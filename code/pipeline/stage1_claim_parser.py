"""Stage 1: claim parser. One LLM text call per claim."""
from prompts.loader import load_prompt
from utils.api_client import call_text
from utils.schema_validator import validate_stage_output, repair_via_llm


def parse_claim(claim_object: str, claim_text: str) -> dict:
    prompt = load_prompt("stage1_claim_parser").format(
        claim_object=claim_object,
        claim_text=claim_text,
    )

    result = call_text("claim_parser", prompt)
    is_valid, repaired = validate_stage_output("claim_parser", result)

    if not is_valid and not repaired.get("primary_issue_type"):
        repaired = repair_via_llm("claim_parser", repaired, prompt)

    return repaired
