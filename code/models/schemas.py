from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
from models.allowed_values import OBJECT_PART_MAP, ALLOWED_RISK_FLAGS, OUTPUT_CSV_COLUMNS


class OutputRow(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: Literal["car", "laptop", "package"]
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: Literal["dent", "scratch", "crack", "glass_shatter", "broken_part",
                        "missing_part", "torn_packaging", "crushed_packaging",
                        "water_damage", "stain", "none", "unknown"]
    object_part: str
    claim_status: Literal["supported", "contradicted", "not_enough_information"]
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: Literal["none", "low", "medium", "high", "unknown"]

    @field_validator("evidence_standard_met", "valid_image", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    @field_validator("object_part")
    @classmethod
    def validate_object_part(cls, v, info):
        claim_object = info.data.get("claim_object")
        if claim_object and claim_object in OBJECT_PART_MAP:
            allowed = OBJECT_PART_MAP[claim_object]
            if v not in allowed:
                return "unknown"
        return v

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v):
        if v == "none":
            return v
        parts = [p.strip() for p in v.split(";")]
        valid_parts = [p for p in parts if p in ALLOWED_RISK_FLAGS]
        return ";".join(valid_parts) if valid_parts else "none"

    @field_validator("claim_status_justification")
    @classmethod
    def justification_quality(cls, v):
        if not v or len(v.strip()) < 25:
            raise ValueError("Justification too short — must reference visual evidence.")
        return v.strip()

    @model_validator(mode="after")
    def enforce_consistency(self):
        if not self.evidence_standard_met:
            self.claim_status = "not_enough_information"
            self.supporting_image_ids = "none"
            self.severity = "unknown"
        if self.claim_status == "not_enough_information":
            self.supporting_image_ids = "none"
        return self

    def to_csv_row(self) -> dict:
        """Return dict in the exact CSV column order required by problem_statement.md."""
        full = {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
            "evidence_standard_met": str(self.evidence_standard_met).lower(),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": self.risk_flags,
            "issue_type": self.issue_type,
            "object_part": self.object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": self.supporting_image_ids,
            "valid_image": str(self.valid_image).lower(),
            "severity": self.severity,
        }
        return {col: full[col] for col in OUTPUT_CSV_COLUMNS}
