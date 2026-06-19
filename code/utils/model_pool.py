"""
ModelPool: Tracks per-model RPD usage and routes calls to available models.
All state is in-process (no Redis, no DB). Reset each run.
"""
from __future__ import annotations
import logging
from threading import Lock
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelSpec:
    provider: str           # "gemini" | "groq"
    model_id: str
    rpm: int
    tpm: int
    rpd: int
    vision: bool = False
    used_rpd: int = field(default=0, init=False)

    @property
    def rpd_available(self) -> int:
        return max(0, self.rpd - self.used_rpd)

    @property
    def rpd_safe(self) -> bool:
        # Keep 5% safety buffer
        return self.used_rpd < int(self.rpd * 0.95)


GEMINI_MODELS: list[ModelSpec] = [
    # Ordered by RPD descending (prefer highest daily budget)
    ModelSpec("gemini", "gemini-3.1-flash-lite",              rpm=15, tpm=250_000, rpd=500, vision=True),
    ModelSpec("gemini", "gemini-2.5-flash-lite",              rpm=10, tpm=250_000, rpd=20,  vision=True),
    ModelSpec("gemini", "gemini-2.5-flash",                   rpm=5,  tpm=250_000, rpd=20,  vision=True),
    ModelSpec("gemini", "gemini-3-flash",                     rpm=5,  tpm=250_000, rpd=20,  vision=True),
    ModelSpec("gemini", "gemini-3.5-flash",                   rpm=5,  tpm=250_000, rpd=20,  vision=True),
]

GROQ_MODELS: list[ModelSpec] = [
    ModelSpec("groq", "llama-3.3-70b-versatile",                  rpm=30, tpm=12_000,  rpd=1000, vision=False),
    ModelSpec("groq", "meta-llama/llama-4-scout-17b-16e-instruct", rpm=30, tpm=30_000,  rpd=1000, vision=False),
    ModelSpec("groq", "qwen/qwen3-32b",                           rpm=60, tpm=6_000,   rpd=1000, vision=False),
]

# Stage-specific text model preference
STAGE_TEXT_PREFERENCE: dict[str, list[str]] = {
    "claim_parser": [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
    ],
    "evidence": [
        "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM suits longer evidence prompts
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
    ],
    "decision": [
        "llama-3.3-70b-versatile",                    # Best quality for the final decision
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
    "judge": [
        # Judge MUST use a different provider than the decision maker (cross-validation)
        "gemini-3.1-flash-lite",
    ],
}


class ModelPool:
    """
    Singleton. Routes calls to available models by stage, tracks RPD usage.
    Thread-safe via Lock (for future async support).
    """
    _instance: Optional["ModelPool"] = None

    def __init__(self):
        self._lock = Lock()
        self._all: dict[str, ModelSpec] = {
            m.model_id: m for m in GEMINI_MODELS + GROQ_MODELS
        }

    @classmethod
    def get(cls) -> "ModelPool":
        if cls._instance is None:
            cls._instance = ModelPool()
        return cls._instance

    def get_vision_model(self) -> Optional[ModelSpec]:
        """Next available vision model. Returns None if all exhausted."""
        with self._lock:
            for spec in GEMINI_MODELS:  # Already ordered by priority
                if spec.rpd_safe:
                    return spec
            logging.error("ALL VISION MODELS EXHAUSTED — will use empty_findings fallback")
            return None

    def get_text_model(self, stage: str) -> Optional[ModelSpec]:
        """Next available text model for given stage."""
        with self._lock:
            preferred = STAGE_TEXT_PREFERENCE.get(stage, [])

            for model_id in preferred:
                spec = self._all.get(model_id)
                if spec and spec.rpd_safe:
                    return spec

            if stage != "judge":
                for spec in GROQ_MODELS:
                    if spec.rpd_safe:
                        return spec

            primary_gemini = self._all.get("gemini-3.1-flash-lite")
            if primary_gemini and primary_gemini.rpd_safe:
                return primary_gemini

            logging.error(f"NO TEXT MODELS AVAILABLE for stage={stage}")
            return None

    def record_usage(self, model_id: str):
        """Call after every successful API call."""
        with self._lock:
            spec = self._all.get(model_id)
            if spec:
                spec.used_rpd += 1

    def usage_report(self) -> dict:
        with self._lock:
            return {
                m.model_id: {
                    "used": m.used_rpd,
                    "limit": m.rpd,
                    "pct": round(m.used_rpd / m.rpd * 100, 1)
                }
                for m in (GEMINI_MODELS + GROQ_MODELS)
                if m.used_rpd > 0
            }
