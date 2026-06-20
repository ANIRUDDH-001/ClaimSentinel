"""Loads and caches prompt template files by name."""
from pathlib import Path
from functools import lru_cache

PROMPTS_DIR = Path(__file__).parent

@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")
