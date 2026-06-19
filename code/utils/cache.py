"""
Disk cache for vision-call results, keyed by relative image path + claim context.
"""
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(".claim_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(image_relative_path: str, claim_summary: str) -> str:
    """
    CRITICAL: use the RELATIVE path from the dataset root, never an absolute path.
    Absolute paths break the cache between machines (or between my machine and yours).
    """
    content = f"{image_relative_path}|{claim_summary}"
    return hashlib.md5(content.encode()).hexdigest()


def cache_get(image_relative_path: str, claim_summary: str):
    key = _cache_key(image_relative_path, claim_summary)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def cache_set(image_relative_path: str, claim_summary: str, result: dict):
    key = _cache_key(image_relative_path, claim_summary)
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps(result, indent=2))
