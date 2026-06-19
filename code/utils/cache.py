"""Disk cache for vision calls, keyed by relative path"""

_MEM_CACHE = {}

def cache_get(image_path: str, claim_summary: str) -> dict:
    return _MEM_CACHE.get((image_path, claim_summary))

def cache_set(image_path: str, claim_summary: str, result: dict):
    _MEM_CACHE[(image_path, claim_summary)] = result
