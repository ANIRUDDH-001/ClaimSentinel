"""
Single entry point for all LLM calls.
Internally routes to Gemini or Groq based on model provider.
Handles: rate limiting, retry with exponential backoff, RPD tracking.
"""
import json
import time
import logging
from pathlib import Path
from PIL import Image
import os
import google.generativeai as genai
from groq import Groq as GroqClient
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from utils.model_pool import ModelPool, ModelSpec
from utils.rate_limiter import rate_limiter
from utils.cache import cache_get, cache_set

KEYS = [
    os.environ.get("GOOGLE_API_KEY", ""),
    os.environ.get("GOOGLE_API_KEY_2", ""),
    os.environ.get("GOOGLE_API_KEY_3", ""),
    os.environ.get("GOOGLE_API_KEY_4", ""),
]
KEYS = [k for k in KEYS if k]

key_idx = 0
if KEYS:
    genai.configure(api_key=KEYS[0])

def rotate_key():
    global key_idx
    key_idx = (key_idx + 1) % len(KEYS)
    genai.configure(api_key=KEYS[key_idx])
    logging.warning(f"Rotated Gemini API Key to index {key_idx}")

_groq = GroqClient(api_key=os.environ["GROQ_API_KEY"])

pool = ModelPool.get()

MAX_RETRIES = 3


class RateLimitError(Exception):
    """Unified rate limit exception across providers."""
    pass


def call_text(stage: str, prompt: str) -> dict:
    """
    Route a text prompt to the appropriate model for the given stage.
    Returns parsed JSON dict. Never raises — returns {} on complete failure.
    """
    spec = pool.get_text_model(stage)
    if spec is None:
        logging.error(f"[{stage}] No text model available")
        return {}
    return _call_with_retry(spec, prompt, image=None)


def call_vision(prompt: str, image_path: str, base_dir: str,
                claim_summary: str = "") -> dict:
    """
    Route a vision call to the best available Gemini model.
    Caches results by (relative_image_path, claim_summary).
    Returns parsed JSON dict.
    """
    cached = cache_get(image_path, claim_summary)
    if cached:
        logging.debug(f"Cache hit: {image_path}")
        return cached

    spec = pool.get_vision_model()
    if spec is None:
        from pipeline.stage2_image_analyzer import _empty_image_findings
        return _empty_image_findings(image_path)

    full_path = Path(base_dir) / image_path
    try:
        image = Image.open(full_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        logging.error(f"Image load failed: {full_path} — {e}")
        from pipeline.stage2_image_analyzer import _empty_image_findings
        return _empty_image_findings(image_path)

    result = _call_with_retry(spec, prompt, image=image)

    if result:
        cache_set(image_path, claim_summary, result)

    return result


def _call_with_retry(spec: ModelSpec, prompt: str, image=None,
                     _excluded: set = None) -> dict:
    _excluded = _excluded or set()
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # rate_limiter.wait(spec.model_id, spec.rpm)

            if spec.provider == "gemini":
                result = _call_gemini(spec.model_id, prompt, image)
            else:
                result = _call_groq(spec.model_id, prompt)

            pool.record_usage(spec.model_id)
            return result

        except (ResourceExhausted, RateLimitError) as e:
            logging.warning(f"Rate limit on {spec.model_id}. Rotating API key.")
            rotate_key()
            time.sleep(2.0)
            last_error = e

        except ServiceUnavailable as e:
            time.sleep(4.0)
            last_error = e

        except Exception as e:
            logging.error(f"Unexpected error on {spec.model_id}: {e}")
            # Mark model as exhausted so the pool stops returning it
            _mark_model_exhausted(spec.model_id)
            last_error = e
            break

    logging.warning(f"All retries failed for {spec.model_id}. Trying next model.")
    _excluded.add(spec.model_id)

    if image is not None:
        next_spec = _get_next_vision_model(exclude=spec.model_id)
    else:
        next_spec = _get_next_text_model(exclude=_excluded)

    if next_spec:
        return _call_with_retry(next_spec, prompt, image, _excluded=_excluded)

    return {}



_usage_log = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "images": 0}

def get_usage_summary() -> dict:
    return dict(_usage_log)

def _call_gemini(model_id: str, prompt: str, image=None) -> dict:
    model = genai.GenerativeModel(
        model_name=model_id,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json",
        )
    )

    if image is not None:
        response = model.generate_content([image, prompt])
    else:
        response = model.generate_content(prompt)

    usage = getattr(response, "usage_metadata", None)
    if usage:
        _usage_log["input_tokens"] += getattr(usage, "prompt_token_count", 0)
        _usage_log["output_tokens"] += getattr(usage, "candidates_token_count", 0)
    _usage_log["calls"] += 1
    if image is not None:
        _usage_log["images"] += 1

    return _safe_json_parse(response.text, model_id)


def _call_groq(model_id: str, prompt: str) -> dict:
    response = _groq.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
        max_tokens=1024,
    )
    
    usage = getattr(response, "usage", None)
    if usage:
        _usage_log["input_tokens"] += getattr(usage, "prompt_tokens", 0)
        _usage_log["output_tokens"] += getattr(usage, "completion_tokens", 0)
    _usage_log["calls"] += 1

    return _safe_json_parse(response.choices[0].message.content, model_id)


def _safe_json_parse(text: str, model_id: str) -> dict:
    """
    Handles JSON wrapped in markdown fences (Gemini does this sometimes
    even with response_mime_type set).
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.error(f"[{model_id}] JSON parse failed. Raw: {text[:300]}")
        return {}


def _get_next_vision_model(exclude: str):
    """Find next available vision model, excluding the failed one."""
    from utils.model_pool import GEMINI_MODELS
    for spec in GEMINI_MODELS:
        if spec.model_id != exclude and spec.rpd_safe:
            return spec
    return None


def _get_next_text_model(exclude: set):
    """Find next available text model, excluding all failed ones.
    Tries Groq models first, then falls back to Gemini."""
    from utils.model_pool import GROQ_MODELS, GEMINI_MODELS
    for spec in GROQ_MODELS:
        if spec.model_id not in exclude and spec.rpd_safe:
            return spec
    # Last resort: Gemini flash-lite for text
    for spec in GEMINI_MODELS:
        if spec.model_id not in exclude and spec.rpd_safe:
            return spec
    return None


def _mark_model_exhausted(model_id: str):
    """Mark a model as exhausted so the pool stops returning it this run."""
    spec = pool._all.get(model_id)
    if spec:
        spec.used_rpd = spec.rpd  # Saturate to prevent further selection
        logging.warning(f"Marked {model_id} as exhausted (TPD/RPD limit hit)")

