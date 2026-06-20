# HackerRank Orchestrate - ClaimSentinel

A multi-modal AI pipeline to verify damage claims based on user conversations and image evidence.

## Architecture
A 9-stage pipeline, run once per claim:
0. Pre-check (pure Python) — injection scan, user risk lookup, evidence requirement lookup
1. Claim parser (LLM, text) — extracts the actual damage claim from the conversation
2. Image analyzer (LLM, vision) — one call per submitted image
3. Cross-image coherence (pure Python) — consistency checks, best-image selection
4. Evidence evaluator (LLM, text) — is the image evidence sufficient to decide at all?
5. Decision maker (LLM, text) — the core claim_status / issue_type / severity decision
5.5. LLM-as-judge (LLM, text, different model family) — adversarial review of Stage 5
6. Flag assembler (pure Python) — final risk_flags union and co-occurrence rules
7. Validator (Pydantic) — final schema enforcement with a safe emergency fallback

## Models Used
- **Groq Llama 3 (Text)**: Lightning fast text processing for Stages 1 and 3.
- **Google Gemini Flash (Vision)**: Cost-effective multi-modal analysis for Stage 2. 
  - *Note: The system implements an automated model-pool router and token bucket rate limiter to gracefully handle the strict 15 RPM limits on the Gemini free tier.*

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

To run the pipeline on the test set (`dataset/claims.csv`) and produce `output.csv`:

```bash
python code/main.py
```

## Running the Evaluation

To evaluate the pipeline against labeled ground truth (`dataset/sample_claims.csv`) and print the F1/Precision/Recall metrics:

```bash
python code/evaluation/main.py
```

## Caching
The system heavily caches vision API responses to `.claim_cache/` using SHA-256 hashes of the images. This drastically speeds up iterative development and bypasses rate limits on repeated runs. To perform a fully clean run, delete the `.claim_cache/` directory.
