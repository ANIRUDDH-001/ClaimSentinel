# HackerRank Orchestrate - ClaimSentinel

A multi-modal AI pipeline to verify damage claims based on user conversations and image evidence.

## Architecture

The system processes each claim through a 3-Stage Pipeline:

1. **Stage 1: Text Analysis (Router)**  
   Analyzes the user's chat transcript to extract the core claim (what object, what part, what issue) and evaluates their historical risk profile.
2. **Stage 2: Vision Analysis (Image-by-Image)**  
   Evaluates each provided image independently against the claimed damage. It checks for image validity, authenticity (manipulation flags), visible damage, and severity.
3. **Stage 3: Coherence & Synthesis (Judge)**  
   Aggregates the findings from the text claim, user history, and all analyzed images to make a final ruling: `supported`, `contradicted`, or `not_enough_information`.

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
