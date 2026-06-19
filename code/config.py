# config.py — No secrets here. All API keys via env vars.
import os

# Paths
DATASET_DIR   = "dataset"
IMAGES_DIR    = os.path.join(DATASET_DIR, "images")
CLAIMS_CSV    = os.path.join(DATASET_DIR, "claims.csv")
SAMPLE_CSV    = os.path.join(DATASET_DIR, "sample_claims.csv")
HISTORY_CSV   = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_CSV  = os.path.join(DATASET_DIR, "evidence_requirements.csv")
OUTPUT_CSV    = "output.csv"
CACHE_DIR     = ".claim_cache"
LOG_DIR       = os.path.expanduser("~/hackerrank_orchestrate")

# Model selection — verify with a live models.list() call if these ever stop resolving
GEMINI_PRIMARY = "gemini-3.1-flash-lite"
GROQ_PRIMARY   = "llama-3.3-70b-versatile"

# Agent config
AGENT_TEMPERATURE = 0.1
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 4.0   # seconds

# Judge config
JUDGE_ENABLED = True       # Set False to disable judge and speed up dev iteration
