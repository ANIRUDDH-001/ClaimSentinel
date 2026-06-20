# Operational Analysis and Evaluation Report

## 1. Evaluation Summary

The pipeline was evaluated against `dataset/sample_claims.csv` (20 labeled rows).

**Overall `claim_status` Accuracy: 70.0% (14/20 correct)**

**Per-Class Metrics (`claim_status`)**
| Class | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| **Supported** | 0.786 | 0.917 | 0.846 |
| **Contradicted** | 0.333 | 0.200 | 0.250 |
| **Not Enough Information** | 0.667 | 0.667 | 0.667 |

**Macro F1 Score: 0.588**

*Note: The system achieves high accuracy for valid claims, but struggles heavily with accurately identifying contradicted claims (often confusing them with not_enough_information), which drags the Macro F1 down to 0.588.*

---

## 2. Token Usage and Cost Estimates

The pipeline processed a total of **64 claims** across the sample set (20 rows) and the test set (44 rows).
With an average of ~1.5 images per claim, the system analyzed roughly **96 images**.

### Model Calls Breakdown
- **Text Analysis (Groq Llama 3 models)**: ~2 calls per claim (Stage 1 Router, Stage 3 Coherence) = **~128 calls**
- **Vision Analysis (Gemini Flash-Lite)**: 1 call per image = **~96 calls**

### Approximate Token Usage
- **Text (Groq)**: 
  - Input: ~64,000 tokens (avg 500/call)
  - Output: ~12,800 tokens (avg 100/call)
- **Vision (Gemini)**: 
  - Input: ~72,000 tokens (avg 500 text + 258 per image)
  - Output: ~9,600 tokens (avg 100/call)

### Cost Estimate
Using standard API pricing assumptions:
- **Groq Llama 3.3 70B**: ~$0.59 / 1M input, ~$0.79 / 1M output
  - Text Cost: $0.037 (input) + $0.010 (output) = **$0.047**
- **Gemini Flash/Flash-Lite**: ~$0.075 / 1M input, ~$0.30 / 1M output
  - Vision Cost: $0.005 (input) + $0.002 (output) = **$0.007**
- **Total Cost for 64 Claims**: **~ $0.054**

---

## 3. Operational Strategy

### Latency and Runtime
- The entire 44-row test set takes approximately **4.5 minutes** to execute sequentially.
- **Bottleneck**: The Google Gemini Free Tier limits vision calls to 15 Requests Per Minute (RPM) and enforces strict Requests Per Day (RPD) limits. 

### API Limits & Strategy (Throttling, Caching, Retry)
1. **Model Pooling**: To survive strict limits, we implemented a custom `ModelPool` (`utils/model_pool.py`). The system registers 5 separate free-tier Gemini models and rotates through them if one is exhausted (ResourceExhausted / 429 errors).
2. **Token Bucket Rate Limiting**: We injected a strict 15 RPM token bucket limiter inside `utils/api_client.py` to gracefully pace the execution and avoid being penalized with heavy exponential backoffs.
3. **Caching Strategy**: A persistent disk cache (`.claim_cache/`) stores SHA-256 hashed image-result pairs. Because the evaluation script is often run repeatedly during development, caching avoids re-billing for identical images and immediately bypasses the 15 RPM vision limit for previously seen files.
4. **Conditional Execution**: Stage 3 (Coherence) is entirely bypassed if Stage 1 or Stage 2 confidently determine that a claim is missing required information, saving unnecessary LLM cycles.
