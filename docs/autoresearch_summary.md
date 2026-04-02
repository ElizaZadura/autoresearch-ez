# Autoresearch Run Summary

## What Worked

- **Smaller total batch size**
  - Best regime: `TOTAL_BATCH_SIZE = 2^15`

- **Long LR warmdown**
  - Strong improvements up to ~**70% warmdown ratio**
  - Linear decay > cosine

- **Shallower model**
  - `depth=4, dim=512` performed best
  - More steps > more depth under current constraints

- **Attention pattern**
  - `WINDOW_PATTERN = LLLL` (full attention)

- **Learning rates**
  - `MATRIX_LR ≈ 0.03–0.04`
  - `EMBEDDING_LR ≈ 1.2–1.3`

- **Regularization**
  - `WEIGHT_DECAY ≈ 0.08`

- **Training dynamics tweaks**
  - Faster Muon momentum warmup (~100 steps)
  - Slightly lower momentum target (~0.93)

---

## What Didn’t Work

- Increasing model depth (e.g. 8 → 10)
- Increasing device batch size (VRAM bottlenecks, fewer steps)
- Too small batch without compensation (instability)
- Cosine warmdown (worse than linear)
- LR warmup (small % added hurt performance)
- Over-extending warmdown beyond sweet spot in some configs
- Final LR → 0 (worse than small non-zero)
- Larger SwiGLU expansion
- GQA (n_kv_head changes)
- Enabling value embeddings on all layers
- Excessive LR tweaks beyond local optimum

---

## Key Insight

> Under constrained compute/time:
>
> **More optimization steps (via smaller model + batch)**
> beats
> **higher model capacity (deeper networks)**

---

## Current Best Known Setup (for this machine/run budget)

```
depth = 4
dim = 512
TOTAL_BATCH_SIZE = 2^15
WINDOW_PATTERN = LLLL
WARMDOWN_RATIO = 0.70
MATRIX_LR ≈ 0.04
EMBEDDING_LR ≈ 1.3
WEIGHT_DECAY ≈ 0.08
```

---

## If Continuing Exploration

Next meaningful directions (not more micro-tweaks):

- Dataset changes (entropy, domain)
- Tokenizer adjustments
- Training duration scaling
- Objective variations

Avoid:
- endless LR micro-tuning
- minor architectural tweaks without step budget changes

---

## One-line Takeaway

> You were overpaying for depth and batch size — cutting both unlocked better learning per step.

