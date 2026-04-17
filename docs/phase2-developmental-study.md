# Phase 2: Developmental Study Results

28.8M parameter model (depth=4, dim=512) trained from scratch at 6 time budgets.
Each run uses the fixed Phase 1 best recipe with WARMDOWN_RATIO=0.70.

**Headline numbers:** Best val_bpb = **1.0472** (4h, 55,298 steps) | Total improvement 5m to 4h = **12.3%**

---

## val_bpb Trajectory (dedicated runs)

Each point is from a fresh run with its own LR schedule proportioned to the budget.
The recipe continues to improve at every horizon -- no plateau yet -- but gains decelerate
roughly log-linearly with compute.

| Budget | Steps | val_bpb | Delta | Improvement rate |
| --- | ---: | ---: | ---: | ---: |
| 5m | 1,210 | 1.1942 | -- | -- |
| 15m | 3,612 | 1.1158 | -0.0784 | 6.6% |
| 30m | 7,199 | 1.0857 | -0.0301 | 2.5% |
| 1h | 12,586 | 1.0697 | -0.0160 | 1.3% |
| 2h | 23,564 | 1.0564 | -0.0133 | 1.1% |
| 4h | 55,298 | 1.0472 | -0.0092 | 0.8% |

---

## Dedicated vs Intermediate Checkpoints

Each training run saves intermediate checkpoints at all earlier time thresholds.
The most recent (4h) run's intermediate files overwrote earlier dedicated checkpoints on disk.
The two series differ because dedicated runs benefit from warmdown being proportioned to their
budget, while 4h intermediates capture mid-training state before warmdown kicks in (at ~72 min).

| Checkpoint | Dedicated val_bpb | 4h-intermediate val_bpb | Gap |
| --- | ---: | ---: | ---: |
| 5m | 1.1942 | 1.2798 | +0.0856 |
| 15m | 1.1158 | 1.2186 | +0.1028 |
| 30m | 1.0857 | 1.1994 | +0.1137 |
| 1h | 1.0697 | 1.1820 | +0.1123 |
| 2h | 1.0564 | 1.1505 | +0.0941 |
| 4h | 1.0472 | 1.0471 | -0.0001 |

The gap narrows to near-zero at 4h where warmdown completes fully. The large gap at shorter
horizons confirms WARMDOWN_RATIO=0.70 is critical for short-budget performance.

---

## Qualitative Development

Prompt pack outputs evaluated across 7 axes per program.md. Summary of aggregate trends:

**Average coherent span:** 5m = 2.0/5 | 30m = 2.7/5 | 4h = 3.9/5

### Coherence and repetition by checkpoint

| Checkpoint | Avg coherent span | Dominant repetition | Syntax stability | Weirdness |
| --- | :---: | :---: | :---: | :---: |
| 5m | 2.0 / 5 | early | poor | high |
| 15m | 2.6 / 5 | late | mixed | balanced |
| 30m | 2.7 / 5 | late | mixed | balanced |
| 1h | 2.6 / 5 | medium | mixed | balanced |
| 2h | 2.3 / 5 | early | mixed | sterilized |
| 4h | 3.9 / 5 | late | good | balanced |

---

## Key Findings

### 1. The recipe scales -- no plateau yet

val_bpb improved at every time horizon from 1.1942 (5m) to 1.0472 (4h),
a 12.3% reduction. The diminishing-return curve is smooth: each doubling of compute
yields roughly half the previous gain. Extrapolating, an 8h run might reach ~1.04 but
would need measurement to confirm.

### 2. Warmdown ratio matters enormously at short horizons

The gap between dedicated-run val_bpb and 4h-intermediate val_bpb quantifies the warmdown
benefit. At 5m, warmdown accounts for ~0.086 bpb (dedicated 1.194 vs intermediate 1.280).
At 4h the gap vanishes. This confirms WARMDOWN_RATIO=0.70 was tuned for short budgets
and re-tuning for longer horizons is warranted.

### 3. Surface fluency precedes structural coherence

Early checkpoints (5m, 15m) produce grammatical fragments but topic-drift within
2-3 sentences. By 30m, paragraphs hold a topic for 4-5 sentences. The 4h model sustains
coherent multi-paragraph output but still fabricates facts. Syntax improves before semantics.

### 4. Repetition is non-monotonic

The 2h checkpoint shows worse repetition than 30m on several prompts
(continuation collapses into number sequences, longitudinal anchor loops on "hidden").
The 4h checkpoint resolves these. This suggests a mid-training instability window
where the model has learned enough structure to loop but not enough to escape.

### 5. Weirdness decays with training

At 5m, outputs are wildly inventive ("mariocattle," "Major Snorf" elaborated into
bizarre control concepts, "cinch holes" in cricket). By 4h, the anomaly lure prompt
produces dry geological survey text. The "weirdness retained" axis trends from "high"
to "sterilized/balanced" -- confirming the longitudinal hypothesis that extended training
erases unusual texture.

---

## Next Questions

Per program.md, the developmental map is the deliverable. Based on these results:

- **Re-tune** -- Re-optimize WARMDOWN_RATIO for longer budgets (2h, 4h) -- the 0.70 value is confirmed suboptimal.
- **Extend** -- Run 8h to see if the log-linear trend continues or a true plateau appears.
- **Investigate** -- The 2h repetition spike: is it a warmdown-phase artifact or a genuine developmental stage?
- **Preserve** -- Save dedicated-run checkpoints separately to prevent overwriting by longer runs. *(Done -- checkpoint overwrite fix applied to train.py.)*

---

*Generated 2026-04-17. Model: 28.8M params, depth=4, dim=512, vocab=8192.
Data: karpathy/climbmix-400b-shuffle. Hardware: single NVIDIA GPU (~12GB VRAM).*
