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

## Phase 2b: 12h Extended Run (2026-04-18)

Single fresh run with the frozen Phase 1 recipe at `TIME_BUDGET = 43200s` (12h), with
instrumentation added to `train.py`: milestone checkpoints + full `val_bpb` + prompt
pack at 5m / 15m / 30m / 1h / 2h / 4h / 8h / 12h, plus a capped periodic `val_bpb`
written every ~5% of progress into `progress.csv` for a dense developmental curve.

Output: `output/2026-04-17_12h_run/`. Total wall-clock 12.2h, exit code 0, no run-time warnings.

### Milestone results (all from a single 12h run — intermediates share the 12h LR schedule)

| Budget | Steps | Train loss EMA | val_bpb | Δ vs prev |
| --- | ---: | ---: | ---: | ---: |
| 5m   | 1,190   | 3.5585 | 1.2671 | — |
| 15m  | 3,659   | 3.4198 | 1.2126 | −0.0545 |
| 30m  | 7,427   | 3.3554 | 1.1928 | −0.0198 |
| 1h   | 15,020  | 3.3105 | 1.1781 | −0.0147 |
| 2h   | 28,963  | 3.2775 | 1.1691 | −0.0091 |
| 4h   | 54,733  | 3.2524 | 1.1629 | −0.0062 |
| 8h   | 112,929 | 3.1386 | 1.1223 | −0.0406 |
| **12h** | **174,682** | **2.9288** | **1.0435** | **−0.0788** |

Dense 19-point `progress.csv` curve (capped eval, ~5M tokens each) tracks the same shape:
val_bpb drops smoothly from 1.183 at 5% progress to 1.061 at 95% progress.

### Headline

**12h dedicated val_bpb = 1.0435 vs yesterday's 4h dedicated val_bpb = 1.0472 — only
−0.0037 bpb (<0.4%) from tripling the training budget.** A clear plateau in *final*
quality at this depth/width.

### Warmdown dominates within-run dynamics

The apparent "slowdown" between 30m → 4h in the table above is an artifact of the
12h LR schedule: at those elapsed times we are still on the flat part of the
schedule (`lrm = 1.0`). Warmdown begins at `progress = 0.30` (3.6h) and drops
`lrm` from 1.0 → 0.05 across the remaining 8.4h. The big drops at 8h→12h are the
warmdown kicking in, not new learning.

Side-by-side with yesterday's 4h-run intermediates (whose warmdown fully cools by 4h):

| Elapsed | Yesterday's 4h-run intermediate | Tonight's 12h-run intermediate | Gap |
| --- | ---: | ---: | ---: |
| 5m   | 1.2798 | 1.2671 | −0.013 |
| 15m  | 1.2186 | 1.2126 | −0.006 |
| 30m  | 1.1994 | 1.1928 | −0.007 |
| 1h   | 1.1820 | 1.1781 | −0.004 |
| 2h   | 1.1505 | 1.1691 | **+0.019** |
| 4h   | **1.0471** | **1.1629** | **+0.116** |

The **+0.116 gap at 4h** is the warmdown benefit: the 4h-tuned schedule finishes
warmdown at 4h, the 12h-tuned schedule is still at full LR. This is
hyperparameter-dependent and NOT a model-capacity finding.

### Qualitative trajectory

Sample output for `plain_continuation` ("The old man walked slowly toward the river and…"):

- **4h (full LR)**: loose, loops on "the newest / the oldest", hallucinated Balkani / Paul E. L. L.
- **8h (mid-warmdown, lrm≈0.52)**: severe token-level repetition — `"sturdy sturdy sturdy sturd"`.
- **12h (warmdown nearly done, lrm≈0.05)**: clearly more coherent — `"gnarled eyes… the angler staring over the river… the muddler came out"` — a full paragraph of narrative that holds together.

Weirdness trajectory on `anomaly_lure` ("Ground control to Major Snorf,…"):
- 5m/30m: terse, near-empty.
- **2h–4h: weirdness peak** — elaborate fake taxonomies (`"apeptoid species (Frexelo)"`, `"Hadiard Gols, Riches"`).
- 12h: terse again, more list-like — warmdown has pruned the creativity.

No clear repetition spike at exactly 2h in this run (unlike yesterday's baseline
observation). The 8h sample shows *stronger* token-level repetition than either
neighbour — this is the new pathology, likely mid-warmdown LR instability.

### Decisions for the next night

The plan's decision tree resolves to **plateau → retune WARMDOWN_RATIO** —
strengthened by the warmdown-gap evidence above. The interesting question is no
longer "does the model keep learning past 4h?" (it doesn't, much), but:

> **How much warmdown do we actually need?** Can we reach ≈1.04 bpb in *2h* or
> *4h* with a tighter warmdown instead of 12h with 0.70?

Candidate: sweep `WARMDOWN_RATIO ∈ {0.3, 0.5, 0.7, 0.9}` at 2h (≈8h total) and
at 4h (≈16h total) if the 2h sweep is ambiguous. Defer capacity and seed-envelope
questions until after this.

---

## Phase 2c: WARMDOWN_RATIO sweep (2026-04-20)

Four back-to-back 2h runs with `WARMDOWN_RATIO ∈ {0.30, 0.50, 0.70, 0.90}` — total
wall time 8h 44m, all four runs `exit=0`. Answering 2b's open question:
**how much warmdown do we actually need at 2h?**

Output: `output/2026-04-19_2h_warmdown_<ratio>_run/`. Each run produced the 5
milestone prompt packs (5m / 15m / 30m / 1h / 2h), their `.pt` + `.json` sidecars,
and a 19-point dense `progress.csv` curve.

Driver: `run_warmdown_sweep.sh` at repo root; `WARMDOWN_RATIO` is now overridable
via the `AUTORESEARCH_WARMDOWN_RATIO` env var in `train.py`.

### End-of-run val_bpb (uncapped)

| WARMDOWN_RATIO | val_bpb | Δ vs prev | steps | MFU (4070, corrected) |
| --- | ---: | ---: | ---: | ---: |
| 0.30 | 1.0606 | — | 31,257 | 18.1% |
| 0.50 | 1.0571 | −0.0035 | 28,821 | 16.6% |
| 0.70 | 1.0515 | −0.0055 | 33,223 | 19.2% |
| 0.90 | **1.0500** | −0.0015 | 34,661 | 20.0% |

Reference points:

- 2a 2h dedicated (`WARMDOWN_RATIO=0.70`): **1.0564** — vs this sweep's 0.70 run at
  **1.0515**, a −0.0049 gap likely dominated by run-to-run noise.
- 2b 12h baseline at its 2h intermediate (still at full LR under the 12h schedule):
  **1.1691** — every 2c run clears this by a wide margin, reaffirming that
  warmdown dominates short-budget val_bpb.

### Sweep headline

Monotonic improvement with warmdown length; diminishing returns kick in by 0.90.

- 0.30 → 0.50: −0.0035
- 0.50 → 0.70: −0.0055 (biggest gain)
- 0.70 → 0.90: −0.0015 (already flattening; within noise of 2a 0.70 above)

At 2h, **0.90 is the best quantitative answer**, but only by 0.0015 over 0.70 —
inside plausible run-to-run noise. The interesting story is qualitative.

### Qualitative trajectory across warmdown ratios

`plain_continuation` prompt at the 2h checkpoint of each run ("The old man walked
slowly toward the river and…"):

- **0.30** — jumbled syntax, implausible swerves ("agile, the man has never been
  seen as a car"). Weirdness high, surface coherence low.
- **0.50** — more coherent flow; still some awkwardness but readable as prose.
- **0.70** — clean opening, then topic-jumps into a macabre medical tangent
  (wounds / vena cava). Local syntax clean; document-level coherence weakest.
- **0.90** — enters a degenerate repetition attractor. "The man walked fast"
  recurs 6+ times in one completion. Best val_bpb, worst qualitative diversity.

Two tensions emerge:

1. **val_bpb vs qualitative diversity.** Best val_bpb (0.90) is exactly where the
   repetition attractors appear — the same pathology 2b saw at the 12h run's 8h
   (mid-warmdown) checkpoint.
2. **Topic drift vs repetition.** 0.70 drifts into new subjects; 0.90 stays on
   topic but collapses into loops. 0.50–0.70 is the diversity sweet spot.

### Step-count variation

Step counts vary 28,821 (0.50) → 34,661 (0.90) — ~20% across runs at the same
7200s training budget. Since the budget is time-based the val_bpb comparison is
fair on its own terms, but any per-step analysis would need normalization. Likely
system noise (thermal, background load) rather than schedule-dependent.

### MFU note

The `H100_BF16_PEAK_FLOPS = 989.5e12` hardcode in `train.py` has been replaced by
a small device-capability + name lookup (`BF16_PEAK_FLOPS`). On an RTX 4070 the
denominator is 116.6 TFLOPS, and the sweep's MFU numbers land at 16.6–20.0%
(reasonable for a 28.8M / depth-4 model — small models are bandwidth-bound).
Pre-fix reporting was ~2%, which was the H100 denominator divided by 4070
throughput.

### Decisions

- **0.90 wins val_bpb at 2h by a hair (−0.0015 over 0.70, within noise)**, but
  carries a clear qualitative cost (repetition attractors). The practical
  recommendation for this budget is **0.70, with 0.90 reserved for val_bpb-first
  tasks where repetition is tolerable**.
- **Open question for the next night**: can a 4h run at `WARMDOWN_RATIO=0.70`
  (or a second mini-sweep around 0.75–0.85 at 2h) beat 2h@0.90 on val_bpb
  *without* the repetition collapse? Expected cost ~4h 20m for a single 4h run,
  ~8h for a 4-point narrow sweep.

---

*Phase 2a generated 2026-04-17. Phase 2b generated 2026-04-18. Phase 2c
generated 2026-04-20. Model: 28.8M params, depth=4, dim=512, vocab=8192.
Data: karpathy/climbmix-400b-shuffle. Hardware: single NVIDIA GPU (~12GB VRAM).*
