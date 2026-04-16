# program.md

## Current Phase: Developmental Continuation

The prior optimization phase substantially improved short-horizon (5 minute) validation performance.  
Easy gains from local hyperparameter search appear mostly exhausted.

Do not continue blind knob-turning.

The objective has changed.

We now study how the current best training recipe develops over longer training horizons.

---

## Primary Questions

1. Does the current 5-minute winner remain strong at 15m, 30m, 1h, 2h, or longer?

2. When do validation gains begin to flatten materially?

3. What changes in generated samples over time?

4. Does surface fluency improve before deeper structural coherence?

5. Does continued training reduce unusual, memorable, or high-texture outputs?

6. What signs indicate diminishing returns for this model/data regime?

---

## Core Instructions

Treat `train.py` as mostly stable.

Prefer continuation runs and observation over mutation.

Use fixed checkpoints such as:

- 5 min
- 15 min
- 30 min
- 1 h
- 2 h
- 4 h

(Adjust if hardware/runtime constraints require.)

At each checkpoint:

- record validation metrics
- save model checkpoint if practical
- generate samples using the same fixed prompt pack
- compare against earlier checkpoints

---

## Prompt Pack Rules

Use the same prompts each time.

Keep prompts short, stable, and varied:

- plain continuation
- factual fragment
- open-ended fragment
- structurally awkward prompt
- one nonsense / anomaly lure
- one recurring signature prompt

Do not constantly redesign prompts.

Comparability matters more than novelty.

---

## Evaluation Priorities

Track both quantitative and qualitative change.

Quantitative:

- val_bpb trend
- speed / throughput
- stability

Qualitative:

- coherent span length
- repetition onset
- syntax stability
- generic filler behavior
- memorability
- anomaly retention or collapse
- interestingness

---

## Mutation Policy

Only modify `train.py` if one of the following is true:

1. Clear instability appears
2. Longer runs expose a specific bottleneck
3. A concrete hypothesis exists worth testing
4. Evidence suggests the 5-minute winner is poor at longer horizons

Do not mutate for the sake of activity.

---

## Anti-Traps

Do not:

- chase tiny 5-minute gains
- overfit to one metric
- change many variables at once
- mistake smoother outputs for better learning
- erase weird but informative behavior unnoticed

---

## Success Criteria

Success is not a tiny benchmark improvement.

Success is:

- a clear developmental map
- known diminishing-return points
- understanding of what emerges first
- understanding of what stabilizes later
- understanding of what gets lost during training
- a better next question
