# Qualitative Notes
Detailed narrative analysis (plateau evidence, warmdown gap, decisions for next night) lives in the rolling log: [CODE0 — Phase 2b section](../../docs/phase2-developmental-study.md#phase-2b-12h-extended-run-2026-04-18).

Memorable generations from this run are collected in [CODE0](../../docs/curios.md).

## Scoring

Scales are defined in [CODE0](../../program.md).

Columns 5m–4h carry over from the 2026-04-17 phase2 baseline for reference. Note that the 12h run's own 5m–4h checkpoints share the 12h LR schedule (still at full LR at 4h), so their prompt-output quality is not directly comparable to the dedicated baseline — re-score this run's own 5m–4h columns when a same-schedule comparison is needed. Columns 8h and 12h are specific to this run; fill in after reviewing the `<label>_prompts.txt` files.

**2026-04-17_phase2_baseline**
| Prompt | Metric | 5m | 15m | 30m | 1h | 2h | 4h | 8h | 12h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain_continuation | coherent span (1–5) | 1 | 1 | 1 | 2 | 1 | 2 | <br> | <br> |
| plain_continuation | syntax stability | poor | mixed | mixed | poor | mixed | good | <br> | <br> |
| plain_continuation | prompt adherence | poor | poor | poor | poor | poor | poor | <br> | <br> |
| factual_fragment | specificity vs sludge | filler-heavy | filler-heavy | filler-heavy | filler-heavy | filler-heavy | filler-heavy | <br> | <br> |
| factual_fragment | prompt adherence | weak | weak | weak | weak | weak | weak | <br> | <br> |
| longitudinal_anchor | coherent span (1–5) | 1 | 2 | 1 | 1 | 2 | 2 | <br> | <br> |
| longitudinal_anchor | specificity vs sludge | filler-heavy | filler-heavy | filler-heavy | filler-heavy | filler-heavy | mixed | <br> | <br> |
| structurally_awkward | syntax stability | good | good | mixed | mixed | mixed | mixed | <br> | <br> |
| structurally_awkward | prompt adherence | weak | weak | weak | weak | weak | weak | <br> | <br> |
| anomaly_lure | weirdness retained | high | high | high | high | high | high | <br> | <br> |
| anomaly_lure | interestingness | none | mild | mild | notable | notable | mild | <br> | <br> |
| signature | coherent span (1–5) | 2 | 2 | 3 | 2 | 1 | 2 | <br> | <br> |
| signature | repetition onset | none | early | none | none | none | none | <br> | <br> |
| continuation | coherent span (1–5) | 1 | 1 | 1 | 1 | 1 | 1 | <br> | <br> |
| continuation | prompt adherence | 1 | 1 | 1 | 1 | 1 | 1 | <br> | <br> |
| **ALL** | **overall interestingness** | mild | mild | mild | notable | notable | notable | <br> | <br> |

**2026-04-17 12h Extended Run**
| Prompt | Metric | 15m | 30m | 1h | 2h | 4h | 8h | 12h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain_continuation | coherent span (1–5) | 1 | 1 | 1 | 2 | 2 | 1 | 2 |
| plain_continuation | syntax stability | mixed | mixed | mixed | mixed | mixed | poor | mixed |
| plain_continuation | prompt adherence | weak | weak | weak | partial | weak | weak | partial |
| factual_fragment | specificity vs sludge | filler-heavy | filler-heavy | filler-heavy | filler-heavy | filler-heavy | filler-heavy | mixed |
| factual_fragment | prompt adherence | weak | weak | weak | weak | weak | weak | partial |
| longitudinal_anchor | coherent span (1–5) | 2 | 1 | 1 | 1 | 3 | 3 | 2 |
| longitudinal_anchor | specificity vs sludge | filler-heavy | filler-heavy | filler-heavy | filler-heavy | mixed | mixed | filler-heavy |
| structurally_awkward | syntax stability | mixed | mixed | mixed | mixed | good | mixed | mixed |
| structurally_awkward | prompt adherence | weak | weak | weak | weak | partial | partial | weak |
| anomaly_lure | weirdness retained | high | high | balanced | high | high | high | balanced |
| anomaly_lure | interestingness | mild | notable | mild | notable | notable | notable | notable |
| signature | coherent span (1–5) | 1 | 1 | 1 | 2 | 2 | 2 | 2 |
| signature | repetition onset | none | none | none | none | medium | none | none |
| continuation | coherent span (1–5) | 1 | 1 | 1 | 1 | 1 | 2 | 2 |
| continuation | prompt adherence | weak | weak | weak | weak | weak | weak | partial |
| **ALL** | **overall interestingness** | mild | notable | mild | mild | notable | notable | notable |

Rename suggestion: for structurally_awkward, prompt adherence is not a good fit. What would it even be? Remain awkward? Produce nested contradiction? Better metrics would be recovery quality, structural handling, syntactic resilience? I'm also still unsure about the specificity vs sludge metric and how to interpret or apply it.
