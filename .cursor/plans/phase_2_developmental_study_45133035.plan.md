---
name: Phase 2 Developmental Study
overview: "Run the Phase 2 developmental study: a smoke test to verify the pipeline, then 6 fresh training runs at increasing time budgets (5m through 4h), followed by quantitative evaluation and qualitative prompt-pack analysis."
todos:
  - id: smoke-test
    content: Run smoke test (uv run train.py --smoke-test) to verify pipeline works end-to-end
    status: completed
  - id: run-5m
    content: Set TIME_BUDGET=300 in prepare.py, run uv run train.py, verify model_5m.pt produced
    status: completed
  - id: run-15m
    content: Set TIME_BUDGET=900, run training, verify model_15m.pt
    status: completed
  - id: run-30m
    content: Set TIME_BUDGET=1800, run training, verify model_30m.pt
    status: completed
  - id: run-1h
    content: Set TIME_BUDGET=3600, run training, verify model_1h.pt
    status: completed
  - id: run-2h
    content: Set TIME_BUDGET=7200, run training, verify model_2h.pt
    status: completed
  - id: run-4h
    content: Set TIME_BUDGET=14400, run training, verify model_4h.pt
    status: completed
  - id: eval-checkpoints
    content: Run eval_checkpoints.py to generate val_bpb results CSV
    status: completed
  - id: eval-prompts
    content: Run eval_prompts.py against all 6 checkpoints for qualitative analysis
    status: completed
  - id: report
    content: "Summarize developmental trajectory: val_bpb curve, qualitative observations, diminishing returns"
    status: completed
isProject: false
---

# Phase 2 — Developmental Study Execution

## Test Run (together with user)

Run a smoke test to verify the full pipeline works end-to-end before committing GPU-hours:

```bash
uv run train.py --smoke-test
```

This runs for ~12 seconds + compile warmup, with max 4 optimizer steps. Confirms data loading, model init, training loop, and checkpoint saving all work. No changes to `prepare.py` needed.

## Production Runs (autonomous, after user confirms)

Six fresh runs, shortest to longest, each editing `TIME_BUDGET` in [prepare.py](prepare.py) before launching:

| Run | `TIME_BUDGET` | Produces | Approx wall-clock |
|-----|--------------|----------|-------------------|
| 1 | `300` | `model_5m.pt` | ~5 min + startup |
| 2 | `900` | `model_15m.pt` | ~15 min + startup |
| 3 | `1800` | `model_30m.pt` | ~30 min + startup |
| 4 | `3600` | `model_1h.pt` | ~1 hr + startup |
| 5 | `7200` | `model_2h.pt` | ~2 hr + startup |
| 6 | `14400` | `model_4h.pt` | ~4 hr + startup |

Each run:
1. Edit `TIME_BUDGET` in `prepare.py` to the target value
2. Run `uv run train.py`
3. Verify the expected checkpoint file and JSON sidecar were saved
4. Note the reported `val_bpb`

**Important:** Each run is fresh (random init), not resumed. The LR schedule (with `WARMDOWN_RATIO=0.70`) scales proportionally to TIME_BUDGET, which is exactly what the developmental study tests.

## Evaluation (after all runs complete)

1. **Quantitative** -- evaluate all checkpoints:

```bash
uv run eval_checkpoints.py
```

Produces `output/eval_results.csv` with `val_bpb` for each timed checkpoint.

2. **Qualitative** -- run the fixed 7-prompt pack against all checkpoints:

```bash
uv run eval_prompts.py model_5m.pt model_15m.pt model_30m.pt model_1h.pt model_2h.pt model_4h.pt
```

Produces `output/<label>_prompts.txt` for each checkpoint.

## Key files involved

- [prepare.py](prepare.py) -- edit `TIME_BUDGET` constant between runs (line with `TIME_BUDGET = 300`)
- [train.py](train.py) -- training script, run as-is (no modifications per program.md mutation policy)
- [eval_checkpoints.py](eval_checkpoints.py) -- batch val_bpb evaluation
- [eval_prompts.py](eval_prompts.py) -- prompt pack generation
- [program.md](program.md) -- governing research protocol

## Total estimated time

Roughly **7.5 hours** of GPU training time (5+15+30+60+120+240 minutes) plus startup/compile overhead per run. The smoke test adds negligible time.