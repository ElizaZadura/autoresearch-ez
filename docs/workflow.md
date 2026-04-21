# Autoresearch workflow reference

Short reference for how experiments, outputs, and analysis are organized.
Complements `docs/phase2-developmental-study.md` (rolling narrative) and
`docs/curios.md` (memorable generations).

---

## Output directory layout

Every training run writes to its own dir under `output/`. All run dirs share
the same internal layout, regardless of whether they're a single run or a
sweep child — the analysis tooling treats them identically.

```text
output/
├─ <YYYY-MM-DD>_<label>/                 # single run
│  ├─ progress.csv                       # dense periodic val_bpb curve
│  ├─ console.log                        # training stdout/stderr
│  ├─ model_<milestone>.pt               # milestone checkpoints (5m, 15m, ...)
│  ├─ model_<milestone>.json             # checkpoint metadata (WARMDOWN_RATIO etc.)
│  ├─ <milestone>_prompts.txt            # raw prompt-pack completions
│  ├─ prompts.md                         # generated: rendered prompt outputs
│  └─ qual-notes.md                      # generated stub + user notes
│
└─ <YYYY-MM-DD>_<sweep-label>/           # sweep parent (cross-run artifacts only)
   ├─ summary.txt                        # wall times, exit codes per child
   ├─ prompts.md                         # generated: cross-run prompt overlay
   └─ *.png                              # sweep figures exported from notebook
```

Naming conventions:

- Single-run label: `phase2_baseline`, `12h_run`, `8h_eval`, …
- Sweep parent: `<date>_<param>_sweep_<budget>` — e.g. `2026-04-19_warmdown_sweep_2h`.
- Sweep child:  `<date>_<budget>_<param>_<value>_run` — e.g. `2026-04-19_2h_warmdown_0.30_run`.
- Sweep children auto-discover via glob: `output/*_<param>_*_run`.

### Version-control convention

Under `output/`, analysis artifacts are committed; bulky/regenerable files are
not. `.gitignore` encodes this via two global rules (`*.pt`, `*.log`) plus an
explicit ignore of `output/_archive/`; everything else under `output/` is
tracked.

| Committed | Not committed |
| --- | --- |
| `progress.csv`, `model_<milestone>.json` | `model_<milestone>.pt` (via `*.pt`) |
| `<milestone>_prompts.txt`, `prompts.md` | `console.log` (via `*.log`) |
| `qual-notes.md`, sweep `summary.txt`, sweep `*.png` | `output/_archive/` |

This keeps each run fully reproducible in the notebook from a fresh clone
(curves, prompts, scoring all drive off the committed files); only re-sampling
from the trained model requires the checkpoints.

---

## `docs/` contents

| File | Purpose |
| --- | --- |
| `phase2-developmental-study.md` | Rolling cross-run narrative (2a baseline, 2b 12h, 2c sweep, …). Append new phases. |
| `curios.md` | Memorable generations, one entry per notable output. |
| `workflow.md` | This file — layout, tooling, and naming reference. |

---

## `development.ipynb` sections

Two orthogonal analysis modes in one notebook:

### Per-run (driven by `RUN_DIR` + optional `BASELINE_DIR`)

1. Checkpoint Inventory
2. Developmental Curve — `val_bpb` & train loss vs time
3. Marginal Returns
4. LR Schedule Shape
5. Prompt Pack Outputs — **also writes `<RUN_DIR>/prompts.md`**
6. Text Quality Metrics
7. Qualitative Notes — renders `<RUN_DIR>/qual-notes.md`

### Cross-run (driven by `SWEEP_DIRS`)

**Section 8 — Phase 2c — Warmdown Sweep Overlay**

- 8.1 Sweep config + reusable loader
- 8.2 Overlaid `val_bpb` curves (baseline overlay toggleable, default **on**)
- 8.3 Overlaid LR schedules
- 8.4 Final `val_bpb` vs ratio
- 8.5 Milestone grouped bars
- 8.6 Per-run small multiples
- 8.7 Cross-run prompt comparison — **also writes `<SWEEP_PARENT>/prompts.md`**

Defaults:

- `SWEEP_DIRS = sorted(OUT.glob("*_warmdown_*_run"))` — auto-detected.
- Section 8 PNG exports go into the sweep parent dir, alongside `summary.txt`.
- Section 8 skips gracefully with a short note when `SWEEP_DIRS` is empty.

The loader in 8.1 is **parameter-agnostic** — takes any list of run dirs and
returns a long-format DataFrame `(run_label, ratio, milestone, progress,
elapsed_s, val_bpb, lr_mult, train_loss)`. Works for a future narrow warmdown
sweep (e.g. 0.75–0.85) or a sweep over a different parameter, as long as
`train.py` writes that parameter into `model_*.json`.

Colouring convention: ratios on a warm→cool gradient (0.30 coolest, 0.90
warmest) so "more warmdown = warmer colour" is legible without a legend.

---

## `qual-notes.md` convention (option D — hybrid)

One `qual-notes.md` per run dir, generated on first analysis pass:

```text
<run_dir>/qual-notes.md
├─ Scoring table          ← auto-populated by the agent from prompt files
│                           (axes: coherent span, repetition, syntax,
│                           specificity, interestingness, prompt adherence,
│                           weirdness)
├─ User notes             ← empty; fill when inspired
└─ Curios hints           ← empty; promote notable entries to docs/curios.md
```

Cross-run insights stay in `docs/phase2-developmental-study.md`, not here.

---

## Prompt output rendering

Three views of the same data, each suited to a different task:

| Task | Where |
| --- | --- |
| One run's outputs, outside Jupyter | `<run_dir>/prompts.md` (IDE markdown preview) |
| Cross-run comparison, outside Jupyter | `<sweep_parent>/prompts.md` |
| Interactive exploration | Section 5 (single run) or 8.7 (sweep) |

- `<run_dir>/prompts.md` — grouped by prompt name, all milestones stacked under each. Matches Section 5's layout.
- `<sweep_parent>/prompts.md` — grouped by prompt name × milestone, all ratios stacked under each. Matches Section 8.7's layout.
- Raw `<milestone>_prompts.txt` files remain the source of truth; the `.md` files are fully regenerated each time the relevant notebook section runs. No truncation in the `.md` files; Section 5/8.7 keep the 600-char in-cell truncation for readability.

To refresh after a new run: rerun Section 5 (single run) or 8.7 (sweep).

---

## Running a sweep

```bash
# 1. Set per-run budget in prepare.py (TIME_BUDGET, seconds)
# 2. Kick off:
./run_warmdown_sweep.sh
```

The driver:

- Creates `output/<date>_warmdown_sweep_<budget>/` (parent).
- Runs `train.py` four times with `AUTORESEARCH_WARMDOWN_RATIO ∈ {0.30, 0.50, 0.70, 0.90}`.
- Each child gets a labelled output dir via `AUTORESEARCH_RUN_LABEL`.
- Writes per-child `console.log` + central `summary.txt`.
- `set -u` (unset-var fatal) but **not** `set -e` — a single failed child doesn't abort the rest of the sweep.

To sweep a different parameter or value set: copy `run_warmdown_sweep.sh`, edit
the `RATIOS` array + env-var name. New `train.py` env overrides follow the
`_WARMDOWN_RATIO_OVERRIDE` pattern (search `train.py` for that symbol).

---

## `train.py` env overrides

| Env var | Effect |
| --- | --- |
| `AUTORESEARCH_WARMDOWN_RATIO` | Override `WARMDOWN_RATIO` (float). |
| `AUTORESEARCH_RUN_LABEL` | Override auto-generated run-dir name. |
| `AUTORESEARCH_USE_FA3` | `0` disables Flash Attention 3 (forces PyTorch SDPA). |

Add new overrides by appending to the pattern near the top of `train.py`; update
`run_*_sweep.sh` to set them accordingly.

---

## MFU reporting

`train.py` computes MFU against the first CUDA device's **BF16 tensor-core
dense peak** (`BF16_PEAK_FLOPS`). Known devices: H100, A100, RTX 40-series, RTX
30-series, A6000. Unknown devices fall back to the H100 figure so MFU reads
visibly low — a clear signal to add a new entry rather than silently
misreport. The detected peak is printed early in run logs:

```text
peak BF16 TFLOPS (MFU denominator): 116.6 (device: NVIDIA GeForce RTX 4070)
```

Historical note: MFU was reported against a hardcoded H100 peak until
2026-04-20; pre-that-date numbers on non-H100 hardware need a post-hoc scaling
factor (`989.5 / <device-peak>`) to compare to post-fix numbers.

---

*Last updated 2026-04-20.*
