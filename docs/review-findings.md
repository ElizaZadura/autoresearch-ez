# Review findings: platform-training-changes

Review of the changes described in `docs/platform-training-changes.md` against
the actual implementation in `train.py`, `prepare.py`, and `pyproject.toml`.
Cross-referenced against upstream `karpathy/autoresearch` at the time of review.

---

## Issue 1 — Critical: default settings crash on non-Windows

**Location:** `train.py` lines 625, 639, 716

The fork reduced `TOTAL_BATCH_SIZE` from the upstream `2**19` (524 288 tokens)
to `2**15` (32 768 tokens), but left `DEVICE_BATCH_SIZE = 128` unchanged.

```python
TOTAL_BATCH_SIZE  = 2**15   # 32 768  (was 2**19 upstream)
DEVICE_BATCH_SIZE = 128

tokens_per_fwdbwd = 128 * 2048  # = 262 144
assert 32768 % 262144 == 0       # → AssertionError, script exits
```

On **Windows** the VRAM-cap logic (see Issue 2) incidentally rescues this by
reducing `DEVICE_BATCH_SIZE` to 16 before the assertion is reached.  On
**non-Windows** — where no cap is applied — the script crashes immediately on
every run with the default values.

**Fix options:**
- Restore `TOTAL_BATCH_SIZE = 2**19` (matches upstream; works with batch 128).
- Or lower the default to `DEVICE_BATCH_SIZE = 16` (matches the effective
  Windows value; `grad_accum_steps` becomes 1).
- Or document clearly that `DEVICE_BATCH_SIZE` must be ≤ 16 with the current
  `TOTAL_BATCH_SIZE`, so non-Windows users know to adjust before running.

---

## Issue 2 — VRAM batch-cap tiers are dead code under current settings

**Location:** `train.py` `_vram_device_batch_cap()` lines 153–162,
`_best_device_batch()` lines 144–150

The VRAM tiers return caps of 16 / 32 / 64 / 256 depending on GPU memory.
`_best_device_batch()` then finds the largest batch ≤ cap such that
`TOTAL_BATCH_SIZE % (batch × MAX_SEQ_LEN) == 0`.

With `TOTAL_BATCH_SIZE = 32768` and `MAX_SEQ_LEN = 2048`:

```
32768 / 2048 = 16   ← only integers ≤ 16 can satisfy the divisibility check
```

So caps of 32, 64, and 256 all reduce to an effective batch of 16.  The tiers
above 16 are never exercised at the current `TOTAL_BATCH_SIZE`.

The tiers would become meaningful again if `TOTAL_BATCH_SIZE` is restored to
`2**19` (where batch 128 = cap 256 gives `grad_accum_steps = 2`, batch 64
gives 4, etc.).

---

## Issue 3 — `AUTORESEARCH_USE_FA3` absent from env-var quick-reference table

**Location:** `docs/platform-training-changes.md` — environment variable table
at the bottom of the document

The FA3 section describes `AUTORESEARCH_USE_FA3=0` to force the SDPA path, but
this variable is missing from the quick-reference table.  Add a row:

| `AUTORESEARCH_USE_FA3` | `0` / `1` | Force SDPA fallback (`0`) or require FA3 (`1`) |

---

## Minor: redundant inductor env-var setting

**Location:** `train.py` lines 17–19 and `_inductor_use_aten_gemm_only()` lines 119–141

`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` / `_CONV_BACKENDS` are set via
`os.environ.setdefault()` before torch is imported (lines 17–19), which is
sufficient for torch to pick them up automatically.  The same config is set
again inside `_inductor_use_aten_gemm_only()` via `inductor_config.*`.

This is harmless — the function still does useful exclusive work (the logging
filter and the non-Windows opt-in via `AUTORESEARCH_INDUCTOR_ATEN_ONLY`).  The
duplication is just worth noting if the function is ever refactored.

---

## What looks correct

- `kernels` dependency gated on `sys_platform != 'win32'` in `pyproject.toml` ✓
- `sdpa_flash_attn_func`: bool sliding-window mask (cached), correct GQA
  passthrough, causal-only guard ✓
- FA3 fallback catches `FileNotFoundError`, `OSError`, `ImportError` ✓
- `_inductor_use_aten_gemm_only()` logging filter for "Not enough SMs" ✓
- `_use_torch_compile()` / `_use_gradient_checkpointing()` env-var overrides ✓
- `Block.forward()` checkpointing with `use_reentrant=False`, training-mode
  guard ✓
- `parse_known_args()` for `--smoke-test` ✓
- Smoke eval cap: `max(MAX_SEQ_LEN * _eval_bs * 2, 8192)` gives ≥ 1 eval step ✓
- `evaluate_bpb` `eval_tokens` cap is backward-compatible (`None` → full eval) ✓
- CUDA allocator env vars set before torch import via `setdefault` ✓
