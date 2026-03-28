# Platform and consumer-GPU training changes

This document summarises changes made **in this repository** so autoresearch can run on **Windows** without relying on an external fork. The README is unchanged; use this file as the reference for flags and behaviour.

## Goals

- Avoid pulling **Triton** / **HF `kernels`** on Windows where they are unavailable or a poor fit.
- Fall back to **PyTorch SDPA** for attention when FA3 kernels are missing.
- Keep repo drift low: preserve upstream-like defaults on non-Windows, and keep the compatibility / low-VRAM policy changes targeted at Windows unless explicitly opted into.
- Provide a **`--smoke-test`** path to validate install and a short training/eval run quickly.

## Files touched

| Area | File |
|------|------|
| Training logic, attention, compile/batch/checkpoint policy, smoke | `train.py` |
| Eval harness (optional token cap, backward compatible) | `prepare.py` |
| Dependencies (no `kernels` on Windows) | `pyproject.toml` |
| Lockfile aligned with platform markers | `uv.lock` (regenerate with `uv lock` if deps change) |

---

## Dependencies (`pyproject.toml`)

- **`kernels>=0.11.7`** is declared only when **`sys_platform != 'win32'`**, so `uv sync` on Windows does not try to install Triton via `kernels`.
- On Windows, training uses the SDPA path described below.

---

## Attention: FA3 vs SDPA (`train.py`)

- **Import `kernels` and FA3** when the package exists and loads; pick Hopper vs community repo by device capability (same idea as upstream).
- On **`FileNotFoundError`**, **`OSError`**, or **`ImportError`**, set **`flash_attn_func = sdpa_flash_attn_func`** and print a short notice.
- **`sdpa_flash_attn_func`** implements causal (and sliding-window) attention compatible with the model’s tensor layout; sliding windows use a **cached boolean `attn_mask`** (not a float `-inf` mask) for SDPA.

---

## CUDA allocator environment

Early in `train.py`, before heavy CUDA use:

- **`PYTORCH_CUDA_ALLOC_CONF`** and **`PYTORCH_ALLOC_CONF`** default to **`expandable_segments:True`** if unset (PyTorch memory-management guidance).

---

## TorchInductor / GEMM autotune (`train.py`)

- On **Windows**, **`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`** and **`TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS`** default to **`ATEN`** to avoid Triton-heavy autotune where it is problematic.
- **`_inductor_use_aten_gemm_only()`** now defaults to that behavior on **Windows only**, which keeps non-Windows runs closer to upstream.
- Opt in elsewhere with:
  - **`AUTORESEARCH_INDUCTOR_ATEN_ONLY=1`** — force ATen/cuBLAS GEMM/conv autotune backends
  - **`AUTORESEARCH_INDUCTOR_ATEN_ONLY=0`** — explicitly leave the default backends alone
- The helper still adds a logging filter to drop the common “not enough SMs to use max_autotune_gemm” line from `torch._inductor.utils` when the ATen-only path is active.

---

## Per-device batch size and `TOTAL_BATCH_SIZE`

- **Windows default:** **`_vram_device_batch_cap(total_memory)`** applies rough upper bounds by VRAM tier (e.g. &lt;16 GiB → 16, &lt;24 → 32, &lt;40 → 64, else up to 256).
- **Non-Windows default:** leave `DEVICE_BATCH_SIZE` as set in the script, to stay closer to upstream behavior.
- **`_best_device_batch(requested, max_batch)`** chooses the largest batch ≤ cap such that **`TOTAL_BATCH_SIZE % (batch × MAX_SEQ_LEN) == 0`**, preserving gradient-accumulation correctness when a cap is active.
- Override with:
  - **`AUTORESEARCH_DEVICE_BATCH_SIZE_CAP=<int>`** — apply an explicit cap on any platform
  - **`AUTORESEARCH_DEVICE_BATCH_SIZE_CAP=off`** — disable auto-capping entirely

---

## `torch.compile` policy

- **Default off** on **Windows**.
- **Default on** on **non-Windows**, matching upstream more closely.
- Override explicitly:
  - **`AUTORESEARCH_USE_TORCH_COMPILE=0`** — force off
  - **`AUTORESEARCH_USE_TORCH_COMPILE=1`** — force on
- When compile is off, **fused AdamW/Muon steps are not compiled**; the full model is only compiled when compile is on.

---

## Gradient checkpointing

- **`GPTConfig.use_gradient_checkpointing`**; each **`Block`** uses **`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`** in **training** mode when enabled.
- **Default:** **off**, except on **Windows with VRAM &lt; 20 GiB**, unless overridden:
  - **`AUTORESEARCH_GRADIENT_CHECKPOINTING=1`** — force on
  - **`AUTORESEARCH_GRADIENT_CHECKPOINTING=0`** — force off

---

## Smoke test: `--smoke-test`

Parsed with **`parse_known_args()`** so extra trailing arguments do not break **`uv run train.py --smoke-test`**.

When enabled:

- **`TRAINING_TIME_BUDGET`** is **12 seconds** (instead of `TIME_BUDGET` from `prepare.py`).
- Training stops after **`SMOKE_MAX_OPTIMIZER_STEPS` (4)** optimizer updates after the compile warmup window.
- The warmup window is now **conditional**: **10 steps** when `torch.compile` is enabled, **0** when it is not.
- Final **`evaluate_bpb`** uses a smaller eval batch (capped at **4**) and passes **`eval_tokens`** so eval stays cheap.

Run:

```bash
uv run train.py --smoke-test
```

---

## Evaluation harness (`prepare.py`)

- **`evaluate_bpb(model, tokenizer, batch_size, eval_tokens=None)`**
  - **`eval_tokens=None`** (default): uses full **`EVAL_TOKENS`** — same as before.
  - **`eval_tokens` set**: caps how many tokens contribute to the metric; **`steps = max(1, cap // (batch_size * MAX_SEQ_LEN))`** so at least one step always runs.

The BPB definition and dataloader contract are unchanged; only an optional cap was added for smoke runs.

---

## Environment variables (quick reference)

| Variable | Values | Effect |
|----------|--------|--------|
| `AUTORESEARCH_USE_TORCH_COMPILE` | `0` / `1` (also `true`/`false`/`on`/`off`) | Force compile off or on |
| `AUTORESEARCH_GRADIENT_CHECKPOINTING` | `0` / `1` | Force checkpointing off or on |
| `AUTORESEARCH_DEVICE_BATCH_SIZE_CAP` | integer, or `off` | Apply or disable the device-batch auto-cap |
| `AUTORESEARCH_INDUCTOR_ATEN_ONLY` | `0` / `1` | Force ATen-only GEMM/conv autotune backends |

---

## Known limitations (unchanged by this work)

- **MFU percentage** in logs still uses a **hardcoded H100 peak FLOPs** constant; on consumer GPUs the **percentage is not comparable** to an H100 run — throughput and `tok/sec` are still meaningful.

---

## Relation to external forks

These changes **reimplement the intent** of community Windows fixes **locally** (attention fallback, no `kernels` on Windows, Windows-oriented compile/batch/checkpoint defaults, smoke path). They are **not** a copy of any single fork; behaviour may differ in details.
