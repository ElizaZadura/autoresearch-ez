"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py [--smoke-test]
"""

import argparse
import os
import sys

# CUDA caching allocator (see https://pytorch.org/docs/stable/notes/cuda.html#memory-management )
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# Windows CUDA wheels ship without Triton; Inductor Triton GEMM autotune can fail or
# spam "not enough SMs" on consumer GPUs (<68 SMs is PyTorch's "big GPU" cutoff).
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS", "ATEN")

_argp = argparse.ArgumentParser(description="Autoresearch single-GPU pretraining")
_argp.add_argument(
    "--smoke-test",
    action="store_true",
    help="Short run (few steps + tiny eval) to validate setup.",
)
_TRAIN_ARGS, _ = _argp.parse_known_args()
SMOKE_TEST = _TRAIN_ARGS.smoke_test

import gc
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    make_dataloader,
    evaluate_bpb,
    EVAL_TOKENS,
)

# Fixed prompt pack — mirrors eval_prompts.py; do not edit between runs.
_PROMPT_PACK = [
    ("plain_continuation", "The old man walked slowly toward the river and"),
    ("factual_fragment", "The capital of France is Paris, and the population of"),
    ("longitudinal_anchor", "In one sentence, the meaning of life is"),
    ("structurally_awkward", "Despite the fact that however, the reason why because"),
    ("anomaly_lure", "Ground control to Major Snorf,"),
    ("signature", "Once upon a time there was a small"),
    ("continuation", "The instructions were clear until line seven:"),
]
_PROMPT_MAX_NEW_TOKENS = 200
_PROMPT_TEMP = 1.0
_PROMPT_TOP_K = 50
_PROMPT_TOP_P = 1.0

TRAINING_TIME_BUDGET = 12 if SMOKE_TEST else TIME_BUDGET
SMOKE_MAX_OPTIMIZER_STEPS = 4  # counted after step 10 (post compile-warmup)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _budget_label(seconds: int) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


_RUN_LABEL_OVERRIDE = os.environ.get("AUTORESEARCH_RUN_LABEL", "").strip()
if _RUN_LABEL_OVERRIDE:
    _run_label = _RUN_LABEL_OVERRIDE
elif SMOKE_TEST:
    _run_label = datetime.now().strftime("%Y-%m-%d_smoke_%H%M%S")
else:
    _run_label = f"{datetime.now().strftime('%Y-%m-%d')}_{_budget_label(TRAINING_TIME_BUDGET)}_run"
RUN_DIR = os.path.join(_HERE, "output", _run_label)
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Flash Attention 3 via HF kernels when available; PyTorch SDPA otherwise
# (e.g. Windows builds often lack matching kernel variants).
# ---------------------------------------------------------------------------

_sdpa_sliding_mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}


def _env_flag(name: str):
    value = os.environ.get(name, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return None


def sdpa_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
    """Drop-in-ish replacement for FA3 flash_attn_func: B,T,H,D tensors, GQA ok."""
    if not causal:
        raise ValueError("sdpa fallback only supports causal attention")
    B, T, hq, D = q.shape
    hkv = k.shape[2]
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    w_left = int(window_size[0])
    gqa_kw = {"enable_gqa": True} if hq != hkv else {}

    if w_left < 0 or w_left >= T:
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, **gqa_kw)
        return out.transpose(1, 2).contiguous()

    dev_s = str(q.device)
    key = (T, w_left, dev_s)
    mask = _sdpa_sliding_mask_cache.get(key)
    if mask is None:
        ii = torch.arange(T, device=q.device).unsqueeze(1)
        jj = torch.arange(T, device=q.device).unsqueeze(0)
        causal_ok = jj <= ii
        band_ok = jj >= (ii - (w_left - 1))
        allowed = causal_ok & band_ok
        # Bool mask: less memory / friendlier to SDPA than float -inf additive mask.
        mask = allowed.view(1, 1, T, T)
        _sdpa_sliding_mask_cache[key] = mask
    out = F.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=mask, is_causal=False, **gqa_kw
    )
    return out.transpose(1, 2).contiguous()


_use_fa3 = _env_flag("AUTORESEARCH_USE_FA3")
if _use_fa3 is False:
    print("kernels: FA3 disabled by AUTORESEARCH_USE_FA3=0; using PyTorch SDPA")
    flash_attn_func = sdpa_flash_attn_func
else:
    try:
        from kernels import get_kernel

        cap = torch.cuda.get_device_capability()
        # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
        _fa_repo = (
            "varunneal/flash-attention-3"
            if cap == (9, 0)
            else "kernels-community/flash-attn3"
        )
        flash_attn_func = get_kernel(_fa_repo).flash_attn_interface.flash_attn_func
    except (FileNotFoundError, OSError, ImportError):
        print(
            "kernels: FA3 not available for this platform; using PyTorch SDPA for attention"
        )
        flash_attn_func = sdpa_flash_attn_func


def _inductor_use_aten_gemm_only():
    """Use ATen/cuBLAS for compiled GEMMs when Triton autotune is unavailable or unsuitable."""
    if not torch.cuda.is_available():
        return
    aten_only = _env_flag("AUTORESEARCH_INDUCTOR_ATEN_ONLY")
    if aten_only is False:
        return
    if aten_only is None and sys.platform != "win32":
        return
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "ATEN"
    inductor_config.max_autotune_conv_backends = "ATEN"

    class _DropSmAutotuneMsg(logging.Filter):
        def filter(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return "Not enough SMs to use max_autotune_gemm" not in msg

    logging.getLogger("torch._inductor.utils").addFilter(_DropSmAutotuneMsg())


def _best_device_batch(requested: int, max_batch: int) -> int:
    """Largest batch such that TOTAL_BATCH_SIZE % (batch * MAX_SEQ_LEN) == 0 and batch <= max_batch."""
    cap = min(requested, max_batch)
    for b in range(cap, 0, -1):
        if TOTAL_BATCH_SIZE % (b * MAX_SEQ_LEN) == 0:
            return b
    return 1


def _vram_device_batch_cap(total_memory_bytes: int) -> int:
    """Rough caps aligned with consumer-GPU practice (cf. jsegov/autoresearch-win-rtx).

    Note: at the current TOTAL_BATCH_SIZE=2**15 and MAX_SEQ_LEN=2048, the
    effective batch chosen by _best_device_batch is capped at 16 regardless of
    the tier (only b in {1,2,4,8,16} divides TOTAL_BATCH_SIZE/MAX_SEQ_LEN=16).
    The higher tiers re-activate when TOTAL_BATCH_SIZE is raised."""
    gib = total_memory_bytes / (1024**3)
    if gib < 16:
        return 16
    if gib < 24:
        return 32
    if gib < 40:
        return 64
    return 256


def _device_batch_cap(total_memory_bytes: int):
    """Optional per-device batch cap. Default heuristic is Windows-only to minimize repo drift."""
    override = os.environ.get("AUTORESEARCH_DEVICE_BATCH_SIZE_CAP", "").strip().lower()
    if override:
        if override in ("0", "false", "no", "off", "none"):
            return None
        try:
            cap = int(override)
        except ValueError as exc:
            raise ValueError(
                "AUTORESEARCH_DEVICE_BATCH_SIZE_CAP must be an integer or one of: off, false, no, none"
            ) from exc
        if cap < 1:
            raise ValueError("AUTORESEARCH_DEVICE_BATCH_SIZE_CAP must be >= 1")
        return cap
    if sys.platform == "win32":
        return _vram_device_batch_cap(total_memory_bytes)
    return None


def _use_torch_compile() -> bool:
    """Match upstream by default off only on Windows; keep env override for opt-in tuning."""
    o = _env_flag("AUTORESEARCH_USE_TORCH_COMPILE")
    if o is False:
        return False
    if o is True:
        return True
    return sys.platform != "win32"


def _use_gradient_checkpointing(total_memory_bytes: int) -> bool:
    """Default off, except on Windows consumer GPUs with <10 GiB VRAM where the
    small-model config would otherwise risk OOM. The 12h extended-run MFU
    analysis (~0.14% milestone overhead, ~5 GiB peak VRAM out of 12 GiB) showed
    checkpointing was significantly slowing 12 GiB cards while buying no
    headroom. Set AUTORESEARCH_GRADIENT_CHECKPOINTING=1 to force on if you
    later train a larger model that would otherwise OOM."""
    e = _env_flag("AUTORESEARCH_GRADIENT_CHECKPOINTING")
    if e is True:
        return True
    if e is False:
        return False
    return sys.platform == "win32" and total_memory_bytes < 10 * (1024**3)


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    use_gradient_checkpointing: bool = False


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = ((8 * config.n_embd // 3) // 128) * 128  # SwiGLU 8/3x, aligned to head_dim
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.silu(self.c_fc(x)) * self.c_gate(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def forward(self, x, ve, cos_sin, window_size):
        if self.training and self.use_gradient_checkpointing:

            def _block_inner(h):
                h = h + self.attn(norm(h), ve, cos_sin, window_size)
                h = h + self.mlp(norm(h))
                return h

            return torch.utils.checkpoint.checkpoint(
                _block_inner, x, use_reentrant=False
            )
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=4, beta2=MUON_BETA2, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 128       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "LLLL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**15 # ~32K tokens per optimizer step
EMBEDDING_LR = 1.3      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.08     # slightly stronger constant Muon regularization
MUON_BETA2 = 0.95       # NorMuon second-moment EMA (variance reduction; 0.90 and 0.99 both worse)
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
_WARMDOWN_RATIO_OVERRIDE = os.environ.get("AUTORESEARCH_WARMDOWN_RATIO", "").strip()
WARMDOWN_RATIO = (
    float(_WARMDOWN_RATIO_OVERRIDE) if _WARMDOWN_RATIO_OVERRIDE else 0.70
)  # brief cooldown at end
FINAL_LR_FRAC = 0.05    # decay to 5% of initial LR

# Model size
DEPTH = 4               # number of transformer layers
DEVICE_BATCH_SIZE = 16   # per-device batch size; max usable is TOTAL_BATCH_SIZE/MAX_SEQ_LEN (=16 here)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
_cuda_mem = torch.cuda.get_device_properties(0).total_memory
_inductor_use_aten_gemm_only()
USE_TORCH_COMPILE = _use_torch_compile()
if USE_TORCH_COMPILE:
    adamw_step_fused = torch.compile(adamw_step_fused, dynamic=False, fullgraph=True)
    muon_step_fused = torch.compile(muon_step_fused, dynamic=False, fullgraph=True)
else:
    print(
        "torch.compile: off. "
        "Set AUTORESEARCH_USE_TORCH_COMPILE=1 to force on."
    )
_batch_cap = _device_batch_cap(_cuda_mem)
if _batch_cap is not None:
    DEVICE_BATCH_SIZE = _best_device_batch(DEVICE_BATCH_SIZE, _batch_cap)
    print(
        f"device batch: {DEVICE_BATCH_SIZE} (VRAM ~{_cuda_mem / (1024**3):.1f} GiB; cap {_batch_cap})"
    )
else:
    print(f"device batch: {DEVICE_BATCH_SIZE}")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def _bf16_peak_flops() -> float:
    """Peak dense BF16 tensor-core throughput for the first CUDA device,
    used as the MFU denominator. Values follow NVIDIA spec-sheet TFLOPS
    for non-sparse BF16/FP16 tensor ops. Unknown devices fall back to the
    H100 figure so MFU reads visibly low — a clear signal to add a new
    entry rather than silently misreport."""
    fallback = 989.5e12  # H100 SXM5 BF16 dense tensor peak
    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return fallback
    if cap >= (9, 0):  # Hopper/Blackwell (H100, H200, etc.)
        return fallback
    if cap == (8, 9):  # Ada Lovelace — RTX 40 series, L40, L4
        if "4090" in name:
            return 330.3e12
        if "4080" in name:
            return 195.0e12
        if "4070 ti" in name:
            return 160.4e12
        if "4070" in name:
            return 116.6e12
        if "4060 ti" in name:
            return 87.2e12
        if "4060" in name:
            return 60.0e12
        return fallback
    if cap == (8, 0):  # Ampere A100
        return 312e12
    if cap in ((8, 6), (8, 7)):  # Ampere consumer (RTX 30, A40, A6000)
        if "3090" in name or "a6000" in name:
            return 142e12
        if "3080" in name:
            return 119e12
        return fallback
    return fallback


BF16_PEAK_FLOPS = _bf16_peak_flops()
print(
    f"peak BF16 TFLOPS (MFU denominator): {BF16_PEAK_FLOPS / 1e12:.1f} "
    f"(device: {torch.cuda.get_device_name(0)})"
)
COMPILE_WARMUP_STEPS = 10 if USE_TORCH_COMPILE else 0

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth, *, use_gradient_checkpointing: bool):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


_use_ckpt = _use_gradient_checkpointing(_cuda_mem)
config = build_model_config(DEPTH, use_gradient_checkpointing=_use_ckpt)
print(f"Model config: {asdict(config)}")
if _use_ckpt:
    print(
        "gradient checkpointing: on (AUTORESEARCH_GRADIENT_CHECKPOINTING=0 to disable)"
    )

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

_raw_model = (
    model  # uncompiled reference; used for shape-dynamic sampling during training
)
if USE_TORCH_COMPILE:
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TRAINING_TIME_BUDGET}s" + (" (smoke test)" if SMOKE_TEST else ""))
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TRAINING_TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 100, 1)
    return (1 - frac) * 0.85 + frac * 0.93

def get_weight_decay(progress):
    return WEIGHT_DECAY

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Intermediate checkpoint thresholds (seconds). Each fires at most once per run.
# Thresholds above TRAINING_TIME_BUDGET simply never fire, so one list covers all budgets.
_CKPT_THRESHOLDS = [300, 900, 1800, 3600, 7200, 14400, 28800, 43200]
_CKPT_LABELS = ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h"]
_ckpt_fired = set()  # tracks which thresholds have already been saved

# Periodic mid-run val_bpb points (every ~5% of progress). Capped eval for cheapness;
# the full uncapped eval is run only at the milestone checkpoints above.
_PERIODIC_EVAL_INTERVAL = 0.05  # fraction of progress
_periodic_eval_fraction = _PERIODIC_EVAL_INTERVAL  # next trigger
_PERIODIC_EVAL_CAP_TOKENS = max(MAX_SEQ_LEN * 32, EVAL_TOKENS // 4)
_PROGRESS_CSV = os.path.join(RUN_DIR, "progress.csv")
_MILESTONE_OVERHEAD_LOGGED = False

def _write_meta(path: str, meta: dict):
    """Write a JSON sidecar next to a checkpoint file."""
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2)


def _patch_meta_val_bpb(json_path: str, val_bpb: float):
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        meta["val_bpb"] = round(float(val_bpb), 6)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"\n[warn] could not patch {json_path}: {e}")


def _append_progress_csv(row: dict):
    new_file = not os.path.exists(_PROGRESS_CSV)
    keys = [
        "elapsed_seconds",
        "num_steps",
        "progress",
        "lr_mult",
        "train_loss_ema",
        "val_bpb_capped",
        "eval_cap_tokens",
    ]
    with open(_PROGRESS_CSV, "a", encoding="utf-8") as f:
        if new_file:
            f.write(",".join(keys) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


@torch.no_grad()
def _generate_once(
    gen_model, tokenizer_local, prompt_text: str, max_new_tokens: int, seq_len: int
) -> str:
    ids = tokenizer_local.encode(prompt_text, prepend=tokenizer_local.bos_token_id)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -seq_len:]
        logits = gen_model(idx_cond)  # targets=None -> returns logits
        logits = logits[:, -1, :]
        logits = logits / max(_PROMPT_TEMP, 1e-6)
        if _PROMPT_TOP_K > 0:
            topk_vals, _ = logits.topk(_PROMPT_TOP_K, dim=-1)
            logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer_local.decode(idx[0, len(ids) :].tolist())


def _write_prompt_pack(
    label: str, ckpt_filename: str, val_bpb, elapsed: float, num_steps: int
):
    """Generate the 7-prompt pack and write <label>_prompts.txt matching eval_prompts.py format."""
    # Use the uncompiled module to avoid torch.compile recompilation on varying seq lengths.
    gen_model = _raw_model
    was_training = gen_model.training
    gen_model.eval()
    try:
        lines = []
        lines.append(f"checkpoint: {ckpt_filename}")
        lines.append(f"label:      {label}")
        lines.append(f"val_bpb:    {val_bpb if val_bpb is not None else 'n/a'}")
        lines.append(f"elapsed:    {elapsed}s")
        lines.append(f"num_steps:  {num_steps}")
        lines.append(f"generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(
            f"temp:       {_PROMPT_TEMP}  top_k: {_PROMPT_TOP_K}  "
            f"top_p: {_PROMPT_TOP_P}  max_new_tokens: {_PROMPT_MAX_NEW_TOKENS}"
        )
        lines.append("=" * 72)
        for name, prompt in _PROMPT_PACK:
            try:
                with autocast_ctx:
                    completion = _generate_once(
                        gen_model,
                        tokenizer,
                        prompt,
                        _PROMPT_MAX_NEW_TOKENS,
                        config.sequence_len,
                    )
            except Exception as e:
                completion = f"<sample_gen_error: {e!r}>"
            lines.append(f"\n[{name}]")
            lines.append(f"PROMPT:     {prompt}")
            lines.append(f"COMPLETION: {prompt}{completion}")
        lines.append("\n" + "=" * 72)
        out_path = os.path.join(RUN_DIR, f"{label}_prompts.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"prompts [{label}]: {out_path}")
    finally:
        if was_training:
            gen_model.train()


def _run_eval(cap_tokens=None):
    """Wrap evaluate_bpb with eval/train mode toggling. Returns val_bpb float."""
    was_training = model.training
    model.eval()
    try:
        with autocast_ctx:
            return evaluate_bpb(
                model, tokenizer, DEVICE_BATCH_SIZE, eval_tokens=cap_tokens
            )
    finally:
        if was_training:
            model.train()


def _periodic_eval(
    elapsed: float, current_step: int, train_loss: float, lr_mult: float
):
    """Cheap val_bpb snapshot every ~5% of progress; appended to progress.csv."""
    t0 = time.time()
    val_bpb_capped = _run_eval(cap_tokens=_PERIODIC_EVAL_CAP_TOKENS)
    dt = time.time() - t0
    _append_progress_csv(
        {
            "elapsed_seconds": round(elapsed, 1),
            "num_steps": current_step,
            "progress": round(elapsed / max(TRAINING_TIME_BUDGET, 1), 4),
            "lr_mult": round(lr_mult, 4),
            "train_loss_ema": round(train_loss, 6),
            "val_bpb_capped": round(val_bpb_capped, 6),
            "eval_cap_tokens": _PERIODIC_EVAL_CAP_TOKENS,
        }
    )
    print(
        f"\nperiodic_eval @ {elapsed:.0f}s  val_bpb(capped)={val_bpb_capped:.4f}  ({dt:.1f}s)"
    )
    return val_bpb_capped


def _save_timed_checkpoint(elapsed: float, current_step: int, train_loss: float):
    """At each milestone: save .pt, run full val_bpb, generate prompt pack."""
    global _MILESTONE_OVERHEAD_LOGGED
    for threshold, label in zip(_CKPT_THRESHOLDS, _CKPT_LABELS):
        if threshold in _ckpt_fired or elapsed < threshold:
            continue
        # Only fire thresholds that are meaningful for this budget.
        if threshold > TRAINING_TIME_BUDGET:
            continue
        _ckpt_fired.add(threshold)
        t_start = time.time()
        path = os.path.join(RUN_DIR, f"model_{label}.pt")
        meta = {
            "label": label,
            "elapsed_seconds": round(elapsed, 1),
            "num_steps": current_step,
            "total_tokens": current_step * TOTAL_BATCH_SIZE,
            "train_loss_ema": round(train_loss, 6),
            "val_bpb": None,
            "hyperparams": {
                "WARMDOWN_RATIO": WARMDOWN_RATIO,
                "FINAL_LR_FRAC": FINAL_LR_FRAC,
                "MATRIX_LR": MATRIX_LR,
                "EMBEDDING_LR": EMBEDDING_LR,
                "MUON_BETA2": MUON_BETA2,
                "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
                "DEPTH": DEPTH,
            },
        }
        torch.save(
            {
                "model_state_dict": _raw_model.state_dict(),
                "config": asdict(config),
                "elapsed_seconds": elapsed,
                "num_steps": current_step,
                "label": label,
            },
            path,
        )
        _write_meta(path, meta)
        print(f"\ncheckpoint [{label}]: {path}")

        # Full (uncapped) eval + prompt pack. On exception, continue training.
        try:
            full_val_bpb = _run_eval(cap_tokens=None)
            _patch_meta_val_bpb(path.replace(".pt", ".json"), full_val_bpb)
            print(f"val_bpb [{label}]: {full_val_bpb:.6f} (full)")
        except Exception as e:
            full_val_bpb = None
            print(f"\n[warn] full eval at [{label}] failed: {e!r}")
        try:
            _write_prompt_pack(
                label,
                os.path.basename(path),
                full_val_bpb,
                round(elapsed, 1),
                current_step,
            )
        except Exception as e:
            print(f"\n[warn] prompt pack at [{label}] failed: {e!r}")

        overhead = time.time() - t_start
        if not _MILESTONE_OVERHEAD_LOGGED:
            print(
                f"milestone overhead [{label}]: {overhead:.1f}s "
                f"(~{100 * overhead / max(TRAINING_TIME_BUDGET, 1):.2f}% of budget)"
            )
            _MILESTONE_OVERHEAD_LOGGED = True

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TRAINING_TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    completed_steps = step + 1

    # EMA loss (updated before checkpoint so it can be recorded in the sidecar)
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

    if completed_steps > COMPILE_WARMUP_STEPS:
        total_training_time += dt
        _save_timed_checkpoint(total_training_time, completed_steps, debiased_smooth_loss)
        if not SMOKE_TEST:
            _prog = total_training_time / max(TRAINING_TIME_BUDGET, 1)
            if _prog >= _periodic_eval_fraction and _prog < 1.0:
                _periodic_eval(
                    total_training_time,
                    completed_steps,
                    debiased_smooth_loss,
                    lrm,
                )
                while _periodic_eval_fraction <= _prog:
                    _periodic_eval_fraction += _PERIODIC_EVAL_INTERVAL

    # Logging
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / BF16_PEAK_FLOPS
    remaining = max(0, TRAINING_TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > COMPILE_WARMUP_STEPS and total_training_time >= TRAINING_TIME_BUDGET:
        break
    if (
        SMOKE_TEST
        and step > COMPILE_WARMUP_STEPS
        and (step - COMPILE_WARMUP_STEPS) >= SMOKE_MAX_OPTIMIZER_STEPS
    ):
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
_eval_bs = DEVICE_BATCH_SIZE
_eval_cap = None
if SMOKE_TEST:
    _eval_bs = max(1, min(DEVICE_BATCH_SIZE, 4))
    _eval_cap = max(MAX_SEQ_LEN * _eval_bs * 2, 8192)
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, _eval_bs, eval_tokens=_eval_cap)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * max(step - COMPILE_WARMUP_STEPS, 0)
    / total_training_time
    / BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

# Save model checkpoint only if this run beats the previous best
_ckpt_path = os.path.join(_HERE, "model.pt")
_prev_bpb = float("inf")
if os.path.exists(_ckpt_path):
    try:
        _prev_bpb = torch.load(_ckpt_path, weights_only=False)["val_bpb"]
    except Exception:
        pass
if val_bpb < _prev_bpb:
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "val_bpb": val_bpb,
        "num_steps": step,
        "total_tokens": total_tokens,
    }, _ckpt_path)
    _write_meta(_ckpt_path, {
        "label": "best",
        "val_bpb": round(val_bpb, 6),
        "num_steps": step,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(total_training_time, 1),
        "train_loss_ema": round(debiased_smooth_loss, 6),
        "hyperparams": {
            "WARMDOWN_RATIO": WARMDOWN_RATIO,
            "FINAL_LR_FRAC": FINAL_LR_FRAC,
            "MATRIX_LR": MATRIX_LR,
            "EMBEDDING_LR": EMBEDDING_LR,
            "MUON_BETA2": MUON_BETA2,
            "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
            "DEPTH": DEPTH,
        },
    })
    print(f"checkpoint:       {_ckpt_path} (new best: {val_bpb:.6f} < {_prev_bpb:.6f})")
else:
    print(f"checkpoint:       skipped (val_bpb {val_bpb:.6f} >= best {_prev_bpb:.6f})")
