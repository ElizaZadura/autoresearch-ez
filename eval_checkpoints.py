"""
Evaluate val_bpb for all timed checkpoints and save to output/eval_results.csv.
Run this once after a training run before opening development.ipynb.

Usage:
    uv run eval_checkpoints.py
    uv run eval_checkpoints.py --force   # re-evaluate even if results exist
"""

import argparse
import csv
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Re-evaluate all checkpoints")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model (self-contained — do not import train.py)
# ---------------------------------------------------------------------------

_sdpa_sliding_mask_cache: dict = {}


def sdpa_flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
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
    key = (T, w_left, str(q.device))
    mask = _sdpa_sliding_mask_cache.get(key)
    if mask is None:
        ii = torch.arange(T, device=q.device).unsqueeze(1)
        jj = torch.arange(T, device=q.device).unsqueeze(0)
        mask = ((jj <= ii) & (jj >= (ii - (w_left - 1)))).view(1, 1, T, T)
        _sdpa_sliding_mask_cache[key] = mask
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, is_causal=False, **gqa_kw)
    return out.transpose(1, 2).contiguous()


try:
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    _fa_repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    flash_attn_func = get_kernel(_fa_repo).flash_attn_interface.flash_attn_func
except Exception:
    flash_attn_func = sdpa_flash_attn_func


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
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = ((8 * config.n_embd // 3) // 128) * 128
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

    def forward(self, x, ve, cos_sin, window_size):
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
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        logits = self.lm_head(norm(x)).float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
        return logits

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

from prepare import Tokenizer, evaluate_bpb

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_directory()

LABELS = ["5m", "15m", "30m", "1h", "2h", "4h"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "output", "eval_results.csv")

os.makedirs(os.path.join(HERE, "output"), exist_ok=True)

# Load existing results to avoid re-evaluating
existing = {}
if os.path.exists(OUT_CSV) and not args.force:
    with open(OUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            existing[row["label"]] = row

results = []
for label in LABELS:
    pt_path = os.path.join(HERE, f"model_{label}.pt")
    json_path = pt_path.replace(".pt", ".json")
    if not os.path.exists(pt_path):
        print(f"[{label}] skipped (not found)")
        continue

    # Load metadata
    meta = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)

    if label in existing and not args.force:
        print(f"[{label}] cached  val_bpb={existing[label]['val_bpb']}")
        results.append(existing[label])
        continue

    print(f"[{label}] evaluating ...", end=" ", flush=True)
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        val_bpb = evaluate_bpb(model, tokenizer, batch_size=32)

    del model
    torch.cuda.empty_cache()

    row = {
        "label": label,
        "elapsed_seconds": meta.get("elapsed_seconds", ck.get("elapsed_seconds", "")),
        "num_steps": meta.get("num_steps", ck.get("num_steps", "")),
        "train_loss_ema": meta.get("train_loss_ema", ""),
        "val_bpb": round(val_bpb, 6),
    }
    results.append(row)

    # Update JSON sidecar with val_bpb
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        meta["val_bpb"] = round(val_bpb, 6)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"val_bpb={val_bpb:.6f}")

if results:
    fieldnames = ["label", "elapsed_seconds", "num_steps", "train_loss_ema", "val_bpb"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {OUT_CSV}")
else:
    print("No checkpoints found. Run a training session first.")
