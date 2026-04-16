"""
Run the fixed prompt pack against one or more checkpoints and write results to
output/<label>_prompts.txt (or stdout with --stdout).

Usage:
    uv run eval_prompts.py model_5m.pt
    uv run eval_prompts.py model_5m.pt model_15m.pt model_30m.pt
    uv run eval_prompts.py model_5m.pt --max-new-tokens 300
    uv run eval_prompts.py model_5m.pt --stdout
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime

# ---------------------------------------------------------------------------
# Fixed prompt pack — do not change between runs
# ---------------------------------------------------------------------------

PROMPTS = [
    ("plain_continuation",   "The old man walked slowly toward the river and"),
    ("factual_fragment",     "The capital of France is Paris, and the population of"),
    ("longitudinal_anchor",  "In one sentence, the meaning of life is"),
    ("structurally_awkward", "Despite the fact that however, the reason why because"),
    ("anomaly_lure",         "Ground control to Major Snorf,"),
    ("signature",            "Once upon a time there was a small"),
    ("continuation",  "The instructions were clear until line seven:"),
]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+", help="Path(s) to .pt checkpoint files")
parser.add_argument("--max-new-tokens", type=int, default=200)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--top-p", type=float, default=1.0)
parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
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

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        logits = self.lm_head(norm(x)).float()
        return 15 * torch.tanh(logits / 15)

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

from prepare import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_directory()


@torch.no_grad()
def generate(model, config, prompt_text: str, max_new_tokens: int) -> str:
    ids = tokenizer.encode(prompt_text, prepend=tokenizer.bos_token_id)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.sequence_len:]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / max(args.temp, 1e-6)
        if args.top_k > 0:
            topk_vals, _ = logits.topk(args.top_k, dim=-1)
            logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
        if args.top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum_probs = sorted_logits.softmax(-1).cumsum(-1)
            sorted_logits[cum_probs - sorted_logits.softmax(-1) > args.top_p] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0, len(ids):].tolist())


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ck["config"])
    model = GPT(config)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, config, ck


def run_prompts(ckpt_path: str):
    print(f"Loading {ckpt_path} ...", end=" ", flush=True)
    model, config, ck = load_model(ckpt_path)
    label = ck.get("label", os.path.splitext(os.path.basename(ckpt_path))[0])
    val_bpb = ck.get("val_bpb", "n/a")
    elapsed = ck.get("elapsed_seconds", "n/a")
    num_steps = ck.get("num_steps", "n/a")
    print("done")

    lines = []
    lines.append(f"checkpoint: {ckpt_path}")
    lines.append(f"label:      {label}")
    lines.append(f"val_bpb:    {val_bpb}")
    lines.append(f"elapsed:    {elapsed}s")
    lines.append(f"num_steps:  {num_steps}")
    lines.append(f"generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"temp:       {args.temp}  top_k: {args.top_k}  top_p: {args.top_p}  max_new_tokens: {args.max_new_tokens}")
    lines.append("=" * 72)

    for name, prompt in PROMPTS:
        completion = generate(model, config, prompt, args.max_new_tokens)
        lines.append(f"\n[{name}]")
        lines.append(f"PROMPT:     {prompt}")
        lines.append(f"COMPLETION: {prompt}{completion}")

    lines.append("\n" + "=" * 72)
    return label, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

os.makedirs("output", exist_ok=True)

for ckpt_path in args.checkpoints:
    label, text = run_prompts(ckpt_path)
    if args.stdout:
        print(text)
    else:
        out_path = os.path.join("output", f"{label}_prompts.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {out_path}")
