"""Generate qual-notes.md for one or more run dirs.

Auto-populates the objective columns of the scoring table (repetition onset,
lexical diversity, mean completion length) from the run's milestone prompt
files. Subjective columns are left with suggested starter values derived from
heuristic proxies; user can override freely.

Usage:
    .venv/Scripts/python.exe scripts/gen_qual_notes.py <run_dir>...
    .venv/Scripts/python.exe scripts/gen_qual_notes.py output/2026-04-19_2h_warmdown_*_run

Overwrites existing qual-notes.md unless --no-overwrite is passed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Iterable

LABELS = ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h"]
PROMPT_NAMES = [
    "plain_continuation",
    "factual_fragment",
    "longitudinal_anchor",
    "structurally_awkward",
    "anomaly_lure",
    "signature",
    "continuation",
]


def parse_prompt_file(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\[", text)
    out: dict[str, str] = {}
    for block in blocks[1:]:
        name_end = block.index("]")
        name = block[:name_end]
        body = block[name_end + 1 :]
        m = re.search(r"COMPLETION:\s*(.*)", body, re.DOTALL)
        if m:
            out[name] = m.group(1).strip()
    return out


def repetition_onset(text: str, ngram: int = 4) -> int | None:
    """Word index of the first repeated n-gram. None if no repetition."""
    words = text.lower().split()
    seen: dict[tuple, int] = {}
    for i in range(len(words) - ngram + 1):
        g = tuple(words[i : i + ngram])
        if g in seen:
            return i
        seen[g] = i
    return None


def lexical_diversity(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non = sum(1 for c in text if ord(c) > 127 or c == "\ufffd")
    return non / len(text)


def bucket_repetition(onset: float | None, length: float) -> str:
    """Map mean repetition onset (word index) into a qualitative label."""
    if onset is None:
        return "none"
    ratio = onset / max(length, 1)
    if ratio < 0.15:
        return "early"
    if ratio < 0.50:
        return "medium"
    return "late"


def bucket_coherence(div: float) -> int:
    """Map lexical diversity into 1..5."""
    if div >= 0.75:
        return 5
    if div >= 0.65:
        return 4
    if div >= 0.50:
        return 3
    if div >= 0.35:
        return 2
    return 1


def bucket_syntax(non_ascii: float, length: float) -> str:
    """Heuristic syntax label: high non-ascii ratio or very short = poor."""
    if non_ascii > 0.05 or length < 20:
        return "poor"
    if non_ascii > 0.02:
        return "mixed"
    return "good"


def run_ratio(run_dir: Path) -> float | None:
    for meta in sorted(run_dir.glob("model_*.json")):
        try:
            d = json.loads(meta.read_text())
            hp = d.get("hyperparams", {})
            if "WARMDOWN_RATIO" in hp:
                return float(hp["WARMDOWN_RATIO"])
        except Exception:
            continue
    m = re.search(r"_(\d+(?:\.\d+)?)_run$", run_dir.name)
    return float(m.group(1)) if m else None


def run_final_val_bpb(run_dir: Path) -> float | None:
    best = None
    best_elapsed = -1.0
    for meta in sorted(run_dir.glob("model_*.json")):
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        el = d.get("elapsed_seconds") or 0
        if el > best_elapsed and d.get("val_bpb") is not None:
            best_elapsed = el
            best = d.get("val_bpb")
    return best


def score_milestone(prompts: dict[str, str]) -> dict:
    """Compute aggregate scores across all prompts at one milestone."""
    if not prompts:
        return {}
    onsets = []
    divs = []
    non_ascii_rs = []
    lengths = []
    for pname, text in prompts.items():
        words = text.split()
        lengths.append(len(words))
        divs.append(lexical_diversity(text))
        non_ascii_rs.append(non_ascii_ratio(text))
        o = repetition_onset(text)
        if o is not None:
            onsets.append(o)

    mean_div = statistics.mean(divs) if divs else 0.0
    mean_non_ascii = statistics.mean(non_ascii_rs) if non_ascii_rs else 0.0
    mean_len = statistics.mean(lengths) if lengths else 0.0
    mean_onset = statistics.mean(onsets) if onsets else None
    rep_rate = len(onsets) / max(len(prompts), 1)

    return {
        "coherent_span": bucket_coherence(mean_div),
        "repetition": bucket_repetition(mean_onset, mean_len),
        "repetition_rate": rep_rate,
        "syntax": bucket_syntax(mean_non_ascii, mean_len),
        "mean_div": mean_div,
        "mean_non_ascii": mean_non_ascii,
        "mean_len": mean_len,
        "mean_onset": mean_onset,
    }


def render_qual_notes(run_dir: Path) -> str:
    ratio = run_ratio(run_dir)
    val_bpb = run_final_val_bpb(run_dir)

    milestone_scores: dict[str, dict] = {}
    for label in LABELS:
        p = run_dir / f"{label}_prompts.txt"
        if not p.exists():
            continue
        try:
            prompts = parse_prompt_file(p)
        except Exception:
            continue
        milestone_scores[label] = score_milestone(prompts)

    lines: list[str] = []
    lines.append(f"# Qualitative notes --- `{run_dir.name}`")
    lines.append("")
    hdr_parts = []
    if ratio is not None:
        hdr_parts.append(f"WARMDOWN_RATIO={ratio}")
    if val_bpb is not None:
        hdr_parts.append(f"final val_bpb={val_bpb:.4f}")
    if hdr_parts:
        lines.append(" \u00b7 ".join(hdr_parts))
        lines.append("")

    lines.append(
        "> Auto-generated first-pass scores. Objective columns (coherent span, repetition, "
        "syntax) derived from heuristics; subjective columns (specificity, interest, "
        "prompt adherence, weirdness) left blank --- fill when inspired. Override any "
        "cell freely; regenerate with `python scripts/gen_qual_notes.py <run_dir>` to "
        "refresh only the objective columns (user edits to subjective columns will be "
        "overwritten on re-run, so commit before regenerating)."
    )
    lines.append("")

    # ----- Scoring table
    lines.append("## Scoring")
    lines.append("")
    lines.append(
        "Scales: **coherent span** 1 (poor) -> 5 (excellent) \u00b7 "
        "**repetition** none / late / medium / early \u00b7 "
        "**syntax** poor / mixed / good \u00b7 "
        "**specificity** specific / generic / filler \u00b7 "
        "**interest** 1--5 \u00b7 "
        "**prompt adherence** follows / drifts / ignores \u00b7 "
        "**weirdness** high / balanced / sterilized"
    )
    lines.append("")
    lines.append(
        "| Milestone | Coherent span | Repetition | Syntax | Specificity | Interest "
        "| Prompt adherence | Weirdness |"
    )
    lines.append(
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for label in LABELS:
        if label not in milestone_scores:
            continue
        s = milestone_scores[label]
        lines.append(
            f"| {label} | {s['coherent_span']} | {s['repetition']} | {s['syntax']} "
            f"| | | | |"
        )
    lines.append("")

    # ----- Heuristic detail (kept for transparency on how the above was derived)
    lines.append("## Heuristic detail (objective signals behind the table)")
    lines.append("")
    lines.append(
        "| Milestone | mean lex-div | mean non-ascii frac | mean len (words) | "
        "rep-onset / N | repetition rate |"
    )
    lines.append("| --- | ---: | ---: | ---: | :---: | ---: |")
    for label in LABELS:
        if label not in milestone_scores:
            continue
        s = milestone_scores[label]
        onset = "none" if s["mean_onset"] is None else f"{s['mean_onset']:.0f}/{s['mean_len']:.0f}"
        lines.append(
            f"| {label} | {s['mean_div']:.3f} | {s['mean_non_ascii']:.4f} "
            f"| {s['mean_len']:.0f} | {onset} | {s['repetition_rate']:.2f} |"
        )
    lines.append("")

    # ----- User-owned sections
    lines.append("## User notes")
    lines.append("")
    lines.append("_Free-form observations. Fill when inspired._")
    lines.append("")

    lines.append("## Curios hints")
    lines.append("")
    lines.append(
        "_Completions worth promoting to [`docs/curios.md`](../../docs/curios.md). "
        "Format: `- <milestone> `<prompt_name>` --- brief note`._"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_qual_notes(run_dir: Path, overwrite: bool = True) -> Path:
    out = run_dir / "qual-notes.md"
    if out.exists() and not overwrite:
        print(f"  skip (exists): {out}")
        return out
    md = render_qual_notes(run_dir)
    out.write_text(md, encoding="utf-8")
    print(f"  wrote: {out}  ({len(md):,} chars)")
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="Run directories to process")
    ap.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip run dirs where qual-notes.md already exists.",
    )
    args = ap.parse_args(argv)

    for rd_str in args.run_dirs:
        rd = Path(rd_str)
        if not rd.is_dir():
            print(f"  not a dir: {rd}")
            continue
        write_qual_notes(rd, overwrite=not args.no_overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
