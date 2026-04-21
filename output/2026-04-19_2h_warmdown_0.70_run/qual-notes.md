# Qualitative notes --- `2026-04-19_2h_warmdown_0.70_run`

WARMDOWN_RATIO=0.7 · final val_bpb=1.0515

> Auto-generated first-pass scores. Objective columns (coherent span, repetition, syntax) derived from heuristics; subjective columns (specificity, interest, prompt adherence, weirdness) left blank --- fill when inspired. Override any cell freely; regenerate with `python scripts/gen_qual_notes.py <run_dir>` to refresh only the objective columns (user edits to subjective columns will be overwritten on re-run, so commit before regenerating).

## Scoring

Scales: **coherent span** 1 (poor) -> 5 (excellent) · **repetition** none / late / medium / early · **syntax** poor / mixed / good · **specificity** specific / generic / filler · **interest** 1--5 · **prompt adherence** follows / drifts / ignores · **weirdness** high / balanced / sterilized

| Milestone | Coherent span | Repetition | Syntax | Specificity | Interest | Prompt adherence | Weirdness |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5m | 3 | late | good | | | | |
| 15m | 2 | medium | good | | | | |
| 30m | 3 | late | good | | | | |
| 1h | 3 | medium | good | | | | |
| 2h | 3 | late | good | | | | |

## Heuristic detail (objective signals behind the table)

| Milestone | mean lex-div | mean non-ascii frac | mean len (words) | rep-onset / N | repetition rate |
| --- | ---: | ---: | ---: | :---: | ---: |
| 5m | 0.581 | 0.0004 | 145 | 85/145 | 0.71 |
| 15m | 0.494 | 0.0012 | 147 | 60/147 | 1.00 |
| 30m | 0.545 | 0.0005 | 157 | 81/157 | 0.86 |
| 1h | 0.628 | 0.0005 | 147 | 68/147 | 0.71 |
| 2h | 0.583 | 0.0000 | 163 | 103/163 | 0.43 |

## User notes

_Free-form observations. Fill when inspired._

## Curios hints

_Completions worth promoting to [`docs/curios.md`](../../docs/curios.md). Format: `- <milestone> `<prompt_name>` --- brief note`._

