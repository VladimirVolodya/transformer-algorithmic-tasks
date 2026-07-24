# transformer-algorithmic-tasks

From-scratch decoder-only transformer for **algorithmic tasks**. The Phase-2
experiment compares four ~800k architectures (Attention vs MLP budget) on
ADDITION / SORT / DYCK / INDEX with absolute PE, lengths ID 1–10 / OOD 20–30.

## Install

```bash
uv sync
uv run pre-commit install   # optional
uv run pytest -q
```

## Architecture grid (main experiment)

4 architectures × 4 tasks × 10 seeds = **160 runs**. No weight checkpoints —
each cell appends one row to `results.csv`.

```bash
# full grid (long; resume-safe if interrupted)
uv run python run_grid.py

# smoke
uv run python run_grid.py --steps 20 --seeds 0 --arches B --tasks addition \
  --n-eval 16 --csv /tmp/grid_smoke.csv
```

| Arch | Role | d_model | n_heads | d_ff |
| --- | --- | ---: | ---: | ---: |
| A | Attention-heavy | 192 | 6 | 192 |
| B | Baseline | 128 | 4 | 512 |
| C | Bottleneck | 80 | 2 | 1088 |
| D | Extreme bottleneck | 48 | 1 | 2048 |

PE is fixed to **absolute**; `max_len=128`. CSV columns:
`Task, Architecture, Seed, ID_Accuracy, OOD_Accuracy`.

## Single-architecture benchmark

Still available via Hydra (JSON per-length curves):

```bash
uv run python benchmark.py
uv run python scripts/plot_benchmark.py saved/benchmark_absolute.json
```

Defaults: train lengths 1–10, OOD 20–30, 5000 steps, AdamW.

## Tasks

Shared 19-token vocabulary; shape `prompt = answer <eos>`; answer-only masked CE;
greedy AR exact match.

| Task | Example |
| --- | --- |
| ADDITION | `29385+81947=013331` (reversed digits) |
| SORT | `372=237` |
| DYCK | `([])()=1` / `([)]=0` |
| INDEX | `372,1=7` |

## Layout

| Path | Purpose |
| --- | --- |
| `run_grid.py` | 160-run architecture grid → `results.csv` |
| `benchmark.py` | single-architecture Hydra benchmark → JSON |
| `scripts/plot_benchmark.py` | plot / verdict from JSON |
| `src/model/architectures.py` | Arch A–D configs |
| `src/benchmark.py` | `run_one_task` / `run_benchmark` |
| `src/tasks/` | tasks, vocab, AR eval |

## Tests

```bash
uv run pytest -q
```
