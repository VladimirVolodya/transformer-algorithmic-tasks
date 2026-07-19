# transformer-algorithmic-tasks

From-scratch decoder-only transformer for **algorithmic tasks**. One entry point
trains and evaluates a fixed architecture on ADDITION / SORT / DYCK / INDEX with
identical seeded data streams, so architecture comparisons stay fair.

## Install

Uses [`uv`](https://docs.astral.sh/uv/) (Python pinned in `.python-version`).

```bash
uv sync
uv run pre-commit install   # optional
uv run pytest -q
```

## Run the benchmark

```bash
uv run python benchmark.py model.pe_variant=rope    # -> saved/benchmark_rope.json
# shorter smoke:
uv run python benchmark.py model.pe_variant=nope benchmark.steps=50 'benchmark.tasks=[addition]'
```

### Inspect results

After one or more runs, plot accuracy vs length and print a short verdict:

```bash
uv run python scripts/plot_benchmark.py saved/benchmark_rope.json
# compare PE variants once you have several JSONs:
uv run python scripts/plot_benchmark.py saved/benchmark_*.json
```

Writes `figures/length_generalization.png` and prints per-length tables plus
PASS/FAIL on in-domain mean and early-OOD (lengths 6–8).

Hydra overrides go on the CLI. Useful knobs live in
[`src/configs/benchmark.yaml`](src/configs/benchmark.yaml):

| Knob | Default | Meaning |
| --- | --- | --- |
| `model.pe_variant` | `absolute` | `absolute` \| `nope` \| `rope` |
| `benchmark.tasks` | all four | subset, e.g. `[addition,sort]` |
| `benchmark.steps` | 5000 | optimizer steps per task |
| `benchmark.train_max_len` | 5 | in-distribution operand / sequence length |
| `benchmark.ood_lengths` | 6…15 | per-length OOD exact-match sweep |

Or from code:

```python
from functools import partial
from src.benchmark import run_benchmark
from src.model.transformer import DecoderTransformer

make_model = partial(DecoderTransformer, d_model=128, n_layers=4, n_heads=2,
                     head_dim=64, d_ff=512, max_len=64, pe_variant="rope")
results = run_benchmark(make_model)   # {"tasks": {"addition": {...}, ...}}
```

## Tasks

All four share one 19-token vocabulary and the shape `prompt = answer <eos>`.
Loss is answer-only masked CE. Eval is greedy autoregressive exact match.

| Task | Tests | Example (`prompt = answer <eos>`) |
| --- | --- | --- |
| ADDITION | carry propagation | `29385+81947= 013331` (reversed digits) |
| SORT | global comparisons | `372= 237` (ascending) |
| DYCK | stack thinking | `([])()= 1` / `([)]= 0` (Dyck-2 membership) |
| INDEX | positional access | `372,1= 7` (0-based; index kept single-digit even OOD) |

- **Train:** lengths 1–5 (fresh, generated on the fly).
- **In-domain eval:** lengths 1–5.
- **OOD eval:** lengths 6…15, scored per length.

Train stream depends only on `(task, train_seed)`; each eval set only on
`(task, length, eval_seed)`, so every architecture sees the same values.

## Model

Small decoder-only transformer (**~0.8M params, CPU-friendly**):
`d_model=128, n_layers=4, n_heads=2, head_dim=64, d_ff=512`, pre-norm, GELU,
causal. Flag `model.pe_variant ∈ {absolute, nope, rope}` selects positional
encoding; everything else is identical.

The three-way PE length-generalization money-plot (train all variants, plot
accuracy vs length) is deferred; change `pe_variant` to train one architecture
end-to-end on the benchmark.

## Layout

| Path | Purpose |
| --- | --- |
| `benchmark.py` | CLI entry point |
| `scripts/plot_benchmark.py` | accuracy-vs-length plot + verdict from JSON |
| `src/benchmark.py` | `run_benchmark()` — train+eval a fixed architecture on all tasks |
| `src/tasks/` | shared vocab + ADDITION/SORT/DYCK/INDEX, batching, AR eval |
| `src/model/` | `DecoderTransformer` (+ RoPE helpers), `generate()` |
| `src/loss/` | masked cross-entropy (answer tokens only) |
| `src/configs/` | Hydra: `benchmark.yaml` + `model/transformer.yaml` |

## Tests

```bash
uv run pytest -q
```
