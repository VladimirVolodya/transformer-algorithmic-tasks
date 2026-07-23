# transformer-algorithmic-tasks

From-scratch decoder-only transformer for **algorithmic tasks**. This branch validates the
training/evaluation harness on **reversed-digit addition** by reproducing a well-documented
length-generalization effect:

> A small transformer learns addition in-distribution with **any** positional encoding, but
> **generalizes to longer operands** very differently: learned **absolute** PE collapses toward 0,
> while **NoPE** and **RoPE** degrade gracefully.

If this reproduces, the data generator, loss masking, training loop, and exact-match evaluator are
trustworthy and can be reused for the larger ablation.

## Task / data format

Operands are written **least-significant digit first** so the carry propagates left-to-right:

```
a=58392, b=74918, s=133310
tokens:  2 9 3 8 5 + 8 1 9 4 7 = 0 1 3 3 3 1 <eos>
         └ rev(a) ┘   └ rev(b) ┘   └── rev(s) ──┘
```

The loss is computed on the **answer tokens only** (`rev(s)` + `<eos>`). Vocabulary (14 tokens):
digits `0-9`, `+`, `=`, `<eos>`, `<pad>`.

- **Train:** operand lengths 1–5 (fresh, generated on the fly).
- **In-distribution eval:** lengths 1–5.
- **OOD eval:** lengths 6…15, evaluated per length (the money-plot curve).

## Model

Small decoder-only transformer (**~0.8M params, CPU-friendly**): `d_model=128, n_layers=4,
n_heads=2, head_dim=64, d_ff=512`, pre-norm, GELU, causal. One flag
`model.pe_variant ∈ {absolute, nope, rope}` selects the positional encoding; everything else is
identical, so any OOD difference is attributable to the PE alone.

**Candidate — shifted absolute PE (`abs_shift`).** Same architecture and parameter count as
`absolute`, but during training the position indices of each sequence are shifted by a random
offset `k ~ U{0, max_len − T}` (SHAPE-style), so the entire `[0, max_len)` embedding table is
trained; at eval the offset is 0. Train it with:

```bash
uv run python train.py model=transformer_shifted writer.run_name=addition_abs_shift
# equivalently: model.pe_variant=abs_shift
```

## Install

Uses [`uv`](https://docs.astral.sh/uv/) (Python pinned in `.python-version`).

```bash
uv sync
uv run pre-commit install   # optional
```

## Train the three variants

```bash
bash scripts/train_all_variants.sh          # absolute, nope, rope (same seed + budget)
# or individually:
uv run python train.py model.pe_variant=nope writer.run_name=addition_nope
```

Checkpoints land in `saved/<run_name>/model_best.pth`. On CPU each variant trains in minutes–tens
of minutes. If too slow, reduce `trainer.epoch_len` / `trainer.n_epochs` (do **not** shrink the
model — its ~0.8M size is fixed by the design).

## Evaluate

Single length (autoregressive greedy exact-match):

```bash
uv run python inference.py model.pe_variant=absolute \
  inferencer.from_pretrained=saved/addition_absolute/model_best.pth \
  inferencer.eval_length=10
```

Full per-length curve, money plot, and PASS/FAIL verdict:

```bash
uv run jupyter lab notebooks/length_generalization.ipynb   # Restart & Run All
```

## Multi-task benchmark (ADDITION / SORT / DYCK / INDEX)

Beyond addition, three more algorithmic probes share one 19-token vocabulary and one
train/eval pipeline (`src/tasks/`):

| Task | Tests | Example (`prompt = answer <eos>`) |
| --- | --- | --- |
| ADDITION | carry propagation | `29385+81947= 013331` (reversed digits) |
| SORT | global comparisons | `372= 237` (ascending) |
| DYCK | stack thinking | `([])()= 1` / `([)]= 0` (Dyck-2 membership, Bhattamishra et al. 2020) |
| INDEX | positional access | `372,1= 7` (0-based; index kept single-digit even OOD) |

All data is model-independent and seeded: the train stream depends only on
`(task, train_seed)` and each eval set only on `(task, length, eval_seed)`, so **every
architecture is trained and scored on identical values**. `run_benchmark` trains a fresh
model per task (same recipe as `train.yaml`: AdamW + OneCycle, answer-only masked CE,
5000×256 budget), evaluates exact match per length in-domain (1–5) and OOD (6–15), and
returns/saves a JSON with per-task metrics:

```bash
uv run python benchmark.py model.pe_variant=rope    # -> saved/benchmark_rope.json
```

or from code:

```python
from functools import partial
from src.benchmark import run_benchmark
from src.model.transformer import DecoderTransformer

make_model = partial(DecoderTransformer, d_model=128, n_layers=4, n_heads=2,
                     head_dim=64, d_ff=512, max_len=64, pe_variant="rope")
results = run_benchmark(make_model)   # {"tasks": {"addition": {"in_domain": ..., "ood": ...}, ...}}
```

DYCK negatives are balanced words corrupted at one position (near-misses that defeat
count-only heuristics); its exact-match floor is the 0.5 coin-flip baseline.

## Experiment tracking (wandb)

Defaults to **offline** — logs to `./wandb/`, no account or login needed. To use live dashboards:

```bash
wandb login                       # stores the key in ~/.netrc (outside the repo)
# or: export WANDB_API_KEY=...    # e.g. from a gitignored .env
uv run python train.py writer.mode=online writer.entity=<your-entity>
```

**Never commit secrets.** `.env` is gitignored; a template lives in `.env.example`. API keys must
come from `wandb login` or the `WANDB_API_KEY` env var — they are never hardcoded in configs.

## Tests

```bash
uv run pytest -q     # data-format + loss-mask guardrails and model sanity (CPU)
```

## Success criteria (checked mechanically in the notebook verdict)

1. In-distribution exact-match ≥ ~0.95 for all three variants.
2. OOD: absolute PE collapses toward 0; NoPE (and RoPE) stay materially higher on the first OOD lengths.
3. The accuracy-vs-length curve visibly separates the variants past the train boundary (length 5).

## Layout

| Path | Purpose |
| --- | --- |
| `src/datasets/` | tokenizer, on-the-fly `AdditionDataset`, padding collate |
| `src/tasks/` | multi-task benchmark: shared vocab + ADDITION/SORT/DYCK/INDEX tasks, eval |
| `src/benchmark.py` | `run_benchmark()` — train+eval a fixed architecture on all four tasks |
| `src/model/` | `DecoderTransformer` (+ RoPE helpers), `generate()` |
| `src/loss/`, `src/metrics/` | masked cross-entropy, teacher-forced exact match |
| `src/evaluation/` | autoregressive per-length exact-match sweep (authoritative metric) |
| `src/trainer/` | training loop (`Trainer`) + AR evaluator (`Inferencer`) |
| `src/configs/` | Hydra configs; `train.py` / `inference.py` entry points |
| `notebooks/` | `length_generalization.ipynb` — money plot + verdict |

Built on the `ml_project_template` PyTorch/Hydra project structure.
