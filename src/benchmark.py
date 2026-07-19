"""Multi-task train-and-evaluate benchmark for a fixed architecture.

:func:`run_benchmark` takes a *model factory*, trains a fresh model per task
(ADDITION, SORT, DYCK, INDEX) on identical data streams, evaluates exact match
in-domain and out-of-domain per length, and returns a JSON-serializable dict.

Data is model-independent and fully seeded -- the train stream depends only on
``(task, train_seed)`` and each eval set only on ``(task, length, eval_seed)``
-- so different architectures are trained and scored on **the same values**.

Example::

    from functools import partial
    from src.benchmark import run_benchmark
    from src.model.transformer import DecoderTransformer

    make_model = partial(DecoderTransformer, d_model=128, n_layers=4,
                         n_heads=2, head_dim=64, d_ff=512, max_len=64,
                         pe_variant="rope")
    results = run_benchmark(make_model, save_path="saved/benchmark_rope.json")

The factory is called as ``make_model(vocab_size=..., pad_id=...)`` and must
return a fresh ``nn.Module`` with the repo's ``forward``/``generate`` contract.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from src.loss.masked_ce import MaskedCrossEntropyLoss
from src.tasks import TASKS, TaskTokenizer
from src.tasks.data import make_train_batch
from src.tasks.evaluate import sweep_lengths

DEFAULT_TASKS = ("addition", "sort", "dyck", "index")


def train_on_task(
    model,
    task,
    tokenizer,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.01,
    min_len: int,
    max_len: int,
    seed: int,
    device: str,
    max_grad_norm: float = 1.0,
    log_every: int = 250,
):
    """Train ``model`` on ``task`` with the repo's standard recipe.

    AdamW + OneCycleLR (5% warmup, cosine anneal), masked answer-only CE,
    grad-norm clipping -- the same recipe as ``src/configs/train.yaml``. The
    train stream comes from a dedicated ``RandomState(seed)``, so it is
    identical for every architecture. Returns the mean loss over the last
    ``log_every`` steps.
    """
    rng = np.random.RandomState(seed)
    criterion = MaskedCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=steps,
        pct_start=0.05,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    model.train()
    losses = []
    for step in range(1, steps + 1):
        batch = make_train_batch(task, tokenizer, rng.randint, batch_size, min_len, max_len)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = criterion(**outputs, **batch)["loss"]
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == steps:
            print(
                f"    [{task.name}] step {step}/{steps} "
                f"loss={np.mean(losses[-log_every:]):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
    return float(np.mean(losses[-log_every:]))


def run_benchmark(
    make_model,
    tasks=DEFAULT_TASKS,
    *,
    steps: int = 5000,
    batch_size: int = 256,
    lr: float = 3e-4,
    train_min_len: int = 1,
    train_max_len: int = 5,
    id_lengths=tuple(range(1, 6)),
    ood_lengths=tuple(range(6, 16)),
    n_eval: int = 300,
    train_seed: int = 1,
    eval_seed: int = 0,
    device: str = "auto",
    eval_batch_size: int = 256,
    save_path: str | None = None,
) -> dict:
    """Train and evaluate one architecture on each task independently.

    Args:
        make_model: callable ``(vocab_size, pad_id) -> nn.Module`` returning a
            **fresh** model (called once per task).
        tasks: task names from :data:`src.tasks.TASKS`.
        steps / batch_size / lr: training budget per task (defaults match the
            validated addition recipe: 5000 steps x 256).
        train_min_len / train_max_len: in-distribution length range.
        id_lengths / ood_lengths: per-length eval sweeps (in/out of domain).
        n_eval: examples per (task, length) eval set.
        train_seed / eval_seed: fix the data; keep them constant across
            architectures so everyone sees the same train/test values.
        device: "auto" resolves to CUDA, else MPS, else CPU.
        save_path: optional path to also dump the results dict as JSON.

    Returns:
        JSON-serializable dict: benchmark settings + per-task metrics
        (in-domain and OOD exact match per length and averaged).
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    id_lengths = list(id_lengths)
    ood_lengths = list(ood_lengths)

    results = {
        "benchmark": {
            "tasks": list(tasks),
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "train_min_len": train_min_len,
            "train_max_len": train_max_len,
            "id_lengths": id_lengths,
            "ood_lengths": ood_lengths,
            "n_eval": n_eval,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "device": device,
        },
        "tasks": {},
    }

    for name in tasks:
        task = TASKS[name]()
        tokenizer = TaskTokenizer()
        model = make_model(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"== {name}: training fresh model ({n_params:,} params) on {device}")

        t0 = time.time()
        final_loss = train_on_task(
            model,
            task,
            tokenizer,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            min_len=train_min_len,
            max_len=train_max_len,
            seed=train_seed,
            device=device,
        )
        train_time = time.time() - t0

        common = dict(
            n=n_eval, seed=eval_seed, device=device,
            tokenizer=tokenizer, batch_size=eval_batch_size,
        )
        id_acc = sweep_lengths(model, task, id_lengths, **common)
        ood_acc = sweep_lengths(model, task, ood_lengths, **common)

        results["tasks"][name] = {
            "n_params": n_params,
            "final_train_loss": final_loss,
            "train_seconds": round(train_time, 1),
            "in_domain": {str(k): v for k, v in id_acc.items()},
            "in_domain_mean": float(np.mean(list(id_acc.values()))),
            "ood": {str(k): v for k, v in ood_acc.items()},
            "ood_mean": float(np.mean(list(ood_acc.values()))),
        }
        print(
            f"== {name}: in-domain mean={results['tasks'][name]['in_domain_mean']:.3f} "
            f"OOD mean={results['tasks'][name]['ood_mean']:.3f} "
            f"({train_time:.0f}s train)"
        )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"Saved results to {path}")

    return results
