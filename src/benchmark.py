"""Multi-task train-and-evaluate benchmark for a fixed architecture.

:func:`run_benchmark` / :func:`run_one_task` take a *model factory*, train a
fresh model on identical seeded data streams, and evaluate exact match
in-domain (lengths 1–10) and out-of-domain (lengths 11–15). Weights are not
saved — callers log metrics only.

Example::

    from functools import partial
    from src.benchmark import run_benchmark
    from src.model.architectures import ARCHITECTURES
    from src.model.transformer import DecoderTransformer

    make_model = partial(DecoderTransformer, **ARCHITECTURES["B"])
    results = run_benchmark(make_model)
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
DEFAULT_ID_LENGTHS = tuple(range(1, 11))
DEFAULT_OOD_LENGTHS = tuple(range(11, 16))


def resolve_device(device: str = "auto") -> str:
    """Map ``auto`` to CUDA, else MPS, else CPU; pass through explicit names."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    eval_id_fn=None,
    early_stop_id: float | None = 0.99,
    early_stop_every: int = 1000,
):
    """Train ``model`` on ``task`` with the repo's standard recipe.

    AdamW + OneCycleLR (5% warmup, cosine anneal), masked answer-only CE,
    grad-norm clipping. Optional early stop when ``eval_id_fn(model)`` reaches
    ``early_stop_id``. Returns ``(final_loss, steps_trained)``.
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
    steps_trained = 0
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
        steps_trained = step
        if step % log_every == 0 or step == steps:
            print(
                f"    [{task.name}] step {step}/{steps} "
                f"loss={np.mean(losses[-log_every:]):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if (
            eval_id_fn is not None
            and early_stop_id is not None
            and early_stop_every > 0
            and step % early_stop_every == 0
        ):
            model.eval()
            id_probe = float(eval_id_fn(model))
            model.train()
            print(
                f"    [{task.name}] early-stop probe @ {step}: "
                f"ID={id_probe:.4f} (threshold={early_stop_id})"
            )
            if id_probe >= early_stop_id:
                print(
                    f"    [{task.name}] early stop at step {step}/{steps} "
                    f"(ID={id_probe:.4f} >= {early_stop_id})"
                )
                break

    return float(np.mean(losses[-log_every:])), steps_trained


def run_one_task(
    make_model,
    task_name: str,
    *,
    steps: int = 15000,
    batch_size: int = 256,
    lr: float = 3e-4,
    train_min_len: int = 1,
    train_max_len: int = 10,
    id_lengths=DEFAULT_ID_LENGTHS,
    ood_lengths=DEFAULT_OOD_LENGTHS,
    n_eval: int = 300,
    train_seed: int = 1,
    eval_seed: int = 0,
    device: str = "auto",
    eval_batch_size: int = 256,
    early_stop_id: float | None = 0.99,
    early_stop_every: int = 1000,
    early_stop_n_eval: int = 100,
) -> dict:
    """Train one fresh model on ``task_name`` and return ID/OOD metrics.

    Does **not** save weights. Returns a dict with ``n_params``,
    ``in_domain`` / ``ood`` per-length maps, and their means.
    """
    device = resolve_device(device)
    id_lengths = list(id_lengths)
    ood_lengths = list(ood_lengths)

    task = TASKS[task_name]()
    tokenizer = TaskTokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"== {task_name}: training fresh model ({n_params:,} params) on {device}")

    def eval_id_fn(m):
        acc = sweep_lengths(
            m,
            task,
            id_lengths,
            n=early_stop_n_eval,
            seed=eval_seed,
            device=device,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
        )
        return float(np.mean(list(acc.values())))

    t0 = time.time()
    final_loss, steps_trained = train_on_task(
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
        eval_id_fn=eval_id_fn if early_stop_id is not None else None,
        early_stop_id=early_stop_id,
        early_stop_every=early_stop_every,
    )
    train_time = time.time() - t0

    common = dict(
        n=n_eval,
        seed=eval_seed,
        device=device,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
    )
    id_acc = sweep_lengths(model, task, id_lengths, **common)
    ood_acc = sweep_lengths(model, task, ood_lengths, **common)

    # Free GPU/MPS memory before the next grid cell.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    result = {
        "n_params": n_params,
        "final_train_loss": final_loss,
        "steps_trained": steps_trained,
        "train_seconds": round(train_time, 1),
        "in_domain": {str(k): v for k, v in id_acc.items()},
        "in_domain_mean": float(np.mean(list(id_acc.values()))),
        "ood": {str(k): v for k, v in ood_acc.items()},
        "ood_mean": float(np.mean(list(ood_acc.values()))),
    }
    print(
        f"== {task_name}: in-domain mean={result['in_domain_mean']:.3f} "
        f"OOD mean={result['ood_mean']:.3f} "
        f"({train_time:.0f}s train, {steps_trained}/{steps} steps)"
    )
    return result


def run_benchmark(
    make_model,
    tasks=DEFAULT_TASKS,
    *,
    steps: int = 15000,
    batch_size: int = 256,
    lr: float = 3e-4,
    train_min_len: int = 1,
    train_max_len: int = 10,
    id_lengths=DEFAULT_ID_LENGTHS,
    ood_lengths=DEFAULT_OOD_LENGTHS,
    n_eval: int = 300,
    train_seed: int = 1,
    eval_seed: int = 0,
    device: str = "auto",
    eval_batch_size: int = 256,
    early_stop_id: float | None = 0.99,
    early_stop_every: int = 1000,
    save_path: str | None = None,
) -> dict:
    """Train and evaluate one architecture on each task independently.

    Args:
        make_model: callable ``(vocab_size, pad_id) -> nn.Module`` returning a
            **fresh** model (called once per task).
        tasks: task names from :data:`src.tasks.TASKS`.
        steps / batch_size / lr: training budget per task (defaults: 15000 x 256,
            with early stop when ID exact-match hits 0.99).
        train_min_len / train_max_len: in-distribution length range.
        id_lengths / ood_lengths: per-length eval sweeps (in/out of domain).
        n_eval: examples per (task, length) eval set.
        train_seed / eval_seed: fix the data; keep them constant across
            architectures so everyone sees the same train/test values.
        device: "auto" resolves to CUDA, else MPS, else CPU.
        save_path: optional path to also dump the results dict as JSON.
        early_stop_id / early_stop_every: ID probe early-stopping controls.

    Returns:
        JSON-serializable dict: benchmark settings + per-task metrics
        (in-domain and OOD exact match per length and averaged).
    """
    device = resolve_device(device)
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
            "early_stop_id": early_stop_id,
            "early_stop_every": early_stop_every,
        },
        "tasks": {},
    }

    for name in tasks:
        results["tasks"][name] = run_one_task(
            make_model,
            name,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            train_min_len=train_min_len,
            train_max_len=train_max_len,
            id_lengths=id_lengths,
            ood_lengths=ood_lengths,
            n_eval=n_eval,
            train_seed=train_seed,
            eval_seed=eval_seed,
            device=device,
            eval_batch_size=eval_batch_size,
            early_stop_id=early_stop_id,
            early_stop_every=early_stop_every,
        )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"Saved results to {path}")

    return results
