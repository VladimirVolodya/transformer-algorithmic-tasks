"""Batch construction for the benchmark: fresh train batches + frozen eval sets.

Train batches are sampled from a caller-owned RNG (one ``np.random.RandomState``
per task run), so the training stream depends only on ``(task, seed)`` -- every
architecture trains on the identical sequence of examples.

Eval instances are generated with a per-index RNG (``default_rng(seed + idx)``),
the same scheme the legacy addition eval uses: deterministic, independent of
batch size, identical across architectures.
"""

from __future__ import annotations

import numpy as np
import torch

from src.tasks.base import Task
from src.tasks.vocab import TaskTokenizer


def pad_batch(examples, pad_id: int):
    """Right-pad ``(input_ids, loss_mask)`` lists into a tensor batch dict."""
    max_len = max(len(ids) for ids, _ in examples)
    batch = len(examples)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, (ids, mask) in enumerate(examples):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        loss_mask[i, : len(mask)] = torch.tensor(mask, dtype=torch.long)
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def make_train_batch(
    task: Task,
    tokenizer: TaskTokenizer,
    randint,
    batch_size: int,
    min_len: int,
    max_len: int,
):
    """Sample a fresh padded training batch from ``randint``."""
    examples = []
    for _ in range(batch_size):
        inst = task.sample_train(randint, min_len, max_len)
        examples.append(
            tokenizer.make_example(task.prompt_symbols(inst), task.answer_symbols(inst))
        )
    return pad_batch(examples, tokenizer.pad_id)


def make_eval_set(task: Task, length: int, n: int, seed: int = 0):
    """Deterministic list of ``n`` fixed-length instances (same across archs)."""
    return [
        task.sample_eval(np.random.default_rng(seed + idx).integers, length)
        for idx in range(n)
    ]
