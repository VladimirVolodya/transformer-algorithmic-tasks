"""Autoregressive greedy exact-match evaluation, generic over tasks.

Decode greedily from the ``=`` prompt and count a sequence correct iff every
answer symbol **and** the ``<eos>`` position are right. Fixed-length eval sets
have identical prompt lengths within a task, so a whole length is decoded as
one batch (asserted below).
"""

from __future__ import annotations

import torch

from src.tasks.base import Task
from src.tasks.data import make_eval_set
from src.tasks.vocab import TaskTokenizer


@torch.no_grad()
def exact_match_at_length(
    model,
    task: Task,
    length: int,
    n: int,
    seed: int = 0,
    device: str = "cpu",
    tokenizer: TaskTokenizer | None = None,
    batch_size: int = 256,
    return_examples: int = 0,
):
    """Greedy-decode ``n`` fixed-length instances of ``task`` and return
    ``(accuracy, examples)`` where ``accuracy`` is the exact-match fraction."""
    tokenizer = tokenizer or TaskTokenizer()
    model.eval()

    instances = make_eval_set(task, length, n, seed)
    prompts = [tokenizer.encode(task.prompt_symbols(inst)) for inst in instances]
    targets = [task.answer_symbols(inst) for inst in instances]
    assert len({len(p) for p in prompts}) == 1, (
        f"{task.name}: fixed-length prompts must share one token length"
    )
    max_new = task.max_new_tokens(length)

    n_correct = 0
    examples = []
    for start in range(0, n, batch_size):
        batch_prompts = prompts[start : start + batch_size]
        prompt_tensor = torch.tensor(batch_prompts, dtype=torch.long, device=device)
        generated = model.generate(
            prompt_tensor, max_new_tokens=max_new, eos_id=tokenizer.eos_id
        ).cpu().tolist()

        for j, gen_ids in enumerate(generated):
            pred = tokenizer.decode(gen_ids, stop_at_eos=True, strip_special=True)
            target = targets[start + j]
            correct = pred == target
            n_correct += int(correct)
            if len(examples) < return_examples:
                examples.append(
                    {
                        "prompt": "".join(
                            tokenizer.decode(
                                batch_prompts[j], stop_at_eos=False, strip_special=True
                            )
                        ),
                        "pred": "".join(pred),
                        "target": "".join(target),
                        "correct": correct,
                    }
                )

    return n_correct / n, examples


def sweep_lengths(
    model,
    task: Task,
    lengths,
    n: int,
    seed: int = 0,
    device: str = "cpu",
    tokenizer: TaskTokenizer | None = None,
    batch_size: int = 256,
):
    """Return ``{length: exact_match_accuracy}`` over the given lengths."""
    tokenizer = tokenizer or TaskTokenizer()
    return {
        length: exact_match_at_length(
            model, task, length, n, seed, device, tokenizer, batch_size
        )[0]
        for length in lengths
    }
