"""Authoritative length-generalization evaluation via greedy autoregressive decoding.

Full-sequence exact match (plan §5): every answer digit **and** the ``<eos>`` must
be correct. Decoding is greedy from the ``=`` prompt until ``<eos>`` or a length
cap. The eval pairs are generated deterministically (same sampler as
``AdditionDataset``'s eval split), so every PE variant is scored on identical
examples.

Because a fixed-length eval set has identical prompt lengths, a whole length is
decoded as one batch.
"""

from __future__ import annotations

import numpy as np
import torch

from src.datasets.sampling import sample_operand
from src.datasets.tokenizer import Tokenizer


def make_eval_pairs(length: int, n: int, seed: int = 0):
    """Deterministic list of ``(a, b)`` pairs, both operands exactly ``length``
    digits. Matches ``AdditionDataset(split="eval", fixed_length=length)``."""
    pairs = []
    for idx in range(n):
        rng = np.random.default_rng(seed + idx)
        a = sample_operand(length, rng.integers)
        b = sample_operand(length, rng.integers)
        pairs.append((a, b))
    return pairs


@torch.no_grad()
def exact_match_at_length(
    model,
    length: int,
    n: int,
    seed: int = 0,
    device: str = "cpu",
    tokenizer: Tokenizer | None = None,
    batch_size: int = 256,
    return_examples: int = 0,
):
    """Greedy-decode ``n`` fixed-length examples and return
    ``(accuracy, examples)`` where ``accuracy`` is the exact-match fraction."""
    tokenizer = tokenizer or Tokenizer()
    model.eval()

    pairs = make_eval_pairs(length, n, seed)
    prompts = [tokenizer.make_prompt(a, b) for a, b in pairs]
    targets = [
        [s for s in tokenizer.target_answer_symbols(a, b) if s != "<eos>"]
        for a, b in pairs
    ]
    # rev(s) has <= length + 1 digits; + <eos> + a little slack.
    max_new = length + 3

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
                a, b = pairs[start + j]
                examples.append(
                    {
                        "a": a,
                        "b": b,
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
    lengths,
    n: int,
    seed: int = 0,
    device: str = "cpu",
    tokenizer: Tokenizer | None = None,
    batch_size: int = 256,
):
    """Return ``{length: exact_match_accuracy}`` over the given lengths."""
    tokenizer = tokenizer or Tokenizer()
    return {
        length: exact_match_at_length(
            model, length, n, seed, device, tokenizer, batch_size
        )[0]
        for length in lengths
    }
