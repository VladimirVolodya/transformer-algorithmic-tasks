"""Operand sampling shared by the dataset and the evaluation module.

Keeping a single sampler here guarantees the deterministic eval sets used by
the training monitor and by the length-generalization curve are generated
identically.
"""

from __future__ import annotations


def sample_operand(length: int, randint) -> int:
    """Sample a ``length``-digit non-negative integer.

    Args:
        length: number of digits.
        randint: a ``randint(low, high)`` callable returning an int in
            ``[low, high)`` (numpy convention: ``np.random.randint`` or
            ``np.random.Generator.integers``).

    A length ``> 1`` operand has a non-zero leading digit so the length is
    genuine; length ``1`` allows ``0``.
    """
    if length <= 1:
        return int(randint(0, 10))
    return int(randint(10 ** (length - 1), 10 ** length))
