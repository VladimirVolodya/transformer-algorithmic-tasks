"""Operand sampling for ADDITION (shared by train and eval).

Keeping a single sampler here guarantees the deterministic eval sets used by
the benchmark are generated identically for every architecture.

Digit-by-digit sampling is used for ``length > 1`` so OOD lengths (20–30) do
not overflow numpy's int64 when drawing ``10 ** length``.
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
    # Avoid ``10 ** length`` (overflows int64 for length >= 19).
    digits = [str(int(randint(1, 10)))]
    for _ in range(length - 1):
        digits.append(str(int(randint(0, 10))))
    return int("".join(digits))
