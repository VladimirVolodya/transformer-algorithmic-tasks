"""ADDITION -- carry propagation (the original validation task).

Reversed-digit addition: ``rev(a) + rev(b) = rev(a+b)``. Length is the operand
digit count; training draws the two operand lengths independently, eval uses
the same length for both (benchmark eval convention).
"""

from __future__ import annotations

from src.tasks.base import Task
from src.tasks.sampling import sample_operand
from src.tasks.vocab import EQ, PLUS


def _rev_digits(n: int) -> list[str]:
    """Digit symbols of a non-negative int, least-significant digit first."""
    return list(str(n))[::-1]


class AdditionTask(Task):
    name = "addition"

    def sample_train(self, randint, min_len, max_len):
        la = int(randint(min_len, max_len + 1))
        lb = int(randint(min_len, max_len + 1))
        return {"a": sample_operand(la, randint), "b": sample_operand(lb, randint)}

    def sample_eval(self, randint, length):
        return {
            "a": sample_operand(length, randint),
            "b": sample_operand(length, randint),
        }

    def prompt_symbols(self, instance):
        return _rev_digits(instance["a"]) + [PLUS] + _rev_digits(instance["b"]) + [EQ]

    def answer_symbols(self, instance):
        return _rev_digits(instance["a"] + instance["b"])

    def max_new_tokens(self, length):
        # rev(s) has <= length + 1 digits; + <eos> + a little slack.
        return length + 3
