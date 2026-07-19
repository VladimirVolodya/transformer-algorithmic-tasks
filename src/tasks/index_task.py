"""INDEX -- positional access.

The model sees ``length`` digits, a comma, and a 0-based index, and must output
the digit at that index::

    3 7 2 , 1 = 7 <eos>

The index is always a single digit -- uniform in ``[0, min(length, 10))`` -- so
OOD arrays (length > 10) are still queried at positions 0-9. This keeps the
index *representation* in-distribution and isolates what we care about:
positional access inside arrays longer than any seen in training. (Otherwise a
two-digit index like ``12`` would be an unseen token pattern and would confound
the length-generalization measurement.)
"""

from __future__ import annotations

from src.tasks.base import Task
from src.tasks.vocab import COMMA, EQ


class IndexTask(Task):
    name = "index"

    @staticmethod
    def _sample(randint, length):
        arr = [int(randint(0, 10)) for _ in range(length)]
        idx = int(randint(0, min(length, 10)))
        return {"arr": arr, "idx": idx}

    def sample_train(self, randint, min_len, max_len):
        length = int(randint(min_len, max_len + 1))
        return self._sample(randint, length)

    def sample_eval(self, randint, length):
        return self._sample(randint, length)

    def prompt_symbols(self, instance):
        return (
            [str(d) for d in instance["arr"]]
            + [COMMA, str(instance["idx"]), EQ]
        )

    def answer_symbols(self, instance):
        return [str(instance["arr"][instance["idx"]])]

    def max_new_tokens(self, length):
        return 3  # one digit + <eos> + slack
