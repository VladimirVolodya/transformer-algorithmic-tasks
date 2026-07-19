"""SORT -- global comparisons.

The model sees ``length`` digits (uniform in 0-9, duplicates allowed) and must
output them in ascending order::

    3 7 2 = 2 3 7 <eos>

Every output position depends on the whole input (a global comparison /
counting problem), unlike addition's local carry chain.
"""

from __future__ import annotations

from src.tasks.base import Task
from src.tasks.vocab import EQ


class SortTask(Task):
    name = "sort"

    @staticmethod
    def _sample_array(randint, length):
        return [int(randint(0, 10)) for _ in range(length)]

    def sample_train(self, randint, min_len, max_len):
        length = int(randint(min_len, max_len + 1))
        return {"arr": self._sample_array(randint, length)}

    def sample_eval(self, randint, length):
        return {"arr": self._sample_array(randint, length)}

    def prompt_symbols(self, instance):
        return [str(d) for d in instance["arr"]] + [EQ]

    def answer_symbols(self, instance):
        return [str(d) for d in sorted(instance["arr"])]

    def max_new_tokens(self, length):
        return length + 2  # length digits + <eos> + slack
