"""DYCK -- stack thinking (Dyck-2 membership classification).

The model sees a bracket string over ``( ) [ ]`` and must output ``1`` if it is
well-formed (balanced with matching types) and ``0`` otherwise::

    ( [ ] ) ( ) = 1 <eos>
    ( [ ) ] = 0 <eos>

This is the classic balanced-brackets membership task used as a formal-language
benchmark for transformers (Bhattamishra et al., 2020). Deciding membership
requires simulating a stack over the whole string.

``length`` counts bracket *pairs*: strings have exactly ``2 * length`` tokens
so both classes are possible at every length (odd-length strings are trivially
unbalanced). Positives are uniform-ish random balanced Dyck-2 words; negatives
are balanced words corrupted at one random position (a single-bracket
replacement always breaks either a type count or a stack match, so the label is
guaranteed ``0`` -- and such near-miss negatives cannot be rejected by shallow
count heuristics alone). Labels are balanced 50/50 by construction.
"""

from __future__ import annotations

from src.tasks.base import Task
from src.tasks.vocab import CLOSERS, EQ, OPENERS

BRACKETS = OPENERS + CLOSERS


def is_balanced(symbols) -> bool:
    """Reference stack checker for Dyck-2 strings (used by tests too)."""
    stack = []
    for s in symbols:
        if s in OPENERS:
            stack.append(CLOSERS[OPENERS.index(s)])
        elif s in CLOSERS:
            if not stack or stack.pop() != s:
                return False
        else:  # pragma: no cover - guarded by the fixed vocabulary
            raise ValueError(f"not a bracket: {s!r}")
    return not stack


def sample_balanced(n_pairs: int, randint) -> list[str]:
    """Random balanced Dyck-2 word of ``2 * n_pairs`` tokens via a stack walk."""
    total = 2 * n_pairs
    out: list[str] = []
    stack: list[str] = []
    for i in range(total):
        remaining = total - i
        must_open = not stack
        must_close = len(stack) == remaining
        if must_open or (not must_close and int(randint(0, 2)) == 0):
            t = int(randint(0, len(OPENERS)))
            out.append(OPENERS[t])
            stack.append(CLOSERS[t])
        else:
            out.append(stack.pop())
    return out


def corrupt(word: list[str], randint) -> list[str]:
    """Replace one random bracket with a different one -> always unbalanced."""
    out = list(word)
    pos = int(randint(0, len(out)))
    choices = [b for b in BRACKETS if b != out[pos]]
    out[pos] = choices[int(randint(0, len(choices)))]
    return out


class DyckTask(Task):
    name = "dyck"

    @staticmethod
    def _sample(randint, n_pairs):
        word = sample_balanced(n_pairs, randint)
        if int(randint(0, 2)) == 0:
            return {"s": word, "label": "1"}
        return {"s": corrupt(word, randint), "label": "0"}

    def sample_train(self, randint, min_len, max_len):
        length = int(randint(min_len, max_len + 1))
        return self._sample(randint, length)

    def sample_eval(self, randint, length):
        return self._sample(randint, length)

    def prompt_symbols(self, instance):
        return list(instance["s"]) + [EQ]

    def answer_symbols(self, instance):
        return [instance["label"]]

    def max_new_tokens(self, length):
        return 3  # "0"/"1" + <eos> + slack
