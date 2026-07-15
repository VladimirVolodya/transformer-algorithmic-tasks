"""Character-level tokenizer for the reversed-digit addition task.

Vocabulary (14 tokens): digits ``0-9``, ``+``, ``=``, ``<eos>``, ``<pad>``.

Sequences use the **reversed-digit** format::

    rev(a) + rev(b) = rev(s) <eos>

so that the carry propagates in the autoregressive (left-to-right) generation
direction. Example: ``a=58392, b=74918, s=133310`` ->
``2 9 3 8 5 + 8 1 9 4 7 = 0 1 3 3 3 1 <eos>``.
"""

from __future__ import annotations

DIGITS = [str(d) for d in range(10)]
PLUS = "+"
EQ = "="
EOS = "<eos>"
PAD = "<pad>"

# Fixed vocabulary order. Do NOT reorder: digit ``d`` maps to id ``d``.
VOCAB = DIGITS + [PLUS, EQ, EOS, PAD]
VOCAB_SIZE = len(VOCAB)  # 14


class Tokenizer:
    """Fixed-vocabulary tokenizer with reversed-digit example construction."""

    def __init__(self) -> None:
        self.itos = list(VOCAB)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.eos_id = self.stoi[EOS]
        self.plus_id = self.stoi[PLUS]
        self.eq_id = self.stoi[EQ]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, symbols) -> list[int]:
        return [self.stoi[s] for s in symbols]

    def decode(self, ids, stop_at_eos: bool = True, strip_special: bool = True) -> list[str]:
        """Ids -> symbols. Stops at the first ``<eos>`` (exclusive) by default
        and drops special tokens, so the result is the answer digit string."""
        out = []
        for i in ids:
            s = self.itos[int(i)]
            if stop_at_eos and s == EOS:
                break
            if strip_special and s in (PAD, EOS):
                continue
            out.append(s)
        return out

    # --- reversed-digit example construction -------------------------------

    @staticmethod
    def _rev_digits(n: int) -> list[str]:
        """Digit symbols of a non-negative int, least-significant digit first."""
        return list(str(n))[::-1]

    def example_symbols(self, a: int, b: int):
        """Symbols for ``rev(a) + rev(b) = rev(s) <eos>`` plus the index at
        which the answer (``rev(s) ... <eos>``) begins."""
        prompt = self._rev_digits(a) + [PLUS] + self._rev_digits(b) + [EQ]
        answer = self._rev_digits(a + b) + [EOS]
        return prompt + answer, len(prompt)

    def make_example(self, a: int, b: int):
        """Return ``(input_ids, loss_mask)`` for one example.

        ``loss_mask[i] == 1`` iff token ``i`` is an answer token (a digit of
        ``rev(s)`` or ``<eos>``); ``0`` for the prompt tokens ``rev(a)``,
        ``+``, ``rev(b)``, ``=``.
        """
        symbols, answer_start = self.example_symbols(a, b)
        input_ids = self.encode(symbols)
        loss_mask = [0] * answer_start + [1] * (len(symbols) - answer_start)
        return input_ids, loss_mask

    def make_prompt(self, a: int, b: int) -> list[int]:
        """Prompt ids up to and including ``=`` (the generation seed)."""
        return self.encode(self._rev_digits(a) + [PLUS] + self._rev_digits(b) + [EQ])

    def target_answer_symbols(self, a: int, b: int) -> list[str]:
        """Symbols the model must generate: ``rev(a + b)`` then ``<eos>``."""
        return self._rev_digits(a + b) + [EOS]

    def print_vocab(self) -> None:
        mapping = ", ".join(f"{i}:{s!r}" for i, s in enumerate(self.itos))
        print(f"Vocab (size {self.vocab_size}): {mapping}")
