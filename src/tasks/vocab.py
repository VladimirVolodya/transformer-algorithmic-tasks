"""Unified vocabulary and tokenizer for all benchmark tasks.

One shared vocabulary covers ADDITION, SORT, DYCK and INDEX so the model
architecture (embedding/head sizes included) is literally identical across
tasks. Digit ``d`` maps to id ``d``; ``pad_id == 13``. Bracket and comma
tokens are appended after the core 14 symbols.

Every task's sequence has the shape::

    <prompt symbols> ... = <answer symbols> <eos>

and the loss mask is 1 exactly on the answer symbols and ``<eos>``.
"""

from __future__ import annotations

DIGITS = [str(d) for d in range(10)]
PLUS = "+"
EQ = "="
EOS = "<eos>"
PAD = "<pad>"
COMMA = ","
OPENERS = ["(", "["]
CLOSERS = [")", "]"]

# Fixed order: digit d -> id d, pad_id 13, then brackets and comma.
VOCAB = DIGITS + [PLUS, EQ, EOS, PAD] + OPENERS + CLOSERS + [COMMA]
VOCAB_SIZE = len(VOCAB)  # 19


class TaskTokenizer:
    """Fixed-vocabulary tokenizer shared by all benchmark tasks."""

    def __init__(self) -> None:
        self.itos = list(VOCAB)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.eos_id = self.stoi[EOS]
        self.eq_id = self.stoi[EQ]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, symbols) -> list[int]:
        return [self.stoi[s] for s in symbols]

    def decode(self, ids, stop_at_eos: bool = True, strip_special: bool = True) -> list[str]:
        """Ids -> symbols. Stops at the first ``<eos>`` (exclusive) by default
        and drops special tokens, so the result is the answer symbol string."""
        out = []
        for i in ids:
            s = self.itos[int(i)]
            if stop_at_eos and s == EOS:
                break
            if strip_special and s in (PAD, EOS):
                continue
            out.append(s)
        return out

    def make_example(self, prompt_symbols, answer_symbols):
        """Return ``(input_ids, loss_mask)`` for ``prompt answer <eos>``.

        ``loss_mask[i] == 1`` iff token ``i`` belongs to the answer span
        (answer symbols and the trailing ``<eos>``).
        """
        symbols = list(prompt_symbols) + list(answer_symbols) + [EOS]
        input_ids = self.encode(symbols)
        n_prompt = len(prompt_symbols)
        loss_mask = [0] * n_prompt + [1] * (len(symbols) - n_prompt)
        return input_ids, loss_mask
