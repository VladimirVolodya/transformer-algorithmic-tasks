"""Task interface for the multi-task algorithmic benchmark.

A task defines how to sample instances at a given *length* (the task's single
difficulty knob) and how to render an instance into prompt / answer symbols
over the shared :mod:`src.tasks.vocab` vocabulary.

Two sampling entry points mirror the addition dataset's split semantics:

* :meth:`Task.sample_train` -- fresh training instance; lengths are drawn
  inside the method (addition, for example, draws the two operand lengths
  independently). Takes a ``randint(low, high)`` callable (numpy convention,
  high exclusive) so both the legacy global RNG and ``Generator.integers``
  work.
* :meth:`Task.sample_eval` -- deterministic fixed-length instance given a
  seeded ``randint``. The benchmark seeds one RNG per example index, so every
  architecture is trained and evaluated on identical data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Task(ABC):
    name: str

    @abstractmethod
    def sample_train(self, randint, min_len: int, max_len: int) -> dict:
        """Fresh training instance with length(s) uniform in [min_len, max_len]."""

    @abstractmethod
    def sample_eval(self, randint, length: int) -> dict:
        """Instance of exactly ``length``; deterministic given ``randint``."""

    @abstractmethod
    def prompt_symbols(self, instance: dict) -> list[str]:
        """Prompt symbols, ending with ``=`` (the generation seed)."""

    @abstractmethod
    def answer_symbols(self, instance: dict) -> list[str]:
        """Answer symbols the model must generate (``<eos>`` excluded)."""

    @abstractmethod
    def max_new_tokens(self, length: int) -> int:
        """Decoding cap at ``length`` (answer + ``<eos>`` + a little slack)."""

    def render(self, instance: dict) -> str:
        """Human-readable one-liner for logs and notebooks."""
        return (
            "".join(self.prompt_symbols(instance))
            + "".join(self.answer_symbols(instance))
        )
