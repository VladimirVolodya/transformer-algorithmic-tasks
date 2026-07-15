"""Autoregressive-greedy evaluator for the addition task.

Unlike the template's teacher-forced inferencer, this decodes answers
token-by-token (the authoritative length-generalization metric) via
:mod:`src.evaluation.length_generalization`. It is used by ``inference.py`` and
can be reused from the analysis notebook.
"""

import torch

from src.datasets.tokenizer import Tokenizer
from src.evaluation.length_generalization import exact_match_at_length, sweep_lengths


class Inferencer:
    def __init__(
        self,
        model,
        device,
        tokenizer=None,
        from_pretrained=None,
        seed=0,
        batch_size=256,
    ):
        """
        Args:
            model (nn.Module): the transformer to evaluate.
            device (str): device for tensors and model.
            tokenizer (Tokenizer | None): shared tokenizer (built if None).
            from_pretrained (str | None): checkpoint path to load weights from.
            seed (int): seed for the deterministic eval sets.
            batch_size (int): decoding batch size.
        """
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer or Tokenizer()
        self.seed = seed
        self.batch_size = batch_size
        if from_pretrained is not None:
            self._from_pretrained(from_pretrained)

    def _from_pretrained(self, path):
        """Load model weights from a checkpoint (``state_dict`` or raw)."""
        print(f"Loading model weights from: {path} ...")
        checkpoint = torch.load(str(path), map_location=self.device)
        state = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state)

    def exact_match(self, length, n, return_examples=0):
        """Exact-match accuracy at a single operand length (+ optional examples)."""
        return exact_match_at_length(
            self.model,
            length,
            n,
            seed=self.seed,
            device=self.device,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            return_examples=return_examples,
        )

    def sweep(self, lengths, n):
        """Exact-match accuracy across a list of lengths -> ``{length: acc}``."""
        return sweep_lengths(
            self.model,
            lengths,
            n,
            seed=self.seed,
            device=self.device,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
        )
