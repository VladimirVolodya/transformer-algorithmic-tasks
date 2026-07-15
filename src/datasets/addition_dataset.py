"""On-the-fly reversed-digit addition dataset.

Two splits (plan §2):

* ``split="train"`` -- ``__getitem__`` ignores the index and samples a **fresh**
  example every call, with each operand's digit length uniform in
  ``[min_len, max_len]``. Randomness comes from the process / DataLoader-worker
  RNG (numpy legacy global, re-seeded per worker by
  :func:`src.utils.init_utils.set_worker_seed`), so the stream is fresh within a
  run yet reproducible across runs given ``(seed, num_workers)``.

* ``split="eval"`` -- ``__getitem__(idx)`` is **deterministic** via a local
  ``np.random.default_rng(seed + idx)``. With ``fixed_length`` set, both operands
  have exactly that many digits (the per-length OOD curve); otherwise lengths are
  drawn in ``[min_len, max_len]`` (the in-distribution val monitor). The result is
  independent of worker count, so every PE variant is scored on identical examples.
"""

from __future__ import annotations

import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.datasets.sampling import sample_operand
from src.datasets.tokenizer import Tokenizer

try:  # torch is always present at runtime; guard only helps static tooling
    import torch
except ImportError:  # pragma: no cover
    torch = None


class AdditionDataset(BaseDataset):
    def __init__(
        self,
        min_len: int = 1,
        max_len: int = 5,
        dataset_length: int = 100_000,
        split: str = "train",
        fixed_length: int | None = None,
        seed: int = 0,
        name: str = "train",
        *args,
        **kwargs,
    ):
        assert split in {"train", "eval"}, f"unknown split {split!r}"
        if split == "eval" and fixed_length is None:
            assert min_len <= max_len, "eval range requires min_len <= max_len"
        self.min_len = min_len
        self.max_len = max_len
        self.split = split
        self.fixed_length = fixed_length
        self.seed = seed
        self.name = name
        self.tokenizer = Tokenizer()

        # Synthetic index so __len__ / get_dataloaders work; content is generated
        # in __getitem__, not stored here.
        index = [{"id": i} for i in range(dataset_length)]
        super().__init__(index, *args, **kwargs)

    def _sample_pair(self, ind: int):
        if self.split == "train":
            # Legacy numpy global RNG -> re-seeded per worker by set_worker_seed.
            la = int(np.random.randint(self.min_len, self.max_len + 1))
            lb = int(np.random.randint(self.min_len, self.max_len + 1))
            randint = np.random.randint
            return sample_operand(la, randint), sample_operand(lb, randint)

        # eval: deterministic per index
        rng = np.random.default_rng(self.seed + ind)
        if self.fixed_length is not None:
            la = lb = self.fixed_length
        else:
            la = int(rng.integers(self.min_len, self.max_len + 1))
            lb = int(rng.integers(self.min_len, self.max_len + 1))
        return sample_operand(la, rng.integers), sample_operand(lb, rng.integers)

    def __getitem__(self, ind: int):
        a, b = self._sample_pair(ind)
        input_ids, loss_mask = self.tokenizer.make_example(a, b)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }

    @staticmethod
    def _assert_index_is_valid(index):
        # The base class expects "path"/"label" keys; our index is synthetic.
        assert len(index) > 0, "dataset_length must be positive"
