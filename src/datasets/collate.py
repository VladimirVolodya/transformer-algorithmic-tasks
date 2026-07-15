"""Collate for the addition task: right-pad variable-length sequences.

``input_ids`` are padded with ``<pad>`` and ``loss_mask`` with ``0`` to the
batch's max length. Because padding is strictly trailing and attention is
causal, real tokens never attend to pad, so no key-padding mask is needed; the
loss/metric simply ignore pad positions via ``loss_mask``.
"""

import torch

from src.datasets.tokenizer import Tokenizer

_PAD_ID = Tokenizer().pad_id


def collate_fn(dataset_items: list[dict]):
    """Convert a list of ``{"input_ids", "loss_mask"}`` items into a batch."""
    input_ids = [item["input_ids"] for item in dataset_items]
    loss_mask = [item["loss_mask"] for item in dataset_items]

    lengths = torch.tensor([x.size(0) for x in input_ids], dtype=torch.long)
    batch_size = len(input_ids)
    max_len = int(lengths.max())

    padded_ids = torch.full((batch_size, max_len), _PAD_ID, dtype=torch.long)
    padded_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids, loss_mask)):
        length = ids.size(0)
        padded_ids[i, :length] = ids
        padded_mask[i, :length] = mask

    return {
        "input_ids": padded_ids,
        "loss_mask": padded_mask,
        "lengths": lengths,
    }
