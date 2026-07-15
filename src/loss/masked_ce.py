"""Next-token cross-entropy computed **only on answer tokens** (plan §2).

``logits[:, :-1]`` predict ``input_ids[:, 1:]``; the loss is averaged over the
positions where the shifted ``loss_mask`` is 1 (the digits of ``rev(s)`` and
``<eos>``). Everything before ``=`` and all padding is ignored. Getting this
mask wrong silently ruins training, so it is asserted in the tests.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MaskedCrossEntropyLoss(nn.Module):
    def forward(self, logits, input_ids, loss_mask, **batch):
        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:].to(shift_logits.dtype)

        vocab = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_targets.reshape(-1),
            reduction="none",
        ).view(shift_targets.shape)

        denom = shift_mask.sum().clamp_min(1.0)
        loss = (token_loss * shift_mask).sum() / denom
        return {"loss": loss}
