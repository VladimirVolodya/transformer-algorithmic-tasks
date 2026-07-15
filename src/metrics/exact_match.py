"""Teacher-forced full-answer exact match (the in-distribution training monitor).

A sequence counts as correct iff the argmax next-token prediction matches the
target at **every** answer position (``loss_mask == 1``), using the same shift as
the loss. This is cheap and drives ``monitor: "max val_ExactMatch"``.

Note: this is *teacher-forced* (the model sees ground-truth previous tokens).
The authoritative length-generalization metric uses free-running greedy
decoding -- see :mod:`src.evaluation.length_generalization`.
"""

from src.metrics.base_metric import BaseMetric


class ExactMatch(BaseMetric):
    def __call__(self, logits, input_ids, loss_mask, **batch):
        preds = logits[:, :-1, :].argmax(dim=-1)
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:].bool()
        # Non-answer positions are treated as always correct so only the answer
        # span decides exact match.
        correct = (preds == targets) | ~mask
        seq_correct = correct.all(dim=1)
        return seq_correct.float().mean().item()
