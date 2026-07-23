"""PoSE + FixMatch semi-supervised training for length generalization.

Combines two ideas on top of the repo's standard recipe (AdamW + OneCycle,
answer-only masked CE):

* **PoSE** (Zhu et al., 2023 -- positional skip-wise training): training
  sequences keep their short length but their absolute-position indices are
  split into two chunks with random skips (``DecoderTransformer.pe_variant=
  "pose"``), so the whole ``[0, max_len)`` embedding table and long relative
  offsets receive gradient.
* **FixMatch** (Sohn et al., 2020): a stream of *unlabeled* OOD-length prompts
  (ground-truth answers are never used). The **weak view** -- plain ``0..T-1``
  positions -- is greedily decoded to produce a pseudo-label with per-token
  confidence; a sequence is accepted iff it emitted ``<eos>`` and the minimum
  confidence over generated tokens is ``>= tau``. The model is then trained to
  reproduce the pseudo-label under the **strong view** -- freshly sampled PoSE
  skip-wise positions -- with the same masked CE, weighted by ``lambda_u``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from src.loss.masked_ce import MaskedCrossEntropyLoss
from src.tasks.data import make_train_batch


def sample_unlabeled_prompts(task, tokenizer, rng, batch_size, min_len, max_len):
    """Sample a batch of same-length OOD prompts (answers are discarded).

    One length per batch so the prompts stack into a rectangular tensor.
    """
    length = int(rng.randint(min_len, max_len + 1))
    prompts = [
        tokenizer.encode(task.prompt_symbols(task.sample_eval(rng.randint, length)))
        for _ in range(batch_size)
    ]
    return torch.tensor(prompts, dtype=torch.long), length


@torch.no_grad()
def pseudo_label_batch(
    model, prompt_ids, max_new_tokens, eos_id, pad_id, tau,
    min_ans_len=None, max_ans_len=None,
):
    """Greedy-decode prompts on the weak view and keep confident sequences.

    Returns ``(input_ids, loss_mask, n_accepted)`` -- a padded batch of
    ``prompt + pseudo-answer + <eos>`` rows for the sequences whose every
    generated token had confidence ``>= tau`` (and that finished with
    ``<eos>``), or ``(None, None, 0)`` if none qualified.

    ``min_ans_len``/``max_ans_len`` optionally reject structurally impossible
    pseudo-answers (token count before ``<eos>`` outside the bounds). This
    blocks degenerate attractors such as the confident empty answer.
    """
    model.eval()
    B, P = prompt_ids.shape
    device = prompt_ids.device
    seq = prompt_ids
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    min_conf = torch.ones(B, device=device)
    for _ in range(max_new_tokens):
        logits = model(input_ids=seq)["logits"][:, -1, :]
        probs = F.softmax(logits, dim=-1)
        conf, tok = probs.max(dim=-1)
        tok = torch.where(finished, torch.full_like(tok, eos_id), tok)
        conf = torch.where(finished, torch.ones_like(conf), conf)
        min_conf = torch.minimum(min_conf, conf)
        seq = torch.cat([seq, tok[:, None]], dim=1)
        finished = finished | (tok == eos_id)
        if bool(finished.all()):
            break
    model.train()

    accept = finished & (min_conf >= tau)
    gen_all = seq[:, P:]
    # Answer length = tokens before the first <eos> (argmax is only valid on
    # finished rows, which is all `accept` candidates).
    ans_len = (gen_all == eos_id).float().argmax(dim=1)
    if min_ans_len is not None:
        accept &= ans_len >= min_ans_len
    if max_ans_len is not None:
        accept &= ans_len <= max_ans_len
    n_accepted = int(accept.sum())
    if n_accepted == 0:
        return None, None, 0

    seq = seq[accept]
    gen = seq[:, P:]
    # Mask covers the pseudo-answer up to and including its first <eos>;
    # the fill tokens emitted after <eos> become padding.
    first_eos = (gen == eos_id).float().argmax(dim=1)
    idx = torch.arange(gen.size(1), device=device)[None]
    answer_span = idx <= first_eos[:, None]
    gen = torch.where(answer_span, gen, torch.full_like(gen, pad_id))
    input_ids = torch.cat([seq[:, :P], gen], dim=1)
    loss_mask = torch.cat(
        [
            torch.zeros(input_ids.size(0), P, dtype=torch.long, device=device),
            answer_span.long(),
        ],
        dim=1,
    )
    return input_ids, loss_mask, n_accepted


def train_pose_fixmatch(
    model,
    task,
    tokenizer,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.01,
    min_len: int,
    max_len: int,
    unlabeled_batch_size: int = 128,
    unlabeled_min_len: int = 6,
    unlabeled_max_len: int = 15,
    unlabeled_every: int = 1,
    tau: float = 0.95,
    lambda_u: float = 1.0,
    answer_len_bounds=None,
    curriculum: bool = False,
    advance_threshold: float = 0.7,
    accept_ema_beta: float = 0.98,
    eval_lengths=(),
    eval_every: int = 500,
    eval_n: int = 100,
    eval_seed: int = 0,
    seed: int,
    device: str,
    max_grad_norm: float = 1.0,
    log_every: int = 250,
):
    """FixMatch on top of the standard supervised recipe.

    The supervised stream is identical to :func:`src.benchmark.train_on_task`
    (same RNG contract), the model's PoSE positions act as the strong
    augmentation for both streams.

    ``answer_len_bounds``: optional ``length -> (min_ans, max_ans)`` structural
    filter on pseudo-answers (e.g. addition: ``lambda L: (L, L + 1)``).

    ``curriculum=True`` enables *frontier self-training*: unlabeled lengths are
    sampled from ``[unlabeled_min_len, frontier]`` where the frontier starts at
    ``unlabeled_min_len`` and advances by one when the EMA of the acceptance
    rate at the frontier exceeds ``advance_threshold`` (capped at
    ``unlabeled_max_len``).

    ``eval_lengths``: if non-empty, autoregressive exact match is measured on
    those lengths every ``eval_every`` steps and recorded in the returned
    history.

    Returns a dict with ``final_loss``, ``accept_rate`` (last window),
    ``history`` (mid-training eval snapshots) and ``final_frontier``.
    """
    from src.tasks.evaluate import sweep_lengths  # local import: avoid cycle
    rng = np.random.RandomState(seed)
    criterion = MaskedCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=steps,
        pct_start=0.05,
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    model.train()
    losses, accept_rates, history = [], [], []
    frontier = unlabeled_min_len if curriculum else unlabeled_max_len
    frontier_ema = 0.0
    for step in range(1, steps + 1):
        sup_batch = make_train_batch(task, tokenizer, rng.randint, batch_size, min_len, max_len)
        sup_batch = {k: v.to(device) for k, v in sup_batch.items()}

        optimizer.zero_grad()
        outputs = model(**sup_batch)
        loss = criterion(**outputs, **sup_batch)["loss"]

        if step % unlabeled_every == 0:
            prompts, length = sample_unlabeled_prompts(
                task, tokenizer, rng, unlabeled_batch_size,
                unlabeled_min_len, frontier,
            )
            bounds = answer_len_bounds(length) if answer_len_bounds else (None, None)
            u_ids, u_mask, n_acc = pseudo_label_batch(
                model,
                prompts.to(device),
                task.max_new_tokens(length),
                tokenizer.eos_id,
                tokenizer.pad_id,
                tau,
                min_ans_len=bounds[0],
                max_ans_len=bounds[1],
            )
            accept_rates.append(n_acc / unlabeled_batch_size)
            if n_acc > 0:
                positions = model.sample_pose_positions(*u_ids.shape, device)
                u_out = model(input_ids=u_ids, positions=positions)
                u_loss = criterion(u_out["logits"], u_ids, u_mask)["loss"]
                loss = loss + lambda_u * u_loss
            if curriculum and length == frontier:
                frontier_ema = (
                    accept_ema_beta * frontier_ema
                    + (1 - accept_ema_beta) * n_acc / unlabeled_batch_size
                )
                if frontier_ema >= advance_threshold and frontier < unlabeled_max_len:
                    frontier += 1
                    frontier_ema = 0.0
                    print(f"    [{task.name}/fixmatch] step {step}: "
                          f"frontier advanced to {frontier}")

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if eval_lengths and (step % eval_every == 0 or step == steps):
            acc = sweep_lengths(
                model, task, list(eval_lengths), n=eval_n, seed=eval_seed,
                device=device, tokenizer=tokenizer,
            )
            model.train()
            history.append(
                {"step": step, "frontier": frontier,
                 "acc": {str(k): v for k, v in acc.items()}}
            )
            print(f"    [{task.name}/fixmatch] step {step} OOD acc: "
                  + " ".join(f"L{k}={v:.2f}" for k, v in acc.items()))
        if step % log_every == 0 or step == steps:
            window = max(1, log_every // unlabeled_every)
            print(
                f"    [{task.name}/fixmatch] step {step}/{steps} "
                f"loss={np.mean(losses[-log_every:]):.4f} "
                f"accept={np.mean(accept_rates[-window:]):.2f} "
                f"frontier={frontier} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
    window = max(1, log_every // unlabeled_every)
    return {
        "final_loss": float(np.mean(losses[-log_every:])),
        "accept_rate": float(np.mean(accept_rates[-window:])),
        "history": history,
        "final_frontier": frontier,
    }
