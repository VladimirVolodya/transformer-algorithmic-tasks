"""Round-based self-improvement (Lee et al. 2025, arXiv:2502.01612) on PoSE.

The FixMatch failure analysis showed two label-poisoning mechanisms: (1) the
live model drifting while it labels its own stream, and (2) confidence being
uncorrelated with correctness under length shift. This module applies the
fixes validated by the self-improving-transformers recipe:

* **Frozen teacher per round** -- pseudo-data is generated once per round by a
  frozen snapshot of the model, then the student trains on it; the label
  source cannot drift mid-round.
* **Majority voting across position views** -- an answer becomes a label only
  if at least ``vote_min`` of ``n_votes`` greedy decodes under *different*
  position assignments (plain + PoSE resamplings) agree exactly. This replaces
  FixMatch's confidence threshold, which is miscalibrated OOD.
* **Length filtering** -- structurally impossible answer lengths are rejected
  (for addition: a sum of two L-digit numbers has L or L+1 digits).
* **+1 length per round** -- the frontier advances on a fixed schedule; data
  quality is maintained by the filters, and pseudo-data accumulates across
  rounds.

Supervised training uses the repo's standard recipe; pseudo-batches get
freshly sampled PoSE positions (the strong view), exactly like the FixMatch
student branch.
"""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from src.loss.masked_ce import MaskedCrossEntropyLoss
from src.tasks.data import make_train_batch, pad_batch


@torch.no_grad()
def _greedy_decode(model, prompt_ids, max_new, eos_id, positions=None):
    """Greedy decode; ``positions`` (``[B, P+max_new]``) fixes the position
    assignment for the whole decode (sliced per step). Returns ``[B, steps]``
    generated tokens (eos-padded) and a finished mask."""
    B = prompt_ids.size(0)
    device = prompt_ids.device
    seq = prompt_ids
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    gen = []
    for _ in range(max_new):
        pos = positions[:, : seq.size(1)] if positions is not None else None
        logits = model(input_ids=seq, positions=pos)["logits"][:, -1, :]
        tok = logits.argmax(dim=-1)
        tok = torch.where(finished, torch.full_like(tok, eos_id), tok)
        gen.append(tok)
        seq = torch.cat([seq, tok[:, None]], dim=1)
        finished = finished | (tok == eos_id)
        if bool(finished.all()):
            break
    return torch.stack(gen, dim=1), finished


@torch.no_grad()
def make_pseudo_dataset(
    teacher,
    task,
    tokenizer,
    rng,
    n_prompts: int,
    length: int,
    *,
    n_votes: int = 5,
    vote_min: int = 3,
    answer_len_bounds=None,
    device: str = "cpu",
    batch_size: int = 128,
):
    """Label ``n_prompts`` frontier-length prompts with the frozen ``teacher``.

    Vote over ``n_votes`` greedy decodes: one with plain positions and the
    rest under independent PoSE position resamplings. Keep a prompt iff the
    modal answer appears ``>= vote_min`` times, ends with ``<eos>``, and
    passes the length filter. Returns ``(examples, n_kept)`` where examples
    are ``(input_ids, loss_mask)`` lists compatible with ``pad_batch``.
    """
    teacher.eval()
    eos, pad = tokenizer.eos_id, tokenizer.pad_id
    bounds = answer_len_bounds(length) if answer_len_bounds else (None, None)
    max_new = task.max_new_tokens(length)
    examples = []
    for start in range(0, n_prompts, batch_size):
        b = min(batch_size, n_prompts - start)
        prompts = torch.tensor(
            [
                tokenizer.encode(
                    task.prompt_symbols(task.sample_eval(rng.randint, length))
                )
                for _ in range(b)
            ],
            dtype=torch.long,
            device=device,
        )
        P = prompts.size(1)
        answers = [[] for _ in range(b)]
        for v in range(n_votes):
            positions = (
                None
                if v == 0
                else teacher.sample_pose_positions(b, P + max_new, device)
            )
            gen, finished = _greedy_decode(teacher, prompts, max_new, eos, positions)
            for i in range(b):
                if not bool(finished[i]):
                    continue
                row = gen[i].tolist()
                ans = tuple(row[: row.index(eos)])
                answers[i].append(ans)
        for i in range(b):
            if not answers[i]:
                continue
            (ans_tuple, n_agree), = Counter(answers[i]).most_common(1)
            if n_agree < vote_min:
                continue
            ans = list(ans_tuple)
            if bounds[0] is not None and len(ans) < bounds[0]:
                continue
            if bounds[1] is not None and len(ans) > bounds[1]:
                continue
            ids = prompts[i].tolist() + ans + [eos]
            mask = [0] * P + [1] * (len(ans) + 1)
            examples.append((ids, mask))
    return examples, len(examples)


def train_pose_self_improve(
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
    sup_steps: int = 2500,
    rounds: int = 5,
    pseudo_per_round: int = 512,
    pseudo_batch_size: int = 128,
    n_votes: int = 5,
    vote_min: int = 3,
    lambda_u: float = 1.0,
    answer_len_bounds=None,
    eval_lengths=(),
    eval_every: int = 500,
    eval_n: int = 100,
    eval_seed: int = 0,
    seed: int,
    device: str,
    max_grad_norm: float = 1.0,
    log_every: int = 250,
):
    """Supervised warmup, then ``rounds`` rounds of frozen-teacher
    self-improvement, frontier ``max_len+1 .. max_len+rounds``.

    ``steps`` is the total budget; per-round budget is
    ``(steps - sup_steps) / rounds``. Returns a dict with ``final_loss``,
    ``history``, ``round_stats`` and ``final_frontier``.
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

    round_steps = (steps - sup_steps) // rounds
    pool: list = []
    round_stats, history, losses = [], [], []
    frontier = max_len

    def run_eval(step):
        if not eval_lengths:
            return
        acc = sweep_lengths(
            model, task, list(eval_lengths), n=eval_n, seed=eval_seed,
            device=device, tokenizer=tokenizer,
        )
        model.train()
        history.append(
            {"step": step, "frontier": frontier,
             "acc": {str(k): v for k, v in acc.items()}}
        )
        print(f"    [{task.name}/self-improve] step {step} acc: "
              + " ".join(f"L{k}={v:.2f}" for k, v in acc.items()))

    model.train()
    step = 0
    phases = [("sup", sup_steps)] + [("round", round_steps)] * rounds
    for phase, n_steps in phases:
        if phase == "round":
            frontier += 1
            teacher = copy.deepcopy(model)
            teacher.eval()
            new, kept = make_pseudo_dataset(
                teacher, task, tokenizer, rng, pseudo_per_round, frontier,
                n_votes=n_votes, vote_min=vote_min,
                answer_len_bounds=answer_len_bounds, device=device,
            )
            del teacher
            pool.extend(new)
            round_stats.append(
                {"frontier": frontier, "kept": kept,
                 "of": pseudo_per_round, "pool": len(pool)}
            )
            print(f"    [{task.name}/self-improve] frontier {frontier}: "
                  f"kept {kept}/{pseudo_per_round} pseudo-examples "
                  f"(pool {len(pool)})")
            model.train()
        for _ in range(n_steps):
            step += 1
            sup_batch = make_train_batch(
                task, tokenizer, rng.randint, batch_size, min_len, max_len
            )
            sup_batch = {k: v.to(device) for k, v in sup_batch.items()}
            optimizer.zero_grad()
            outputs = model(**sup_batch)
            loss = criterion(**outputs, **sup_batch)["loss"]
            if pool:
                idx = rng.choice(len(pool), size=min(pseudo_batch_size, len(pool)))
                u_batch = pad_batch([pool[i] for i in idx], tokenizer.pad_id)
                u_ids = u_batch["input_ids"].to(device)
                u_mask = u_batch["loss_mask"].to(device)
                positions = model.sample_pose_positions(*u_ids.shape, device)
                u_out = model(input_ids=u_ids, positions=positions)
                loss = loss + lambda_u * criterion(u_out["logits"], u_ids, u_mask)["loss"]
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % eval_every == 0 or step == steps:
                run_eval(step)
            if step % log_every == 0 or step == steps:
                print(
                    f"    [{task.name}/self-improve] step {step}/{steps} "
                    f"loss={np.mean(losses[-log_every:]):.4f} "
                    f"frontier={frontier} pool={len(pool)} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
    return {
        "final_loss": float(np.mean(losses[-log_every:])),
        "history": history,
        "round_stats": round_stats,
        "final_frontier": frontier,
    }
