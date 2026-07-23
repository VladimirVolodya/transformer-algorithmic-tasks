"""Compare baseline vs PoSE / PoSE+FixMatch on one task (default: ADDITION).

Trains with the repo's standard recipe and data seeds (identical streams across
variants), then reports autoregressive exact match in-domain (lengths 1-5) and
OOD (lengths 6-15). Results are saved as JSON under ``saved/pose_fixmatch/``.

    uv run python run_pose_fixmatch.py --variant baseline
    uv run python run_pose_fixmatch.py --variant pose
    uv run python run_pose_fixmatch.py --variant pose_fixmatch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.benchmark import train_on_task
from src.model.transformer import DecoderTransformer
from src.tasks import TASKS, TaskTokenizer
from src.tasks.evaluate import sweep_lengths
from src.tasks.fixmatch import train_pose_fixmatch
from src.tasks.self_improve import train_pose_self_improve

VARIANTS = ("baseline", "pose", "pose_fixmatch", "pose_fixmatch_curr", "pose_selfimprove")

# Structural filter for pseudo-answers, per task: sum of two L-digit numbers
# has L or L+1 digits.
ANSWER_LEN_BOUNDS = {"addition": lambda L: (L, L + 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--task", default="addition")
    p.add_argument("--train-max-len", type=int, default=5,
                   help="max operand length in supervised training; ID eval is "
                        "1..this, OOD eval is this+1..ood-max-len")
    p.add_argument("--ood-max-len", type=int, default=15)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--sup-steps", type=int, default=None,
                   help="pose_selfimprove: supervised warmup steps before "
                        "self-improvement rounds (default steps//2)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--unlabeled-batch-size", type=int, default=128)
    p.add_argument("--unlabeled-every", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--lambda-u", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-seed", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.train_seed)

    task = TASKS[args.task]()
    tokenizer = TaskTokenizer()
    pe_variant = "absolute" if args.variant == "baseline" else "pose"
    model = DecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_layers=4, n_heads=2, head_dim=64, d_ff=512,
        max_len=64, pe_variant=pe_variant, pad_id=tokenizer.pad_id,
    ).to(device)
    print(f"== {args.variant} on {args.task}: "
          f"{model.num_parameters():,} params, device={device}")

    L = args.train_max_len
    ood_lo, ood_hi = L + 1, args.ood_max_len
    common = dict(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        min_len=1, max_len=L, seed=args.train_seed, device=device,
    )
    eval_snapshot_lengths = tuple(range(ood_lo, min(ood_lo + 5, ood_hi + 1)))
    t0 = time.time()
    accept_rate = None
    fm_info = {}
    if args.variant.startswith("pose_fixmatch"):
        curriculum = args.variant == "pose_fixmatch_curr"
        fm_info = train_pose_fixmatch(
            model, task, tokenizer,
            unlabeled_batch_size=args.unlabeled_batch_size,
            unlabeled_min_len=ood_lo, unlabeled_max_len=ood_hi,
            unlabeled_every=args.unlabeled_every,
            tau=args.tau, lambda_u=args.lambda_u,
            answer_len_bounds=ANSWER_LEN_BOUNDS.get(args.task) if curriculum else None,
            curriculum=curriculum,
            eval_lengths=eval_snapshot_lengths, eval_every=500,
            eval_n=100, eval_seed=args.eval_seed,
            **common,
        )
        final_loss, accept_rate = fm_info["final_loss"], fm_info["accept_rate"]
    elif args.variant == "pose_selfimprove":
        fm_info = train_pose_self_improve(
            model, task, tokenizer,
            sup_steps=args.sup_steps or args.steps // 2,
            rounds=min(5, ood_hi - L),
            pseudo_per_round=512,
            pseudo_batch_size=args.unlabeled_batch_size,
            n_votes=5, vote_min=3,
            lambda_u=args.lambda_u,
            answer_len_bounds=ANSWER_LEN_BOUNDS[args.task],
            eval_lengths=eval_snapshot_lengths, eval_every=500,
            eval_n=100, eval_seed=args.eval_seed,
            **common,
        )
        final_loss = fm_info["final_loss"]
    else:
        final_loss = train_on_task(model, task, tokenizer, **common)
    train_time = time.time() - t0

    eval_common = dict(
        n=args.n_eval, seed=args.eval_seed, device=device,
        tokenizer=tokenizer, batch_size=256,
    )
    id_acc = sweep_lengths(model, task, list(range(1, L + 1)), **eval_common)
    ood_acc = sweep_lengths(model, task, list(range(ood_lo, ood_hi + 1)), **eval_common)

    results = {
        "variant": args.variant,
        "task": args.task,
        "config": vars(args),
        "device": device,
        "final_train_loss": final_loss,
        "final_accept_rate": accept_rate,
        "final_frontier": fm_info.get("final_frontier"),
        "training_history": fm_info.get("history"),
        "round_stats": fm_info.get("round_stats"),
        "train_seconds": round(train_time, 1),
        "in_domain": {str(k): v for k, v in id_acc.items()},
        "in_domain_mean": float(np.mean(list(id_acc.values()))),
        "ood": {str(k): v for k, v in ood_acc.items()},
        "ood_mean": float(np.mean(list(ood_acc.values()))),
    }
    suffix = f"_L{L}" if L != 5 else ""
    out = Path(args.out or f"saved/pose_fixmatch/{args.task}_{args.variant}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"== {args.variant}: in-domain mean={results['in_domain_mean']:.3f} "
          f"OOD mean={results['ood_mean']:.3f} ({train_time:.0f}s train)")
    print(f"Saved to {out}")

    ckpt = out.with_suffix(".pth")
    torch.save(model.state_dict(), ckpt)
    print(f"Checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
