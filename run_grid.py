#!/usr/bin/env python3
"""Architecture x PE grid runner -> CSV (no weight checkpoints).

Each finished cell appends one row::

    Task, Architecture, PE, Seed, ID_Accuracy, OOD_Accuracy, Steps_Trained

Reuses :func:`src.benchmark.run_one_task`. Default budget is 15000 steps with
early stop when ID exact-match reaches 0.99.

Examples::

    # factor grid (~7h on MPS): ADD, A/B/C x nope/rope/absolute x 3 seeds
    uv run python run_grid.py --device mps --arches A,B,C --tasks addition \\
      --pe-variants nope,rope,absolute --seeds 0,1,2 \\
      --train-max-len 12 --ood-min-len 13 --ood-max-len 20 \\
      --csv results_add_pe_arch_3seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import resolve_device, run_one_task  # noqa: E402
from src.model.architectures import ARCH_NAMES, ARCHITECTURES  # noqa: E402
from src.model.transformer import DecoderTransformer  # noqa: E402
from src.tasks import TASKS  # noqa: E402
from src.utils.init_utils import set_random_seed  # noqa: E402

CSV_COLUMNS = [
    "Task",
    "Architecture",
    "PE",
    "Seed",
    "ID_Accuracy",
    "OOD_Accuracy",
    "Steps_Trained",
]
PE_VARIANTS = ("nope", "rope", "absolute")


def _parse_list(raw: str | None, default: tuple) -> list:
    if raw is None or raw.strip() == "":
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_done(csv_path: Path) -> set[tuple]:
    """Keys: (task, arch, pe, seed). Legacy rows without PE count as pe='nope'."""
    done: set[tuple] = set()
    if not csv_path.is_file():
        return done
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pe = row.get("PE") or "nope"
                done.add((row["Task"], row["Architecture"], pe, int(row["Seed"])))
            except (KeyError, ValueError, TypeError):
                continue
    return done


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def make_arch_factory(arch_name: str, pe_variant: str):
    """Return ``make_model(vocab_size, pad_id)`` for arch + PE override."""
    if pe_variant not in PE_VARIANTS:
        raise ValueError(f"unknown pe_variant {pe_variant!r}")
    kwargs = dict(ARCHITECTURES[arch_name])
    kwargs["pe_variant"] = pe_variant

    def make_model(vocab_size, pad_id):
        return DecoderTransformer(vocab_size=vocab_size, pad_id=pad_id, **kwargs)

    return make_model


def run_grid(
    *,
    arches: list[str],
    tasks: list[str],
    pe_variants: list[str],
    seeds: list[int],
    csv_path: Path,
    steps: int,
    batch_size: int,
    lr: float,
    n_eval: int,
    device: str,
    train_min_len: int,
    train_max_len: int,
    id_lengths: list[int],
    ood_lengths: list[int],
    early_stop_id: float | None = 0.99,
    early_stop_every: int = 1000,
) -> None:
    device = resolve_device(device)
    done = _load_done(csv_path)
    total = len(arches) * len(tasks) * len(pe_variants) * len(seeds)
    remaining = total - sum(
        1
        for a in arches
        for t in tasks
        for pe in pe_variants
        for s in seeds
        if (t, a, pe, s) in done
    )
    print(
        f"Grid: {len(arches)} arches x {len(tasks)} tasks x {len(pe_variants)} PE "
        f"x {len(seeds)} seeds = {total} cells ({remaining} remaining) on {device}"
    )
    print(
        f"CSV: {csv_path}  |  steps<={steps}  early_stop_id={early_stop_id} "
        f"every={early_stop_every}"
    )
    print(f"ID lengths: {id_lengths}  |  OOD lengths: {ood_lengths}")
    print(f"PE variants: {pe_variants}")
    for name in arches:
        base = {k: v for k, v in ARCHITECTURES[name].items() if k != "pe_variant"}
        probe = DecoderTransformer(
            vocab_size=19, pad_id=13, pe_variant=pe_variants[0], **base
        )
        print(f"  Arch {name}: {probe.num_parameters():,} params  {base}")

    cell = 0
    for arch in arches:
        for pe in pe_variants:
            make_model = make_arch_factory(arch, pe)
            for task in tasks:
                for seed in seeds:
                    cell += 1
                    key = (task, arch, pe, seed)
                    tag = (
                        f"[{cell}/{total}] arch={arch} pe={pe} "
                        f"task={task} seed={seed}"
                    )
                    if key in done:
                        print(f"{tag}  SKIP (already in CSV)")
                        continue
                    print(f"\n{tag}")
                    set_random_seed(seed)
                    metrics = run_one_task(
                        make_model,
                        task,
                        steps=steps,
                        batch_size=batch_size,
                        lr=lr,
                        train_min_len=train_min_len,
                        train_max_len=train_max_len,
                        id_lengths=id_lengths,
                        ood_lengths=ood_lengths,
                        n_eval=n_eval,
                        train_seed=seed,
                        eval_seed=seed,
                        device=device,
                        early_stop_id=early_stop_id,
                        early_stop_every=early_stop_every,
                    )
                    _append_row(
                        csv_path,
                        {
                            "Task": task,
                            "Architecture": arch,
                            "PE": pe,
                            "Seed": seed,
                            "ID_Accuracy": f"{metrics['in_domain_mean']:.6f}",
                            "OOD_Accuracy": f"{metrics['ood_mean']:.6f}",
                            "Steps_Trained": int(
                                metrics.get("steps_trained", steps)
                            ),
                        },
                    )
                    print(
                        f"{tag}  -> ID={metrics['in_domain_mean']:.4f} "
                        f"OOD={metrics['ood_mean']:.4f}  "
                        f"steps={metrics.get('steps_trained', '?')}  (appended)"
                    )

    print(f"\nDone. Results in {csv_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--arches", default=",".join(ARCH_NAMES))
    parser.add_argument("--tasks", default=",".join(TASKS.keys()))
    parser.add_argument(
        "--pe-variants",
        default="nope",
        help="Comma-separated: nope,rope,absolute",
    )
    parser.add_argument("--seeds", default=",".join(str(i) for i in range(10)))
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-min-len", type=int, default=1)
    parser.add_argument("--train-max-len", type=int, default=10)
    parser.add_argument(
        "--ood-min-len",
        type=int,
        default=None,
        help="Default: train_max_len + 1",
    )
    parser.add_argument(
        "--ood-max-len",
        type=int,
        default=None,
        help="Default: train_max_len + 5",
    )
    parser.add_argument(
        "--early-stop-id",
        type=float,
        default=0.99,
        help="0 disables early stopping",
    )
    parser.add_argument("--early-stop-every", type=int, default=1000)
    args = parser.parse_args(argv)

    arches = _parse_list(args.arches, ARCH_NAMES)
    tasks = _parse_list(args.tasks, tuple(TASKS.keys()))
    pe_variants = _parse_list(args.pe_variants, ("nope",))
    seeds = [int(s) for s in _parse_list(args.seeds, tuple(range(10)))]
    early_stop_id = None if args.early_stop_id <= 0 else args.early_stop_id

    ood_min = args.ood_min_len if args.ood_min_len is not None else args.train_max_len + 1
    ood_max = args.ood_max_len if args.ood_max_len is not None else args.train_max_len + 5
    if ood_min <= args.train_max_len:
        raise SystemExit(
            f"OOD must not overlap ID: ood_min_len={ood_min} "
            f"<= train_max_len={args.train_max_len}"
        )
    id_lengths = list(range(args.train_min_len, args.train_max_len + 1))
    ood_lengths = list(range(ood_min, ood_max + 1))

    for a in arches:
        if a not in ARCHITECTURES:
            raise SystemExit(f"Unknown architecture {a!r}; choose from {ARCH_NAMES}")
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"Unknown task {t!r}; choose from {tuple(TASKS)}")
    for pe in pe_variants:
        if pe not in PE_VARIANTS:
            raise SystemExit(f"Unknown PE {pe!r}; choose from {PE_VARIANTS}")

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    run_grid(
        arches=arches,
        tasks=tasks,
        pe_variants=pe_variants,
        seeds=seeds,
        csv_path=csv_path,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_eval=args.n_eval,
        device=args.device,
        train_min_len=args.train_min_len,
        train_max_len=args.train_max_len,
        id_lengths=id_lengths,
        ood_lengths=ood_lengths,
        early_stop_id=early_stop_id,
        early_stop_every=args.early_stop_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
