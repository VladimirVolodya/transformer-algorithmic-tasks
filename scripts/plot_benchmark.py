#!/usr/bin/env python3
"""Plot length-generalization curves from benchmark JSON files.

Reads one or more ``saved/benchmark_*.json`` results, prints a per-length
table, draws accuracy-vs-length curves (one panel per task), and prints a
short PASS/FAIL-style verdict.

Examples::

    uv run python scripts/plot_benchmark.py saved/benchmark_rope.json
    uv run python scripts/plot_benchmark.py saved/benchmark_*.json
    uv run python scripts/plot_benchmark.py saved/benchmark_rope.json \\
        --out figures/length_gen.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

ID_THRESHOLD = 0.95
# Early OOD lengths used for the soft "still generalizes a bit" check.
EARLY_OOD = (6, 7, 8)
EARLY_OOD_FLOOR = 0.05  # above chance-ish noise for exact-match collapse


def _resolve(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT_PATH / p
        matches = sorted(p.parent.glob(p.name)) if any(c in p.name for c in "*?[") else [p]
        for m in matches:
            if m.is_file():
                out.append(m)
    # de-dupe, keep order
    seen = set()
    unique = []
    for m in out:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def _label(path: Path, data: dict) -> str:
    model = data.get("model") or {}
    pe = model.get("pe_variant")
    if pe:
        return str(pe)
    # fallback: benchmark_rope.json -> rope
    stem = path.stem
    if stem.startswith("benchmark_"):
        return stem[len("benchmark_") :]
    return stem


def _curve(task_res: dict) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for split in ("in_domain", "ood"):
        for length, acc in (task_res.get(split) or {}).items():
            pairs.append((int(length), float(acc)))
    pairs.sort()
    if not pairs:
        return np.array([]), np.array([])
    lengths, accs = zip(*pairs)
    return np.asarray(lengths), np.asarray(accs)


def _print_table(runs: list[tuple[str, dict]]) -> None:
    tasks = sorted({t for _, d in runs for t in d.get("tasks", {})})
    for task in tasks:
        print(f"\n=== {task} ===")
        # union of lengths
        lengths = sorted(
            {
                int(L)
                for _, d in runs
                for split in ("in_domain", "ood")
                for L in (d.get("tasks", {}).get(task, {}).get(split) or {})
            }
        )
        if not lengths:
            print("  (no data)")
            continue
        header = f"{'len':>4}" + "".join(f"  {lab:>10}" for lab, _ in runs)
        print(header)
        print("-" * len(header))
        for L in lengths:
            row = f"{L:>4}"
            for lab, data in runs:
                tres = data.get("tasks", {}).get(task, {})
                acc = (tres.get("in_domain") or {}).get(str(L))
                if acc is None:
                    acc = (tres.get("ood") or {}).get(str(L))
                row += f"  {acc:10.3f}" if acc is not None else f"  {'—':>10}"
            print(row)
        means = f"{'IDμ':>4}"
        for _, data in runs:
            tres = data.get("tasks", {}).get(task, {})
            m = tres.get("in_domain_mean")
            means += f"  {m:10.3f}" if m is not None else f"  {'—':>10}"
        print(means)
        means = f"{'ODμ':>4}"
        for _, data in runs:
            tres = data.get("tasks", {}).get(task, {})
            m = tres.get("ood_mean")
            means += f"  {m:10.3f}" if m is not None else f"  {'—':>10}"
        print(means)


def _verdict(runs: list[tuple[str, dict]]) -> None:
    print("\n=== verdict ===")
    train_max = None
    for _, data in runs:
        b = data.get("benchmark") or {}
        if b.get("train_max_len") is not None:
            train_max = int(b["train_max_len"])
            break
    if train_max is None:
        train_max = 5

    for lab, data in runs:
        print(f"\n[{lab}]")
        for task, tres in sorted((data.get("tasks") or {}).items()):
            id_mean = float(tres.get("in_domain_mean") or 0.0)
            id_ok = id_mean >= ID_THRESHOLD
            early = []
            for L in EARLY_OOD:
                v = (tres.get("ood") or {}).get(str(L))
                if v is not None:
                    early.append(float(v))
            early_mean = float(np.mean(early)) if early else float("nan")
            early_ok = (not early) or early_mean >= EARLY_OOD_FLOOR
            id_flag = "PASS" if id_ok else "FAIL"
            ood_flag = "PASS" if early_ok else "FAIL"
            print(
                f"  {task:<10} ID mean={id_mean:.3f} [{id_flag}]  "
                f"early OOD {list(EARLY_OOD)} mean={early_mean:.3f} [{ood_flag}]"
            )
            if task == "addition" and id_ok and not early_ok:
                print(
                    "             note: in-domain perfect but OOD collapsed "
                    f"(train_max_len={train_max})"
                )


def _plot(runs: list[tuple[str, dict]], out: Path, train_max_len: int) -> None:
    tasks = sorted({t for _, d in runs for t in d.get("tasks", {})})
    if not tasks:
        print("No tasks to plot.", file=sys.stderr)
        return

    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, task in zip(axes, tasks):
        for lab, data in runs:
            tres = data.get("tasks", {}).get(task)
            if not tres:
                continue
            lengths, accs = _curve(tres)
            if lengths.size == 0:
                continue
            ax.plot(lengths, accs, marker="o", markersize=3.5, label=lab)
        ax.axvline(train_max_len + 0.5, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_title(task)
        ax.set_xlabel("length")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("exact-match accuracy")
    axes[-1].legend(title="pe_variant", loc="best", fontsize=8)

    fig.suptitle("Length generalization (benchmark)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote plot to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsons",
        nargs="+",
        help="Benchmark JSON path(s); globs allowed, e.g. saved/benchmark_*.json",
    )
    parser.add_argument(
        "--out",
        default="figures/length_generalization.png",
        help="Output figure path (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print tables / verdict, skip writing a figure",
    )
    args = parser.parse_args(argv)

    paths = _resolve(args.jsons)
    if not paths:
        print("No JSON files found.", file=sys.stderr)
        return 1

    runs: list[tuple[str, dict]] = []
    for path in paths:
        data = json.loads(path.read_text())
        runs.append((_label(path, data), data))
        print(f"Loaded {path}  ({_label(path, data)})")

    train_max = 5
    for _, data in runs:
        b = data.get("benchmark") or {}
        if b.get("train_max_len") is not None:
            train_max = int(b["train_max_len"])
            break

    _print_table(runs)
    _verdict(runs)

    if not args.no_plot:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT_PATH / out
        _plot(runs, out, train_max)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
