#!/usr/bin/env python3
"""Plot ID convergence speed vs attention share (merged by n_heads).

Reads ``results_convergence_steps.csv`` (steps until ID ≥ 0.99).
X = mean attn/(attn+MLP) within each head-count group.
Y = steps_trained. No arch / head-count labels on points.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.architectures import ARCHITECTURES  # noqa: E402


def attn_frac(cfg: dict) -> float:
    d, ff, L = cfg["d_model"], cfg["d_ff"], cfg["n_layers"]
    return (4 * L * d * d) / (4 * L * d * d + 2 * L * d * ff)


def main() -> None:
    path = ROOT / "results_convergence_steps.csv"
    rows = list(csv.DictReader(path.open()))
    by_h: dict[int, list[float]] = defaultdict(list)
    frac_h: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if (r.get("PE") or "nope") != "nope":
            continue
        arch = r["Architecture"]
        if arch not in ARCHITECTURES:
            continue
        cfg = ARCHITECTURES[arch]
        h = cfg["n_heads"]
        by_h[h].append(float(r["Steps_Trained"]))
        frac_h[h].append(attn_frac(cfg))

    order = sorted(by_h.keys(), reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, means, stds = [], [], []
    for h in order:
        vals = np.array(by_h[h], dtype=float)
        x = float(np.mean(frac_h[h]))
        ax.scatter([x] * len(vals), vals, s=36, alpha=0.5, color="#3b6d9c", zorder=3)
        xs.append(x)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    ax.errorbar(
        xs,
        means,
        yerr=stds,
        fmt="-o",
        color="#1a3a5c",
        capsize=4,
        lw=2,
        ms=7,
        zorder=4,
        label="mean ± std",
    )
    ax.axhline(15000, ls="--", color="#999", lw=1, label="budget 15k")
    ax.set_xlabel("Attention share  attn / (attn + MLP)")
    ax.set_ylabel("Steps until ID ≥ 0.99")
    ax.set_title("ADDITION · NoPE · ID convergence speed")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 16000)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = ROOT / "figures" / "convergence_vs_attn_frac.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")
    print(f"{'h':>3} {'attn≈':>7} {'steps±std':>14} {'n':>3}")
    for h, x, m, s in zip(order, xs, means, stds):
        print(f"{h:>3} {x:>7.3f} {m:>7.0f}±{s:<5.0f} {len(by_h[h]):>3}")


if __name__ == "__main__":
    main()
