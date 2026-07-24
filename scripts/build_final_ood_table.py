#!/usr/bin/env python3
"""Build the final ADDITION NoPE OOD summary table + plot.

Merges:
  * results_add_5seeds.csv      (A/B/C/D)
  * results_add_attn_ratio.csv  (E/F/G)
  * results_add_heads_extra.csv (H 7h / I 3h)

Seed policy for the head-sweep columns in the summary:
  * 1 head (D) and 5 heads (E): all available seeds (target ≥5)
  * 3 heads (I) and 7 heads (H): seeds 0-4 (5 runs)
Other arches keep all available seeds.
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

from src.model.architectures import ARCHITECTURES, ARCH_NAMES  # noqa: E402


def attn_frac(cfg: dict) -> float:
    d, ff, L = cfg["d_model"], cfg["d_ff"], cfg["n_layers"]
    return (4 * L * d * d) / (4 * L * d * d + 2 * L * d * ff)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    sources = [
        ("results_add_5seeds.csv", False),
        ("results_add_attn_ratio.csv", True),
        ("results_add_heads_extra.csv", True),
    ]
    for name, has_pe in sources:
        path = ROOT / name
        if not path.exists():
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "Architecture": r["Architecture"],
                        "Seed": int(r["Seed"]),
                        "ID_Accuracy": float(r["ID_Accuracy"]),
                        "OOD_Accuracy": float(r["OOD_Accuracy"]),
                        "source": name,
                    }
                )
    # dedup by (arch, seed) keeping last
    uniq: dict[tuple, dict] = {}
    for r in rows:
        uniq[(r["Architecture"], r["Seed"])] = r
    return list(uniq.values())


def main() -> None:
    rows = load_rows()
    # per-run final table
    run_path = ROOT / "results_final_ood_runs.csv"
    with run_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Architecture",
                "n_heads",
                "d_model",
                "d_ff",
                "attn_frac",
                "Seed",
                "ID_Accuracy",
                "OOD_Accuracy",
                "source",
            ]
        )
        for r in sorted(rows, key=lambda x: (-attn_frac(ARCHITECTURES[x["Architecture"]]), x["Architecture"], x["Seed"])):
            cfg = ARCHITECTURES[r["Architecture"]]
            w.writerow(
                [
                    r["Architecture"],
                    cfg["n_heads"],
                    cfg["d_model"],
                    cfg["d_ff"],
                    f"{attn_frac(cfg):.3f}",
                    r["Seed"],
                    f"{r['ID_Accuracy']:.6f}",
                    f"{r['OOD_Accuracy']:.6f}",
                    r["source"],
                ]
            )

    by_arch: dict[str, list] = defaultdict(list)
    for r in rows:
        by_arch[r["Architecture"]].append(r["OOD_Accuracy"])

    summary_path = ROOT / "results_final_ood.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Architecture",
                "n_heads",
                "d_model",
                "d_ff",
                "attn_frac",
                "OOD_mean",
                "OOD_std",
                "n_seeds",
                "seeds_used",
            ]
        )
        print(f"{'Arch':<5} {'h':>3} {'attn%':>6} {'OOD±std':>14} {'n':>3}  seeds")
        for arch in ARCH_NAMES:
            if arch not in by_arch:
                continue
            cfg = ARCHITECTURES[arch]
            arch_rows = [r for r in rows if r["Architecture"] == arch]
            seeds = sorted(r["Seed"] for r in arch_rows)
            vals = np.array([r["OOD_Accuracy"] for r in arch_rows])
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            w.writerow(
                [
                    arch,
                    cfg["n_heads"],
                    cfg["d_model"],
                    cfg["d_ff"],
                    f"{attn_frac(cfg):.3f}",
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    len(vals),
                    ",".join(map(str, seeds)),
                ]
            )
            print(
                f"{arch:<5} {cfg['n_heads']:>3} {100*attn_frac(cfg):>5.1f}% "
                f"{mean:>6.3f}±{std:<5.3f} {len(vals):>3}  {seeds}"
            )

    # plot
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs, means, stds = [], [], []
    for arch in ARCH_NAMES:
        arch_rows = [r for r in rows if r["Architecture"] == arch]
        if not arch_rows:
            continue
        cfg = ARCHITECTURES[arch]
        vals = np.array([r["OOD_Accuracy"] for r in arch_rows])
        x = attn_frac(cfg)
        ax.scatter([x] * len(vals), vals, s=36, alpha=0.55, color="#3b6d9c", zorder=3)
        xs.append(x)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        ax.annotate(
            f"{arch}\n{cfg['n_heads']}h",
            (x, vals.mean()),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )
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
    ax.set_xlabel("Attention share  attn / (attn + MLP)")
    ax.set_ylabel("OOD exact-match accuracy")
    ax.set_title("ADDITION · NoPE · ID 1–10 · OOD 11–15  (final)")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = ROOT / "figures" / "ood_vs_attn_frac.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"\nwrote {run_path.name}, {summary_path.name}, {out}")


if __name__ == "__main__":
    main()
