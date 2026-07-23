"""Accuracy-vs-length curves for the PoSE/FixMatch/self-improve study.

Reads the result JSONs in ``saved/pose_fixmatch/`` and renders one figure per
experiment family (train<=5 study, train<=10 study). Colors/chrome follow the
validated reference palette (categorical slots in fixed order, direct labels
as contrast relief, recessive grid, single axis).

    uv run python scripts/plot_length_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "saved" / "pose_fixmatch"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
# Categorical slots 1..4 (validated, fixed order)
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]


def load(name):
    path = ROOT / f"{name}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    acc = {int(k): v for k, v in r["in_domain"].items()}
    acc.update({int(k): v for k, v in r["ood"].items()})
    return dict(sorted(acc.items()))


def plot_family(specs, train_max, title, out_name):
    curves = [(label, load(name)) for label, name in specs]
    curves = [(label, c) for label, c in curves if c is not None]
    if not curves:
        return False

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    max_len = max(max(c) for _, c in curves)
    ax.axvspan(train_max + 0.5, max_len + 0.5, color=GRID, alpha=0.35, zorder=0)
    ax.text(train_max + 0.62, 1.06, "OOD →", color=MUTED, fontsize=9, va="bottom")

    for i, (label, acc) in enumerate(curves):
        xs, ys = list(acc.keys()), list(acc.values())
        color = SERIES[i]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5.5,
                markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1, zorder=3, label=label)
        # Direct label at the last point where the curve is still visible,
        # nudged apart via per-series vertical offsets set below.

    # Direct labels: place at x of last value >= 0.02, else at train boundary.
    used_y = []
    for i, (label, acc) in enumerate(curves):
        xs, ys = list(acc.keys()), list(acc.values())
        vis = [x for x, y in acc.items() if y >= 0.02]
        lx = vis[-1] if vis else train_max
        ly = acc[lx]
        while any(abs(ly - u) < 0.07 for u in used_y):
            ly += 0.075
        used_y.append(ly)
        ax.annotate(label, (lx, acc[lx]), xytext=(8, (ly - acc[lx]) * 100 + 4),
                    textcoords="offset points", color=SECONDARY, fontsize=9,
                    fontweight="medium")

    ax.set_xlim(0.5, max_len + 2.2)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xticks(range(1, max_len + 1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("operand length (digits)", color=SECONDARY, fontsize=10)
    ax.set_ylabel("exact match", color=SECONDARY, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY)

    out = ROOT / out_name
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return True


if __name__ == "__main__":
    plot_family(
        [
            ("baseline (abs PE)", "addition_baseline_L10"),
            ("PoSE", "addition_pose_L10"),
            ("PoSE + self-improve", "addition_pose_selfimprove_L10"),
        ],
        train_max=10,
        title="ADDITION - exact match vs operand length (train 1-10, OOD 11-15)",
        out_name="accuracy_vs_length_L10.png",
    )
    plot_family(
        [
            ("baseline (abs PE)", "addition_baseline"),
            ("PoSE", "addition_pose"),
            ("PoSE + FixMatch", "addition_pose_fixmatch"),
            ("PoSE + FixMatch curr.", "addition_pose_fixmatch_curr"),
        ],
        train_max=5,
        title="ADDITION - exact match vs operand length (train 1-5, OOD 6-15)",
        out_name="accuracy_vs_length_L5.png",
    )
