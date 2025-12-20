"""
Generate a parity comparison plot (BeaconGrad vs PyTorch).

This script runs the parity checks from `examples/compare_with_pytorch.py` and
plots the forward/grad max errors (log scale) as a simple bar chart.

Run:
  uv run python examples/10_parity_plot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse

import numpy as np

# Ensure local imports work when running as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BeaconGrad vs PyTorch parity plot.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Error threshold to visualize on the plot (default: 1e-6).",
    )
    args = parser.parse_args()

    # Local import so this file stays lightweight if imported elsewhere.
    from examples.compare_with_pytorch import (
        _require_torch,
        parity_attention,
        parity_linear,
        parity_mlp_2layer,
    )

    _require_torch()

    # Run parity checks (float64) and collect max errors.
    rows = [
        ("Linear",) + tuple(parity_linear()),
        ("MLP",) + tuple(parity_mlp_2layer()),
        ("Attention",) + tuple(parity_attention()),
    ]

    names = [r[0] for r in rows]
    fwd = np.array([r[1] for r in rows], dtype=np.float64)
    grad = np.array([r[2] for r in rows], dtype=np.float64)

    # Plot
    import matplotlib.pyplot as plt

    x = np.arange(len(names))
    threshold = float(args.threshold)
    # Avoid log(0) issues for perfect matches: plot a tiny floor.
    floor = 1e-18
    fwd_plot = np.maximum(fwd, floor)
    grad_plot = np.maximum(grad, floor)

    fig, (ax_fwd, ax_grad) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 6.0), dpi=160, sharex=True
    )

    def _style_axis(ax, title: str):
        ax.set_title(title)
        ax.set_ylabel("max error (log scale)")
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.axhline(
            threshold,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"threshold={threshold:.0e}",
        )

    # Forward panel
    bars_f = ax_fwd.bar(
        x, fwd_plot, width=0.6, color="#4C78A8", label="Forward max error"
    )
    _style_axis(ax_fwd, "BeaconGrad vs PyTorch parity (float64) — Forward")
    ax_fwd.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=9,
    )

    # Grad panel
    bars_g = ax_grad.bar(
        x, grad_plot, width=0.6, color="#F58518", label="Grad max error"
    )
    _style_axis(ax_grad, "BeaconGrad vs PyTorch parity (float64) — Gradients")
    ax_grad.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=9,
    )

    # X labels + PASS/FAIL annotations (relative to threshold)
    ax_grad.set_xticks(x, names)
    for i, name in enumerate(names):
        f_ok = fwd[i] < threshold
        g_ok = grad[i] < threshold

        ax_fwd.text(
            x[i],
            fwd_plot[i] * 1.25,
            "PASS" if f_ok else "FAIL",
            ha="center",
            va="bottom",
            fontsize=8,
            color=("green" if f_ok else "crimson"),
        )
        ax_grad.text(
            x[i],
            grad_plot[i] * 1.25,
            "PASS" if g_ok else "FAIL",
            ha="center",
            va="bottom",
            fontsize=8,
            color=("green" if g_ok else "crimson"),
        )

    # Make room on the right for the outside legends.
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    out_path = Path(__file__).resolve().parents[1] / "assets" / "parity_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")

    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()


