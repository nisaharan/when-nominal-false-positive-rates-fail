#!/usr/bin/env python3
"""Build the design-margin figure from the reviewed confirmation-gate artifact.

The other figures in the manuscript come from the stored scores by way of
`analyse_phase2_nominal_fpr.py` and `analyse_phase2_human_null.py`. This script
owns the one figure whose numbers live in the gate report artifact rather than
in a score file:

    paper/figures/fig11-design-margin.pdf

Panel (a) is the calibration-to-confirmation movement in mean strict exceedances
per scheme, against the maximum count a cell could carry and still pass. Panel
(b) is the prospective union-bound planning curve. They argue one point together
and share a float in the paper, so they share a figure.

Usage (repo root): python validation/build_phase2_publication_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
MAX_PASSING = 28


def load_datasets(relative: str) -> dict[str, list[dict]]:
    artifact = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    return artifact["snapshot"]["datasets"]


def design_margin_figure(gate: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.7))

    # (a) calibration -> confirmation, per scheme. A dumbbell, because the
    # question is how far the pair moved, not how tall either end is.
    rows = gate["scheme_split_means"]
    schemes = ["KGW", "SynthID"]
    ax = axes[0]
    y = np.arange(len(schemes))[::-1]
    for yi, scheme in zip(y, schemes):
        pair = {r["split"]: r["mean_exceedances"] for r in rows if r["scheme"] == scheme}
        cal, con = pair["calibration"], pair["confirmation"]
        ax.plot([cal, con], [yi, yi], color=fs.RULE, linewidth=1.4, zorder=2,
                solid_capstyle="round")
        ax.scatter([cal], [yi], s=30, color=fs.CAT[0], marker="o", zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter([con], [yi], s=30, color=fs.CAT[1], marker="s", zorder=3,
                   edgecolor="white", linewidth=0.6)
        # the two ends can sit almost on top of each other, which is the finding:
        # label them outward so the numbers never collide with the marks.
        ax.annotate(f"{cal:.1f}", (min(cal, con), yi), xytext=(-7, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.4, color=fs.CAT[0])
        ax.annotate(f"{con:.1f}", (max(cal, con), yi), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.4, color=fs.CAT[1])
    ax.axvline(MAX_PASSING, color=fs.INK, linestyle=(0, (2.5, 2.5)), linewidth=0.9, zorder=1)
    ax.annotate("maximum a cell can carry\nand still pass", (MAX_PASSING, 1.62),
                xytext=(-6, 0), textcoords="offset points", ha="right", va="top",
                fontsize=7, color=fs.INK, linespacing=1.35)
    ax.set_yticks(y)
    ax.set_yticklabels(schemes)
    ax.set_ylim(-0.75, 1.8)
    ax.set_xlim(13, 33)
    ax.set_xlabel("Mean strict exceedances across 30 cells")
    handles = [plt.Line2D([], [], color=fs.CAT[0], marker="o", linestyle="none",
                          markersize=5, label="calibration"),
               plt.Line2D([], [], color=fs.CAT[1], marker="s", linestyle="none",
                          markersize=5, label="confirmation")]
    ax.legend(handles=handles, loc="lower left", ncol=2, columnspacing=1.0,
              handletextpad=0.3, bbox_to_anchor=(-0.02, -0.04))
    fs.title(ax, "(a)  The held-out split barely moved")
    fs.style(ax, grid="x")

    # (b) prospective planning curve. One series, so one hue and no legend box.
    ax = axes[1]
    rows = sorted(gate["power_scenarios"], key=lambda r: r["samples_per_cell"])
    labels = [f"{r['samples_per_cell']:,}" for r in rows]
    values = [r["family_95_rate"] * 100 for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, values, width=0.6, color=fs.CAT[0], linewidth=0, zorder=3)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.2f}%", (xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.2, color=fs.INK)
    ax.axhline(1.0, color=fs.INK, linestyle=(0, (2.5, 2.5)), linewidth=0.9, zorder=4)
    ax.text(len(rows) - 0.55, 1.02, "the 1% claim being made", ha="right", va="bottom",
            fontsize=7, color=fs.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25%", "0.50%", "0.75%", "1.00%"])
    ax.set_xlabel("Fresh observations per cell")
    ax.set_ylabel("True per-cell rate the design can carry")
    fs.title(ax, "(b)  Margin needed for 95% family pass")
    fs.style(ax)

    fig.tight_layout()
    fs.save(fig, out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = load_datasets("reports/phase2-confirmation-gate/artifact.json")
    design_margin_figure(gate, OUT / "fig11-design-margin.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
