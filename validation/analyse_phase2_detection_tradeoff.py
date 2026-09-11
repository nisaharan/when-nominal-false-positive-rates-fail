"""What per-key calibration costs in detection rate.

The paper's remedy is a threshold indexed by detector key and length. The
obvious objection is that for an inflated key the calibrated threshold is far
above the nominal one -- key 07 moves from z > 2.326 to z > 6.40 at 512 tokens
-- and that such a threshold might detect nothing. This script answers that
from stored scores. No text is generated.

The watermarked outputs are the 1,000-output development positive screen. It
shares the model, the watermark configuration and the ten-key schedule with the
confirmatory null, so the thresholds frozen on the null's calibration split
apply to it unchanged. Thresholds were fitted on unwatermarked text only, so
using them here is not circular.

This is a development-scale answer: 50 watermarked outputs per key and length,
so the intervals are wide. It is not a confirmatory detection claim.

Inputs:
    results/phase2-v2-positive-sensitivity/run/batches/*.json  watermarked scores,
        or results/phase2-detection-tradeoff/watermarked-scores.csv, the compact
        export of them that ships with the release
    results/phase2-confirmatory-null/thresholds.json           frozen thresholds
    results/phase2-nominal-fpr/nominal-fpr-cells.csv           nominal FPR, n = 10,000
    results/phase2-confirmatory-null/failure-diagnosis.json    held-out FPR, n = 5,000
Outputs:
    results/phase2-detection-tradeoff/watermarked-scores.csv
    results/phase2-detection-tradeoff/detection-cells.csv
    results/phase2-detection-tradeoff/detection-summary.json
    paper/figures/fig10-detection-tradeoff.pdf
    paper/phase2-detection-tables.md

Usage (repo root): python validation/analyse_phase2_detection_tradeoff.py
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = os.environ.get("AIWM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POS = os.path.join(ROOT, "results", "phase2-v2-positive-sensitivity", "run", "batches")
CONF = os.path.join(ROOT, "results", "phase2-confirmatory-null")
NOM = os.path.join(ROOT, "results", "phase2-nominal-fpr")
OUT = os.path.join(ROOT, "results", "phase2-detection-tradeoff")
FIG = os.path.join(ROOT, "paper", "figures")
LENGTHS = (128, 256, 512)
Z_NOMINAL = stats.norm.ppf(0.99)


def cp(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


SCORE_CACHE = os.path.join(OUT, "watermarked-scores.csv")


def watermarked_scores() -> dict[tuple[str, int], list[float]]:
    """Per-output KGW z, scored under the key the output was generated with.

    Read from the raw batches when they are present, and exported to a compact
    CSV so the analysis also runs from a release, where the batches are not
    published: they contain generated text that has not been reviewed.
    """
    hits: dict[tuple[str, int], list[float]] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(POS, "*.json")))
    if not files:
        if os.path.exists(SCORE_CACHE):
            cached = pd.read_csv(SCORE_CACHE)
            for r in cached.itertuples():
                hits[(r.key_id, int(r.length))].append(float(r.value))
            return hits
        raise SystemExit(f"no watermarked batches under {POS} and no {SCORE_CACHE}")
    for path in files:
        batch = json.load(open(path))
        if batch["scheme"] != "kgw":
            continue
        for record in batch["records"]:
            prefixes = record["prefix_results"]
            if isinstance(prefixes, str):
                prefixes = ast.literal_eval(prefixes)
            for item in prefixes:
                score = item["score"]
                if score["key_id"] == record["key_id"]:
                    hits[(record["key_id"], int(item["length"]))].append(float(score["value"]))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame([{"key_id": k, "length": L, "value": v}
                  for (k, L), vals in sorted(hits.items()) for v in vals]).to_csv(
        SCORE_CACHE, index=False)
    return hits


def build_cells() -> pd.DataFrame:
    thresholds = {(c["key_id"], int(c["length"])): float(c["threshold"])
                  for c in json.load(open(os.path.join(CONF, "thresholds.json")))["operational_thresholds"]
                  if c["scheme"] == "kgw"}
    heldout = {(c["key_id"], int(c["length"])): c["confirmation_exceedances"] / 5000
               for c in json.load(open(os.path.join(CONF, "failure-diagnosis.json")))["cells"]
               if c["scheme"] == "kgw"}
    nominal = pd.read_csv(os.path.join(NOM, "nominal-fpr-cells.csv"))
    nominal = nominal[(nominal.scheme == "kgw") & (nominal.run == "smollm2_n10000")]
    nom_fpr = {(r.key_id, int(r.length)): float(r.fpr) for r in nominal.itertuples()}

    rows = []
    for (key, length), values in sorted(watermarked_scores().items()):
        n = len(values)
        t = thresholds[(key, length)]
        tp_nom = int(sum(v > Z_NOMINAL for v in values))
        tp_cal = int(sum(v > t for v in values))
        nlo, nhi = cp(tp_nom, n)
        clo, chi = cp(tp_cal, n)
        rows.append(dict(
            key_id=key, length=length, n_watermarked=n,
            threshold_nominal=Z_NOMINAL, threshold_calibrated=t,
            threshold_shift=t - Z_NOMINAL,
            tpr_nominal=tp_nom / n, tpr_nominal_lo=nlo, tpr_nominal_hi=nhi,
            tpr_calibrated=tp_cal / n, tpr_calibrated_lo=clo, tpr_calibrated_hi=chi,
            tpr_change=(tp_cal - tp_nom) / n,
            fpr_nominal=nom_fpr[(key, length)], fpr_calibrated=heldout[(key, length)]))
    return pd.DataFrame(rows)


def figure(cells: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.95))

    # (a) where each key's operating point moves when the threshold is calibrated.
    # Labels sit at the nominal end, which is spread across two decades; the
    # calibrated end is a tight cluster where labels would collide.
    ax = axes[0]
    d = cells[cells.length == 512].sort_values("fpr_nominal")
    # two keys can sit almost on top of each other at the nominal end; alternate
    # the label above and below so neither is lost
    placed: list[tuple[float, float]] = []
    for r in d.itertuples():
        up = r.threshold_shift > 0
        col = fs.POS if up else fs.NEG
        ax.annotate("", xy=(r.fpr_calibrated * 100, r.tpr_calibrated * 100),
                    xytext=(r.fpr_nominal * 100, r.tpr_nominal * 100),
                    arrowprops=dict(arrowstyle="-|>", color=col, linewidth=0.9, alpha=0.85,
                                    shrinkA=2.5, shrinkB=2.0, mutation_scale=6))
        ax.scatter([r.fpr_nominal * 100], [r.tpr_nominal * 100], s=11, color=col,
                   alpha=0.4, edgecolor="none", zorder=3)
        ax.scatter([r.fpr_calibrated * 100], [r.tpr_calibrated * 100], s=18, color=col,
                   zorder=4, edgecolor="white", linewidth=0.5,
                   marker="o" if up else "s")
        crowded = any(abs(np.log10(r.fpr_nominal) - lx) < 0.09 and abs(r.tpr_nominal - ly) < 0.02
                      for lx, ly in placed)
        placed.append((np.log10(r.fpr_nominal), r.tpr_nominal))
        ax.annotate(r.key_id.replace("kgw-", ""),
                    (r.fpr_nominal * 100, r.tpr_nominal * 100),
                    xytext=(2.5, -9 if crowded else 4), textcoords="offset points",
                    fontsize=6.6, color=fs.MUTED)
    ax.axvline(1.0, color=fs.INK, linestyle=(0, (2.5, 2.5)), linewidth=0.9, zorder=2)
    ax.text(1.12, 93.4, "1% target", fontsize=7, color=fs.INK, rotation=90,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlim(0.15, 95)
    ax.set_ylim(90, 101.5)
    ax.set_xlabel("False-positive rate (%, log scale)")
    ax.set_ylabel("Detection rate (%)")
    handles = [plt.Line2D([], [], color=fs.POS, marker="o", markersize=4, linewidth=1,
                          label="threshold raised"),
               plt.Line2D([], [], color=fs.NEG, marker="s", markersize=4, linewidth=1,
                          label="threshold lowered")]
    ax.legend(handles=handles, loc="lower left", handlelength=1.8)
    fs.title(ax, "(a)  Where calibration moves each key, 512 tokens")
    fs.style(ax, grid="both")

    # (b) the correction is two-way: for a conservative key the threshold falls
    # and detection improves
    ax = axes[1]
    keys = sorted(cells.key_id.unique())
    x = np.arange(len(keys))
    width = 0.27
    for i, L in enumerate(LENGTHS):
        g = cells[cells.length == L].set_index("key_id").loc[keys]
        ax.bar(x + (i - 1) * width, g.tpr_change * 100, width * 0.86,
               color=fs.LEN[L], linewidth=0, label=f"{L} tokens", zorder=3)
    ax.axhline(0, color=fs.INK, linewidth=0.8, zorder=4)
    for xi in range(len(keys) - 1):
        ax.axvline(xi + 0.5, color=fs.GRID, linewidth=0.6, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("kgw-", "") for k in keys])
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylim(-23, 12)
    ax.set_xlabel("Frozen detector key")
    ax.set_ylabel("Change in detection rate (points)")
    ax.legend(loc="lower left", ncol=3, columnspacing=1.0, handletextpad=0.4)
    fs.title(ax, "(b)  What that costs, and where it gains")
    fs.style(ax)

    fig.tight_layout()
    fs.save(fig, out)


def markdown(cells: pd.DataFrame, path: str) -> None:
    lines = ["# Detection-rate trade-off of per-key calibration (generated)\n",
             "Source: `validation/analyse_phase2_detection_tradeoff.py`. Watermarked outputs",
             "are the 1,000-output development positive screen, which shares the model, the",
             "watermark configuration and the key schedule with the confirmatory null, so the",
             "frozen thresholds apply unchanged. Thresholds were fitted on unwatermarked text",
             "only. Detection rate is the fraction of watermarked outputs scoring above the",
             "threshold under their own key, n = 50 per cell, with exact two-sided 95%",
             "Clopper-Pearson intervals. Development scale, not a confirmatory detection claim.\n"]
    for L in LENGTHS:
        d = cells[cells.length == L].set_index("key_id")
        lines.append(f"\n## {L} tokens\n")
        lines.append("| Key | Nominal z | Calibrated z | FPR nominal | FPR calibrated | "
                     "TPR nominal | TPR calibrated [95% CI] |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for key in sorted(d.index):
            r = d.loc[key]
            lines.append(
                f"| {key} | {r.threshold_nominal:.2f} | {r.threshold_calibrated:.2f} | "
                f"{r.fpr_nominal*100:.2f}% | {r.fpr_calibrated*100:.2f}% | {r.tpr_nominal*100:.0f}% | "
                f"{r.tpr_calibrated*100:.0f}% [{r.tpr_calibrated_lo*100:.0f}, {r.tpr_calibrated_hi*100:.0f}] |")
    open(path, "w").write("\n".join(lines) + "\n")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    cells = build_cells()
    cells.to_csv(os.path.join(OUT, "detection-cells.csv"), index=False)
    figure(cells, os.path.join(FIG, "fig10-detection-tradeoff.pdf"))
    markdown(cells, os.path.join(ROOT, "paper", "phase2-detection-tables.md"))

    raised = cells[cells.threshold_shift > 0]
    lowered = cells[cells.threshold_shift < 0]
    summary = {
        "scope": ("development-scale detection rates at the frozen per-key thresholds; "
                  "not a confirmatory detection claim"),
        "n_watermarked_per_cell": int(cells.n_watermarked.iloc[0]),
        "cells": int(len(cells)),
        "cells_threshold_raised": int(len(raised)),
        "cells_threshold_lowered": int(len(lowered)),
        "tpr_nominal_median_by_length": {int(L): float(cells[cells.length == L].tpr_nominal.median())
                                         for L in LENGTHS},
        "tpr_calibrated_median_by_length": {int(L): float(cells[cells.length == L].tpr_calibrated.median())
                                            for L in LENGTHS},
        "tpr_calibrated_min_by_length": {int(L): float(cells[cells.length == L].tpr_calibrated.min())
                                         for L in LENGTHS},
        "worst_cell": cells.loc[cells.tpr_calibrated.idxmin(),
                                ["key_id", "length", "tpr_calibrated", "tpr_calibrated_lo",
                                 "tpr_calibrated_hi"]].to_dict(),
        "mean_tpr_change_where_threshold_raised": float(raised.tpr_change.mean()),
        "mean_tpr_change_where_threshold_lowered": float(lowered.tpr_change.mean()),
        "worst_key_512": cells[(cells.key_id == "kgw-07") & (cells.length == 512)][
            ["threshold_calibrated", "fpr_nominal", "fpr_calibrated",
             "tpr_nominal", "tpr_calibrated"]].iloc[0].to_dict(),
    }
    json.dump(summary, open(os.path.join(OUT, "detection-summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
