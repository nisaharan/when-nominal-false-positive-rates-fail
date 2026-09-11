"""How much of the key space inflates? The ten-key result as a distribution.

Section IV-B measures ten keys. Ten establish that the effect exists and that its
ranking is stable, but they cannot say what fraction of the key space carries it,
and a reviewer is right to ask. This reads the wider sweep and reports the
marginal distribution of the null green rate and of the empirical false-positive
rate at the nominal threshold.

The sweep re-derives keys from the same schedule, so conditions 0-9 are the
paper's ten and appear in the distribution as a labelled subset rather than as a
separate experiment.

Inputs:  results/phase2-key-sweep/sweep-cells.csv, run.json
Outputs: results/phase2-key-sweep/key-sweep-summary.json
         paper/figures/fig9-key-sweep.pdf
         paper/phase2-key-sweep-tables.md

Usage (repo root): .venv/bin/python validation/analyse_phase2_key_sweep.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

ROOT = os.environ.get("AIWM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "phase2-key-sweep")
FIG = os.path.join(ROOT, "paper", "figures")
GAMMA = fs.GAMMA
Z = fs.Z_NOMINAL
PAPER_KEYS = 10


def cp(x, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def figure(c: pd.DataFrame, L: int, out: str) -> None:
    g = c[c.length == L].sort_values("key_id").reset_index(drop=True)
    idx = g.index.to_numpy()
    is_paper = np.array([int(k.split("-")[1]) < PAPER_KEYS for k in g.key_id])

    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH, 2.75),
                            gridspec_kw={"width_ratios": [1.05, 1, 1]})

    # (a) where the keys land, ten against many
    ax = axes[0]
    ax.scatter(idx[~is_paper], g.p_k[~is_paper] * 100, s=9, color=fs.NEUTRAL,
               zorder=3, label=f"the other {int((~is_paper).sum())}")
    ax.scatter(idx[is_paper], g.p_k[is_paper] * 100, s=22, color=fs.CAT[1],
               marker="s", zorder=4, edgecolor="white", linewidth=0.4,
               label=f"the {PAPER_KEYS} in Section IV-B")
    ax.axhline(GAMMA * 100, color=fs.INK, linewidth=1.0, zorder=5)
    ax.text(len(g) * 0.42, GAMMA * 100 - 0.22, "assumed 25%", ha="left", va="top",
            fontsize=7.0, color=fs.INK)
    ax.set_xlabel("Key, in derivation order")
    ax.set_ylabel(f"Null green rate $p_k$ (%), {L} tokens")
    ax.legend(loc="upper left", fontsize=6.9, handletextpad=0.3)
    fs.title(ax, "(a)  The ten were not unusual")
    fs.style(ax)

    # (b) the marginal distribution the ten could not show
    ax = axes[1]
    ax.hist(g.p_k * 100, bins=22, color=fs.SEQ[3], edgecolor="white", linewidth=0.4,
            zorder=3)
    ax.axvline(GAMMA * 100, color=fs.INK, linewidth=1.0, zorder=5)
    ax.set_xlabel(f"Null green rate $p_k$ (%), {L} tokens")
    ax.set_ylabel("Keys")
    fs.title(ax, f"(b)  Spread over {len(g)} keys")
    fs.style(ax)

    # (c) what that costs at the nominal threshold, drawn at the length where
    # every key is scored on all 1,500 texts rather than the 179 that reach 512
    gc = c[c.length == c.length.min()]
    ax = axes[2]
    f = np.sort(gc.fpr.values * 100)
    ecdf = np.arange(1, len(f) + 1) / len(f) * 100
    ax.step(f, ecdf, where="post", color=fs.CAT[0], linewidth=1.4, zorder=4)
    ax.axvline(1.0, color=fs.INK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=5)
    above = float((gc.fpr > 0.01).mean() * 100)
    ax.text(1.35, 4, f"{above:.0f}% of keys sit above\nthe nominal 1%", fontsize=7.0,
            color=fs.POS, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(f"Empirical false-positive rate (%), {int(gc.length.iloc[0])} tokens")
    ax.set_ylabel("Keys at or below (%)")
    fs.title(ax, "(c)  Most keys miss the promise")
    fs.style(ax, grid="both")

    fig.tight_layout(w_pad=2.2, rect=(0, 0.06, 1, 0.9))
    fig.text(0.008, 0.985, "Widening the key sample: the same schedule, "
             f"{len(g)} keys instead of ten", ha="left", va="top", fontsize=9.2,
             color=fs.INK)
    fig.text(0.008, 0.045, "Detector-only scoring of human-written Dolly passages; keys "
             "derived by the schedule of Section III-B, of which conditions 0-9 are the "
             "ten used throughout the paper. (c) is drawn at 128 tokens, where every key "
             "is scored on all 1,500 passages.",
             ha="left", va="top", fontsize=6.9, color=fs.MUTED)
    fs.save(fig, out)


def main() -> int:
    c = pd.read_csv(os.path.join(SRC, "sweep-cells.csv"))
    run = json.load(open(os.path.join(SRC, "run.json")))
    lengths = sorted(c.length.unique())
    L = max(lengths)

    def block(L2):
        g = c[c.length == L2]
        x = int((g.fpr > 0.01).sum())
        lo, hi = cp(x, len(g))
        # A point estimate above 1% is not proof for that key: with a few hundred
        # texts a true 1% cell clears 1% often enough to matter. The conservative
        # count keeps only keys whose own exact lower bound clears it.
        bounds = [cp(int(r.exceed), int(r.n_texts))[0] for r in g.itertuples()]
        certain = int(sum(b > 0.01 for b in bounds))
        return {
            "keys_above_1pct_certain": certain,
            "keys_above_1pct_certain_frac": certain / len(g),
            "n_keys": int(len(g)), "n_texts": int(g.n_texts.iloc[0]),
            "p_k_min": float(g.p_k.min()), "p_k_max": float(g.p_k.max()),
            "p_k_median": float(g.p_k.median()),
            "p_k_above_gamma_frac": float((g.p_k > GAMMA).mean()),
            "fpr_min": float(g.fpr.min()), "fpr_max": float(g.fpr.max()),
            "fpr_median": float(g.fpr.median()),
            "keys_above_1pct": x,
            "keys_above_1pct_frac": x / len(g),
            "keys_above_1pct_ci95": [lo, hi],
            "keys_above_5pct_frac": float((g.fpr > 0.05).mean()),
            "keys_above_10pct_frac": float((g.fpr > 0.10).mean()),
        }

    paper = c[(c.length == L) & (c.key_id.str.split("-").str[1].astype(int) < PAPER_KEYS)]
    wide = c[c.length == L]
    summary = {
        "scope": ("marginal distribution of the KGW null green rate and nominal-threshold "
                  "false-positive rate over a wider key sample; detector-only, human text"),
        "n_keys": int(run["n_keys"]),
        "texts_scored": int(run["texts_scored"]),
        "paper_keys_are_prefix": bool(run.get("paper_keys_are_prefix")),
        "by_length": {int(L2): block(L2) for L2 in lengths},
        "paper_ten_vs_all_at_max_length": {
            "length": int(L),
            "paper_ten_p_k_range": [float(paper.p_k.min()), float(paper.p_k.max())],
            "all_keys_p_k_range": [float(wide.p_k.min()), float(wide.p_k.max())],
            "paper_ten_median_p_k": float(paper.p_k.median()),
            "all_keys_median_p_k": float(wide.p_k.median()),
        },
    }
    json.dump(summary, open(os.path.join(SRC, "key-sweep-summary.json"), "w"), indent=2)
    figure(c, L, os.path.join(FIG, "fig9-key-sweep.pdf"))

    lines = ["# Key-space sweep (generated)\n",
             "`keys > 1%` counts point estimates; `certain` keeps only keys whose own",
             "exact 95% lower bound also clears 1%.\n",
             "Source: `validation/analyse_phase2_key_sweep.py`. Detector-only KGW scoring",
             "of human-written Dolly passages under keys from the same schedule as the",
             "paper's ten, which are conditions 0-9 of this sweep.\n",
             "| Length | Keys | Texts | $p_k$ range | median $p_k$ | FPR range | keys > 1% | certain |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for L2 in lengths:
        b = summary["by_length"][int(L2)]
        lines.append(
            f"| {L2} | {b['n_keys']} | {b['n_texts']:,} | "
            f"{b['p_k_min']:.3f}-{b['p_k_max']:.3f} | {b['p_k_median']:.3f} | "
            f"{b['fpr_min']*100:.2f}%-{b['fpr_max']*100:.2f}% | "
            f"{b['keys_above_1pct']}/{b['n_keys']} ({b['keys_above_1pct_frac']*100:.0f}%) | "
            f"{b['keys_above_1pct_certain']}/{b['n_keys']} "
            f"({b['keys_above_1pct_certain_frac']*100:.0f}%) |")
    open(os.path.join(ROOT, "paper", "phase2-key-sweep-tables.md"), "w").write(
        "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
