"""Does the key ranking survive a change of vocabulary? Now with 128 keys.

Section IV-E answers this with the paper's ten keys and gets rho = -0.15 with a
95% interval of [-0.71, +0.53]. That interval contains almost every hypothesis
worth distinguishing, so the honest reading was "no evidence of strong transfer"
rather than a result. Ten keys is simply too few for a rank correlation.

This repeats the comparison over 128 keys. Both runs score an identical set of
passages at an identical set of prefix lengths, because the sweep filters
candidates on joint eligibility under both tokenizers; without that the two runs
select different passages and the comparison is not paired.

Inputs:  results/phase2-key-sweep-paired/{smollm2,qwen}/sweep-cells.csv
Outputs: results/phase2-key-sweep-paired/vocabulary-sweep-summary.json
         paper/figures/fig8-vocabulary-sweep.pdf
         paper/phase2-vocabulary-sweep-tables.md

Usage (repo root): .venv/bin/python validation/analyse_phase2_vocabulary_sweep.py
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
SRC = os.path.join(ROOT, "results", "phase2-key-sweep-paired")
FIG = os.path.join(ROOT, "paper", "figures")
GAMMA = fs.GAMMA
VOCAB = {"smollm2": "SmolLM2-135M, 49k tokens", "qwen": "Qwen2.5-0.5B, 152k tokens"}


def spearman_ci(rho, n, conf=0.95):
    if n < 4:
        return float("nan"), float("nan")
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    z = np.arctanh(rho)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def load(name):
    d = pd.read_csv(os.path.join(SRC, name, "sweep-cells.csv"))
    d["key_n"] = d.key_id.str.split("-").str[1].astype(int)
    return d


def figure(a, b, L, rho, ci, out):
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 3.0),
                             gridspec_kw={"width_ratios": [1, 1]})

    # (a) the paired scatter, now with enough points to mean something
    ax = axes[0]
    lo = min(a.p_k.min(), b.p_k.min()) * 100 - 0.4
    hi = max(a.p_k.max(), b.p_k.max()) * 100 + 0.4
    ax.plot([lo, hi], [lo, hi], color=fs.RULE, linewidth=0.8, zorder=2)
    ax.axhline(GAMMA * 100, color=fs.INK, linewidth=0.7, zorder=1)
    ax.axvline(GAMMA * 100, color=fs.INK, linewidth=0.7, zorder=1)
    ten = a.key_n < 10
    ax.scatter(a.p_k[~ten] * 100, b.p_k[~ten] * 100, s=9, color=fs.NEUTRAL,
               zorder=3, label=f"the other {int((~ten).sum())} keys")
    ax.scatter(a.p_k[ten] * 100, b.p_k[ten] * 100, s=24, color=fs.CAT[1],
               marker="s", zorder=4, edgecolor="white", linewidth=0.4,
               label="the 10 of Section IV-E")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("$p_k$ under the SmolLM2 vocabulary (%)")
    ax.set_ylabel("$p_k$ under the Qwen2.5 vocabulary (%)")
    ax.legend(loc="upper left", fontsize=6.9, handletextpad=0.3)
    fs.title(ax, f"(a)  Paired over {len(a)} keys, $\\rho = {rho:+.2f}$ "
                 f"[{ci[0]:+.2f}, {ci[1]:+.2f}]")
    fs.style(ax, grid="both")

    # (b) what ten keys could have told you: the interval, as a function of n
    ax = axes[1]
    ns = np.arange(5, len(a) + 1)
    half = [np.tanh(1.96 / np.sqrt(n - 3)) for n in ns]
    ax.plot(ns, half, color=fs.CAT[0], linewidth=1.4, zorder=3)
    for n_mark, lab in ((10, "ten keys"), (len(a), f"{len(a)} keys")):
        h = np.tanh(1.96 / np.sqrt(n_mark - 3))
        ax.scatter([n_mark], [h], s=26, color=fs.CAT[1], zorder=4,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(f"{lab}: ±{h:.2f}", (n_mark, h), xytext=(6, 6),
                    textcoords="offset points", fontsize=7.0, color=fs.MUTED)
    ax.set_xlabel("Keys compared")
    ax.set_ylabel("Half-width of the 95% interval on $\\rho$")
    ax.set_ylim(0, 1.02)
    fs.title(ax, "(b)  Why ten keys could not answer this")
    fs.style(ax)

    fig.tight_layout(w_pad=2.4, rect=(0, 0.06, 1, 0.9))
    fig.text(0.008, 0.985, "The key ranking across two vocabularies, powered",
             ha="left", va="top", fontsize=9.4, color=fs.INK)
    fig.text(0.008, 0.045,
             f"Identical human-written passages scored at {L} tokens under both "
             "vocabularies with the same 128 KGW keys and watermark parameters; "
             "passages are filtered on joint eligibility so the comparison is paired. "
             "Detector-only.", ha="left", va="top", fontsize=6.9, color=fs.MUTED)
    fs.save(fig, out)


def main() -> int:
    smol, qwen = load("smollm2"), load("qwen")
    lengths = sorted(set(smol.length) & set(qwen.length))
    out = {"scope": ("paired cross-vocabulary comparison of the null green rate over a "
                     "wide key sample; identical passages, identical lengths, "
                     "detector-only"),
           "n_keys": int(smol.key_n.nunique()), "by_length": {}}

    for L in lengths:
        a = smol[smol.length == L].sort_values("key_n").reset_index(drop=True)
        b = qwen[qwen.length == L].sort_values("key_n").reset_index(drop=True)
        assert (a.key_n.values == b.key_n.values).all(), "key sets differ"
        if int(a.n_texts.iloc[0]) != int(b.n_texts.iloc[0]):
            raise SystemExit(f"unpaired at {L}: {a.n_texts.iloc[0]} vs {b.n_texts.iloc[0]}")
        rho, p = stats.spearmanr(a.p_k, b.p_k)
        pear, _ = stats.pearsonr(a.p_k, b.p_k)
        ci = spearman_ci(float(rho), len(a))
        out["by_length"][int(L)] = {
            "n_texts": int(a.n_texts.iloc[0]), "n_keys": int(len(a)),
            "spearman": float(rho), "spearman_ci95": list(ci), "spearman_p": float(p),
            "pearson": float(pear),
            "smollm2_p_k_range": [float(a.p_k.min()), float(a.p_k.max())],
            "qwen_p_k_range": [float(b.p_k.min()), float(b.p_k.max())],
            "smollm2_spread": float(a.p_k.max() - a.p_k.min()),
            "qwen_spread": float(b.p_k.max() - b.p_k.min()),
        }

    L = min(lengths)  # every passage is scored here, so this cell is best powered
    a = smol[smol.length == L].sort_values("key_n").reset_index(drop=True)
    b = qwen[qwen.length == L].sort_values("key_n").reset_index(drop=True)
    rho = out["by_length"][int(L)]["spearman"]
    ci = out["by_length"][int(L)]["spearman_ci95"]
    out["headline_length"] = int(L)
    out["ten_key_interval_halfwidth"] = float(np.tanh(1.96 / np.sqrt(10 - 3)))
    out["sweep_interval_halfwidth"] = float(np.tanh(1.96 / np.sqrt(len(a) - 3)))
    json.dump(out, open(os.path.join(SRC, "vocabulary-sweep-summary.json"), "w"), indent=2)
    figure(a, b, L, rho, ci, os.path.join(FIG, "fig8-vocabulary-sweep.pdf"))

    lines = ["# Paired cross-vocabulary key comparison, 128 keys (generated)\n",
             "Source: `validation/analyse_phase2_vocabulary_sweep.py`. Identical human",
             "passages scored under both vocabularies with the same 128 KGW keys;",
             "candidates filtered on joint eligibility so the comparison is paired.\n",
             "| Length | Keys | Texts | Spearman $\\rho$ | 95% CI | Pearson $r$ | SmolLM2 spread | Qwen spread |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for L2 in lengths:
        d = out["by_length"][int(L2)]
        lines.append(
            f"| {L2} | {d['n_keys']} | {d['n_texts']:,} | {d['spearman']:+.2f} | "
            f"[{d['spearman_ci95'][0]:+.2f}, {d['spearman_ci95'][1]:+.2f}] | "
            f"{d['pearson']:+.2f} | {d['smollm2_spread']:.4f} | {d['qwen_spread']:.4f} |")
    open(os.path.join(ROOT, "paper", "phase2-vocabulary-sweep-tables.md"), "w").write(
        "\n".join(lines) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
