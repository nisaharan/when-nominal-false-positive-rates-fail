"""Does a larger vocabulary dampen the key effect? Same text, two tokenizers.

The paper shows that each detector key induces its own null green rate on real
text, and that a Qwen2.5 run fails different keys than the SmolLM2 run. That
comparison changes the model and the tokenizer at once, so it cannot say which
one decides. A natural objection is also that a bigger vocabulary might average
the effect away.

This script separates the two. It scores the identical human-written passages
under two vocabularies, SmolLM2-135M at about 49k tokens and Qwen2.5-0.5B at
about 152k, holding the ten KGW hashing keys, the watermark parameters and the
text fixed. Whatever differs is the vocabulary alone. No text is generated and
no model weights are loaded.

Inputs:
    results/phase2-human-null/human-scores.csv[.gz], human-texts.csv
    results/phase2-human-null-qwen/human-scores.csv[.gz], human-texts.csv
Outputs:
    results/phase2-human-null-qwen/vocabulary-cells.csv
    results/phase2-human-null-qwen/vocabulary-summary.json
    paper/figures/fig7-vocabulary.pdf
    paper/phase2-vocabulary-tables.md

Usage (repo root): python validation/analyse_phase2_vocabulary.py
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
SMOL = os.path.join(ROOT, "results", "phase2-human-null")
QWEN = os.path.join(ROOT, "results", "phase2-human-null-qwen")
FIG = os.path.join(ROOT, "paper", "figures")
Z = stats.norm.ppf(0.99)
GAMMA = 0.25
LENGTHS = (128, 256, 512)
VOCAB = {"SmolLM2": "SmolLM2-135M, 49k tokens", "Qwen2.5": "Qwen2.5-0.5B, 152k tokens"}


def spearman_ci(rho, n, conf=0.95):
    """Fisher-z interval. Ten keys is a small sample and the interval says so;
    without it a non-significant rho reads as an established zero."""
    if n < 4:
        return float("nan"), float("nan")
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    z = np.arctanh(rho)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def cp(x, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if x == 0 else stats.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def load(folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = os.path.join(folder, "human-scores.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(folder, "human-scores.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing scores under {folder}")
    scores = pd.read_csv(p)
    texts = pd.read_csv(os.path.join(folder, "human-texts.csv"))
    return scores[scores.scheme == "kgw"], texts


def cells(scores: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for (key, L), g in scores.groupby(["key_id", "length"]):
        n, x = len(g), int((g.value > Z).sum())
        lo, hi = cp(x, n)
        green, elig = int(g.green_tokens.sum()), int(g.eligible_positions.sum())
        plo, phi = cp(green, elig)
        rows.append(dict(vocabulary=label, key_id=key, length=int(L), n=n, exceed=x,
                         fpr=x / n, fpr_lo=lo, fpr_hi=hi,
                         p_k=green / elig, p_lo=plo, p_hi=phi,
                         mean_z=g.value.mean(), sd_z=g.value.std(ddof=1)))
    return pd.DataFrame(rows)


def figure(c: pd.DataFrame, L: int, rho: float, ci: tuple, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.95))
    keys = sorted(c.key_id.unique())
    x = np.arange(len(keys))

    # (a) the spread of p_k under each vocabulary, on identical text
    ax = axes[0]
    for i, vocab in enumerate(VOCAB):
        g = c[(c.vocabulary == vocab) & (c.length == L)].set_index("key_id").loc[keys]
        ax.vlines(x + (i - 0.5) * 0.34, g.p_lo, g.p_hi, color=fs.CAT[i], linewidth=1.1)
        ax.scatter(x + (i - 0.5) * 0.34, g.p_k, s=20, color=fs.CAT[i], marker=fs.CAT_MARKER[i],
                   zorder=3, edgecolor="white", linewidth=0.5, label=VOCAB[vocab])
    ax.axhline(GAMMA, color=fs.INK, linewidth=0.9, zorder=2)
    ax.text(len(keys) - 0.45, GAMMA + 0.0015, "assumed $\\gamma = 0.25$", ha="right",
            va="bottom", fontsize=7.2, color=fs.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("kgw-", "") for k in keys])
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xlabel("Frozen detector key")
    ax.set_ylabel(f"Null green rate $p_k$ ({L} tokens)")
    ax.legend(loc="upper left", fontsize=7)
    fs.title(ax, "(a)  Same passages, two vocabularies")
    fs.style(ax)

    # (b) does the ranking transfer?
    ax = axes[1]
    a = c[(c.vocabulary == "SmolLM2") & (c.length == L)].set_index("key_id").loc[keys]
    b = c[(c.vocabulary == "Qwen2.5") & (c.length == L)].set_index("key_id").loc[keys]
    lim = [min(a.p_lo.min(), b.p_lo.min()) - 0.006, max(a.p_hi.max(), b.p_hi.max()) + 0.006]
    ax.plot(lim, lim, color=fs.MUTED, linewidth=0.8, zorder=2)
    ax.axhline(GAMMA, color=fs.INK, linewidth=0.7, zorder=1)
    ax.axvline(GAMMA, color=fs.INK, linewidth=0.7, zorder=1)
    for key in keys:
        ra, rb = a.loc[key], b.loc[key]
        col = fs.POS if ra.p_k > GAMMA else fs.NEG
        ax.plot([ra.p_lo, ra.p_hi], [rb.p_k, rb.p_k], color=col, linewidth=0.9, zorder=3)
        ax.plot([ra.p_k, ra.p_k], [rb.p_lo, rb.p_hi], color=col, linewidth=0.9, zorder=3)
        ax.scatter([ra.p_k], [rb.p_k], s=20, color=col, zorder=4, edgecolor="white",
                   linewidth=0.5, marker="o" if col == fs.POS else "s")
        ax.annotate(key.replace("kgw-", ""), (ra.p_k, rb.p_k), xytext=(4, 3.5),
                    textcoords="offset points", fontsize=6.8, color=fs.MUTED)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("$p_k$ under the SmolLM2 vocabulary")
    ax.set_ylabel("$p_k$ under the Qwen2.5 vocabulary")
    # State rho with its interval. With ten keys the interval is wide, so the
    # caption must not report a non-significant correlation as a proven zero.
    lo, hi = ci
    verdict = ("The ranking transfers" if lo > 0.3 else
               "No evidence the ranking transfers")
    fs.title(ax, f"(b)  {verdict}, $\\rho = {rho:+.2f}$ [{lo:+.2f}, {hi:+.2f}]")
    fs.style(ax, grid="both")
    fig.tight_layout()
    fs.save(fig, out)


def main() -> int:
    s_scores, s_texts = load(SMOL)
    q_scores, q_texts = load(QWEN)

    # the two runs select passages by the same hashed rank, but eligibility
    # depends on the tokenizer, so restrict to the passages both scored
    key_cols = ["source_row", "field"]
    shared = s_texts.merge(q_texts, on=key_cols, suffixes=("_s", "_q"))[
        key_cols + ["prompt_id_s", "prompt_id_q"]]
    s_scores = s_scores[s_scores.prompt_id.isin(set(shared.prompt_id_s))]
    q_scores = q_scores[q_scores.prompt_id.isin(set(shared.prompt_id_q))]

    # Eligibility at each prefix length is itself tokenizer-dependent: a passage
    # that reaches 512 tokens under the 49k vocabulary can fall short under the
    # 152k one, which would leave the longer cells comparing different text and
    # reintroduce the confound this experiment removes. Restrict every length to
    # the passages both runs scored at that length.
    s_to_q = dict(zip(shared.prompt_id_s, shared.prompt_id_q))
    s_at = s_scores.groupby("length").prompt_id.agg(set).to_dict()
    q_at = q_scores.groupby("length").prompt_id.agg(set).to_dict()
    s_keep, q_keep = set(), set()
    for L in LENGTHS:
        both = {p for p in s_at.get(L, set()) if s_to_q.get(p) in q_at.get(L, set())}
        s_keep |= {(p, L) for p in both}
        q_keep |= {(s_to_q[p], L) for p in both}
    s_scores = s_scores[[k in s_keep for k in zip(s_scores.prompt_id, s_scores.length)]]
    q_scores = q_scores[[k in q_keep for k in zip(q_scores.prompt_id, q_scores.length)]]

    c = pd.concat([cells(s_scores, "SmolLM2"), cells(q_scores, "Qwen2.5")], ignore_index=True)
    c.to_csv(os.path.join(QWEN, "vocabulary-cells.csv"), index=False)

    L = max(l for l in LENGTHS if c[c.length == l].n.min() >= 200)
    a = c[(c.vocabulary == "SmolLM2") & (c.length == L)].set_index("key_id")
    b = c[(c.vocabulary == "Qwen2.5") & (c.length == L)].set_index("key_id")
    keys = sorted(set(a.index) & set(b.index))
    rho, pval = stats.spearmanr(a.loc[keys].p_k, b.loc[keys].p_k)
    ci = spearman_ci(float(rho), len(keys))
    figure(c, L, float(rho), ci, os.path.join(FIG, "fig7-vocabulary.pdf"))

    def spread(vocab, length):
        g = c[(c.vocabulary == vocab) & (c.length == length)]
        return [float(g.p_k.min()), float(g.p_k.max())]

    summary = {
        "scope": ("identical human passages scored under two vocabularies with the same ten "
                  "KGW keys and watermark parameters; detector-only, nothing generated"),
        "shared_passages": int(len(shared)),
        "comparison_length": int(L),
        "n_by_length": {int(l): int(c[c.length == l].n.min()) for l in LENGTHS
                        if (c.length == l).any()},
        "p_k_range": {v: {int(l): spread(v, l) for l in LENGTHS if (c.length == l).any()}
                      for v in VOCAB},
        "p_k_spread_width": {v: {int(l): round(spread(v, l)[1] - spread(v, l)[0], 4)
                                 for l in LENGTHS if (c.length == l).any()} for v in VOCAB},
        "spearman_p_k_across_vocabularies": float(rho),
        "spearman_ci95": list(ci),
        "spearman_n_keys": len(keys),
        "spearman_p": float(pval),
        "spearman_note": ("ten keys give a wide interval; this is an absence of "
                          "strong transfer, not an established zero"),
        "worst_key": {v: str(c[(c.vocabulary == v) & (c.length == L)].set_index("key_id").p_k.idxmax())
                      for v in VOCAB},
        "fpr_range_at_nominal": {v: [float(c[(c.vocabulary == v) & (c.length == L)].fpr.min()),
                                     float(c[(c.vocabulary == v) & (c.length == L)].fpr.max())]
                                 for v in VOCAB},
        "cells_above_1pct": {v: int((c[c.vocabulary == v].fpr > 0.01).sum()) for v in VOCAB},
    }
    json.dump(summary, open(os.path.join(QWEN, "vocabulary-summary.json"), "w"), indent=2)

    lines = ["# Vocabulary comparison on identical human text (generated)\n",
             "Source: `validation/analyse_phase2_vocabulary.py`. The same human-written Dolly",
             "passages scored under two tokenizers with the same ten KGW hashing keys and the",
             "same watermark parameters. Detector-only; no text generated and no weights loaded.",
             f"Restricted to the {len(shared):,} passages both runs scored.\n",
             "| Vocabulary | Length | n | $p_k$ range | FPR range at nominal $z>2.326$ |",
             "|---|---:|---:|---:|---:|"]
    for v in VOCAB:
        for L2 in LENGTHS:
            g = c[(c.vocabulary == v) & (c.length == L2)]
            if g.empty:
                continue
            lines.append(f"| {v} | {L2} | {int(g.n.min()):,} | {g.p_k.min():.3f}-{g.p_k.max():.3f} | "
                         f"{g.fpr.min()*100:.2f}%-{g.fpr.max()*100:.2f}% |")
    lines.append(f"\nSpearman correlation of $p_k$ across the two vocabularies at {L} tokens: "
                 f"{rho:+.2f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] (p = {pval:.2g}) over "
                 f"{len(keys)} keys. The interval is wide because ten keys is a small "
                 f"sample: this is an absence of strong transfer, not an established zero.")
    open(os.path.join(ROOT, "paper", "phase2-vocabulary-tables.md"), "w").write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
