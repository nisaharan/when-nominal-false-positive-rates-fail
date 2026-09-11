# When nominal false-positive rates fail

Measuring how far the KGW watermark detector's assumed null is from real text,
and why the gap depends on which key you hold.

**Paper:** [`paper/when-nominal-false-positive-rates-fail.pdf`](paper/when-nominal-false-positive-rates-fail.pdf)
· source in [`paper/arxiv/`](paper/arxiv/)
**Author:** Nisaharan Genhatharan, UNSW Sydney

---

## The finding

A KGW green-list detector reports a $z$-score. Its operating point rests on one
assumption: that in unwatermarked text each scored token is green with
probability exactly $\gamma$, independently of the others. Under that
assumption $z > 2.326$ is a one-sided 1% test, and that 1% is quoted widely as
though it were a property of the scheme.

Scored against 10,000 unwatermarked SmolLM2-135M outputs with ten frozen
detector keys, it is not a property of the scheme. It is a property of the key.

| At 512 tokens, nominal $z > 2.326$ | Empirical false-positive rate |
|---|---|
| Best key (03) | 0.27% |
| Median across ten keys | 6.9% |
| Worst key (07) | 48.4%, exact 95% CI [47.4, 49.3] |

Twenty of thirty key-by-length cells exceed 1%, and for the inflated keys the
rate grows with text length rather than shrinking.

**The mechanism is one number per key.** Each key colours a different fraction
$p_k$ of real text green, from 0.205 to 0.302 against the assumed 0.25.
Substituting $p_k$ for $\gamma$ in the detector's own formula predicts the mean
null $z$ of all thirty cells with $r = 0.9999$, and the resulting bias grows as
$\sqrt{T}$ while the assumed standard deviation does not.

**The bias belongs to the key and the vocabulary, not to the model.** Scoring
5,000 human-written passages with the same keys reproduces the ranking of $p_k$
(Spearman $\rho = +0.94$) and the same worst key. A small model amplifies the
effect by concentrating its output on the frequent context-token pairs the key
already favours, but it does not create it. Which key fails is decided by the
tokenizer: rescoring the same passages under Qwen2.5's three-times-larger
vocabulary, with keys, parameters and text held fixed, does not narrow the
spread of $p_k$, and over 128 keys the ranking does not survive
($\rho = +0.01$ [-0.16, +0.18]). A rate measured under one tokenizer does not
transfer to another.

**Per-key, per-length calibration fixes it, and it is affordable.** Thresholds
fitted once on 5,000 outputs and evaluated once on 5,000 disjoint outputs hold
every one of sixty cells between 0.16% and 0.96%. On watermarked text that costs
a median two points of detection at 512 tokens. For key 07 the false-positive
rate falls from 48.4% to 0.54% while detection falls from 98% to 96%. The
correction runs both ways: for the three cells where calibration lowers the
threshold, detection improves.

SynthID-Text, measured against a naive-independence reference because its
mean-$g$ detector has no nominal $z$, stays between 0.74% and 1.58% with no
key-specific bias. That is reported as a reference, not as a failure of that
scheme.

## What this does not claim

Findings apply to the named variants and settings, not to everything called KGW
or SynthID. The primary corpus is one small model on English prompts; the size
of the effect will differ elsewhere, though the mechanism does not depend on
model size. No robustness, attack, or paraphrase result is claimed. The detection rates are
a development screen of 50 outputs per cell, reported to price calibration
rather than to claim a detection result.

A watermark detector score is evidence of compatibility with a specified
embedding and detection process. It is not proof of authorship, and should not
be used alone in disciplinary, employment, legal, or authorship decisions.

## Reproducing the paper

Every table and figure is generated from stored detector scores by a fixed
script. Nothing needs to be regenerated with a language model: the three
compressed score files and the aggregated key-sweep cells are in the
repository, and the analyses read token ids and scores, never new text.

```bash
uv sync --extra ml --group dev

python validation/analyse_phase2_nominal_fpr.py         # Figures 1-5, Table 1
python validation/analyse_phase2_human_null.py          # Figure 6, Table 2
python validation/analyse_phase2_vocabulary.py          # Figure 7
python validation/analyse_phase2_vocabulary_sweep.py    # Figure 8
python validation/analyse_phase2_key_sweep.py           # Figure 9
python validation/analyse_phase2_detection_tradeoff.py  # Figure 10
python validation/build_phase2_publication_figures.py   # Figure 11
python validation/build_phase2_v1_latex_tables.py       # LaTeX tables 1, 2, A1
python -m pytest -q

cd paper/arxiv && tectonic main.tex                     # or latexmk -pdf
```

This was checked by cloning the repository and rebuilding the paper from the
clone alone.

## Layout

| Path | What is in it |
|---|---|
| `paper/` | Manuscript, generated figures and tables, built PDF |
| `src/ai_watermarks_phase2/` | Watermark adapters, generation and scoring harness, protocol machinery |
| `validation/` | One script per analysis, each with a fixed table, figure or JSON output |
| `configs/` | Frozen protocol definitions: keys, lengths, seeds, acceptance rules |
| `data/` | Prompt selection manifests with source hashes |
| `results/` | The reported package: aggregate artifacts, the key-sweep cells and the three compressed score files |
| `reports/` | Reviewed report artifacts behind the confirmation-gate figure |
| `docs/research-transformation/` | Protocols, decision memos and analysis reports, phases 0 to 2 |
| `tests/` | Estimator, protocol and analysis tests |

The scripts that build the paper are the eight named above. Regenerating the
stored scores and sweep cells themselves is documented in `paper/README.md`.
`docs/research-transformation/phase-2/nominal-fpr-report.md` is the analysis
record behind Sections 4 and 5, with every number quoted in the paper.

The remaining scripts and configs are the record of development studies that
support no claim in the paper: a positive-sensitivity screen, two KGW bias
brackets, and a joint parameter study that was abandoned before its first stage
was evaluated. They are published because the paper refers to them and because
a reader should be able to see what was tried, not because they carry a result.

## What is deliberately not here

Raw generated continuations are kept offline. They have not been reviewed for
memorised personal, copyrighted or unsafe content, and publishing 10,000 model
outputs is not needed to check any claim in the paper. What is published
instead is every aggregate the claims rest on, plus the per-text tables, which
carry token statistics and a SHA-256 per text rather than the text itself.

Model weights and cached datasets are not redistributed. Every one is pinned by
revision in `configs/`, so retrieval is deterministic.

The development runs that support no claim are kept offline for audit for the
same reason: they are large, and the paper explicitly rests nothing on them.

## Licence

Source code is MIT ([`LICENSE`](LICENSE)). Prose, figures and derived tables are
CC BY-SA 4.0 ([`LICENSE-DOCS`](LICENSE-DOCS)), because the analysis is derived
from CC BY-SA material. Attribution for every borrowed dataset, model and
reference implementation is in [`NOTICE.md`](NOTICE.md).
