# Paper package

The manuscript and everything generated for it. All of it is built from stored
detector scores by fixed scripts; no step regenerates text.

The manuscript is set two-column in the IEEEtran conference class, with STIX
Two for both body text and figures.

## Contents

1. `arxiv/main.tex` and `arxiv/references.bib`: the manuscript.
2. `arxiv/tables/`: generated LaTeX fragments (Table 1 nominal false-positive
   rates, Table 2 the human-text null, Table A1 the sixty calibrated cells).
   Generated, so do not edit by hand.
3. `figures/`: generated vector PDFs, numbered as they appear in the paper.
   `figure-captions.md` records which script owns each one, and
   `../validation/figstyle.py` holds the shared style.
4. `phase2-nominal-fpr-tables.md`, `phase2-human-null-tables.md`,
   `phase2-vocabulary-tables.md`, `phase2-vocabulary-sweep-tables.md`,
   `phase2-key-sweep-tables.md`, `phase2-detection-tables.md`,
   `phase2-appendix-table-a1.md`: Markdown twins of the manuscript tables and
   of the vocabulary, key-sweep and detection analyses, for reading outside
   LaTeX.
5. `release-and-license-audit.md`: the release boundary and licence checks.
6. `when-nominal-false-positive-rates-fail.pdf`: the current build.

The analysis record behind Sections 4 and 5, with every number quoted in the
paper, is `../docs/research-transformation/phase-2/nominal-fpr-report.md`.

## Rebuild

From the repository root:

```bash
python validation/analyse_phase2_nominal_fpr.py         # Figures 1-5, Table 1
python validation/analyse_phase2_human_null.py          # Figure 6, Table 2
python validation/analyse_phase2_vocabulary.py          # Figure 7
python validation/analyse_phase2_vocabulary_sweep.py    # Figure 8
python validation/analyse_phase2_key_sweep.py           # Figure 9
python validation/analyse_phase2_detection_tradeoff.py  # Figure 10
python validation/build_phase2_publication_figures.py   # Figure 11
python validation/build_phase2_v1_latex_tables.py       # LaTeX tables
cd paper/arxiv && tectonic main.tex                     # or latexmk -pdf
```

The human-text null reads `results/phase2-human-null/human-scores.csv.gz`,
which is in the repository. Regenerating that file from scratch instead needs
`validation/score_phase2_human_null.py` and about 25 minutes of detector-only
scoring; it loads no model weights.

The vocabulary comparison reads that file and
`results/phase2-human-null-qwen/human-scores.csv.gz`, which is also in the
repository. Regenerating the second one scores the same passages under the
Qwen2.5 tokenizer and took 63 minutes on an M4 Pro:

```bash
python validation/score_phase2_human_null.py --max-texts 5000 \
  --tokenizer-id Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output results/phase2-human-null-qwen
```

It loads the tokenizer and config only, never the Qwen weights.

Figures 8 and 9 read aggregated cells, not score files: 128 keys x 3 lengths
per run, in `results/phase2-key-sweep/` and `results/phase2-key-sweep-paired/`,
both in the repository. Regenerating them needs
`validation/score_phase2_key_sweep.py`, KGW only and detector-only. The 128-key
sweep behind Figure 9 took 89 minutes:

```bash
python validation/score_phase2_key_sweep.py --keys 128 --max-texts 1500
```

The paired sweep behind Figure 8 is two runs over the same 800 passages, one
per tokenizer. Each names the other as `--pair-tokenizer`, so a passage is kept
only at lengths both vocabularies support and the two runs score an identical
passage set. The SmolLM2 leg took 52 minutes and the Qwen2.5 leg about 120:

```bash
python validation/score_phase2_key_sweep.py --keys 128 --max-texts 800 \
  --pair-tokenizer-id Qwen/Qwen2.5-0.5B-Instruct \
  --pair-tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --output results/phase2-key-sweep-paired/smollm2
python validation/score_phase2_key_sweep.py --keys 128 --max-texts 800 \
  --tokenizer-id Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --pair-tokenizer-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --pair-tokenizer-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --output results/phase2-key-sweep-paired/qwen
```

The scorer keeps everything in memory and writes nothing until the last passage
is scored, so an interrupted run leaves no partial output. On a laptop, keep it
awake for the whole run (on macOS, `caffeinate -i -m -w <pid>` with the lid
open).
