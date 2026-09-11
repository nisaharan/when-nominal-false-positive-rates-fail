# Vocabulary comparison on identical human text (generated)

Source: `validation/analyse_phase2_vocabulary.py`. The same human-written Dolly
passages scored under two tokenizers with the same ten KGW hashing keys and the
same watermark parameters. Detector-only; no text generated and no weights loaded.
Restricted to the 4,857 passages both runs scored.

| Vocabulary | Length | n | $p_k$ range | FPR range at nominal $z>2.326$ |
|---|---:|---:|---:|---:|
| SmolLM2 | 128 | 4,857 | 0.217-0.282 | 0.21%-9.47% |
| SmolLM2 | 256 | 1,875 | 0.217-0.282 | 0.16%-16.27% |
| SmolLM2 | 512 | 511 | 0.215-0.280 | 0.59%-25.83% |
| Qwen2.5 | 128 | 4,857 | 0.228-0.310 | 0.37%-26.21% |
| Qwen2.5 | 256 | 1,875 | 0.228-0.309 | 0.64%-44.27% |
| Qwen2.5 | 512 | 511 | 0.229-0.306 | 0.59%-67.71% |

Spearman correlation of $p_k$ across the two vocabularies at 512 tokens: -0.15, 95% CI [-0.71, +0.53] (p = 0.68) over 10 keys. The interval is wide because ten keys is a small sample: this is an absence of strong transfer, not an established zero.
