# Paired cross-vocabulary key comparison, 128 keys (generated)

Source: `validation/analyse_phase2_vocabulary_sweep.py`. Identical human
passages scored under both vocabularies with the same 128 KGW keys;
candidates filtered on joint eligibility so the comparison is paired.

| Length | Keys | Texts | Spearman $\rho$ | 95% CI | Pearson $r$ | SmolLM2 spread | Qwen spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 800 | +0.01 | [-0.16, +0.18] | +0.01 | 0.0975 | 0.0998 |
| 256 | 128 | 313 | +0.01 | [-0.16, +0.19] | +0.01 | 0.1014 | 0.0990 |
| 512 | 128 | 93 | -0.02 | [-0.19, +0.15] | -0.02 | 0.1066 | 0.0977 |
