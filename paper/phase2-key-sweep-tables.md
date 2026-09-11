# Key-space sweep (generated)

`keys > 1%` counts point estimates; `certain` keeps only keys whose own
exact 95% lower bound also clears 1%.

Source: `validation/analyse_phase2_key_sweep.py`. Detector-only KGW scoring
of human-written Dolly passages under keys from the same schedule as the
paper's ten, which are conditions 0-9 of this sweep.

| Length | Keys | Texts | $p_k$ range | median $p_k$ | FPR range | keys > 1% | certain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1,500 | 0.217-0.315 | 0.248 | 0.13%-30.00% | 88/128 (69%) | 68/128 (53%) |
| 256 | 128 | 588 | 0.214-0.317 | 0.248 | 0.00%-53.74% | 90/128 (70%) | 58/128 (45%) |
| 512 | 128 | 179 | 0.216-0.319 | 0.248 | 0.00%-82.12% | 91/128 (71%) | 57/128 (45%) |
