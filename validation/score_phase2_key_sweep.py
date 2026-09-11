"""How much of the key space inflates? The ten-key result, widened.

Section IV-B measures ten frozen keys and finds the nominal 1% threshold spans
0.27% to 48.4%. Ten keys establish that the effect exists and that its ranking is
stable; they cannot say what fraction of the key space is affected, and they leave
the vocabulary Spearman in Section IV-E with a 95% interval of [-0.71, +0.53].

This sweeps the same derivation to many more keys. `derive_schedule` is
deterministic, so conditions 0-9 reproduce the paper's ten exactly and the sweep
is a strict superset of them.

Detector-only: KGW alone, no SynthID, no model weights, nothing generated. The
same human-written Dolly passages as Section IV-D, selected by the same hashed
seed, so the corpus is identical to the one already in the paper.

Rows are aggregated as they are scored rather than written per score, because
n_keys x n_texts x 3 lengths individual rows is millions of lines that no
analysis reads.

Output: results/phase2-key-sweep/sweep-cells.csv, run.json
Usage (repo root), the Section IV-F sweep; the paired Section IV-E runs are in
paper/README.md:
    .venv/bin/python validation/score_phase2_key_sweep.py --keys 128 --max-texts 1500
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_watermarks_phase2 import canonical  # noqa: E402
from ai_watermarks_phase2.key_schedule import derive_kgw_key  # noqa: E402
from ai_watermarks_phase2.native import require_ml_dependencies  # noqa: E402
from ai_watermarks_phase2.smoke import load_json  # noqa: E402
from ai_watermarks_phase2.variance_pilot import CompactKGWScorer  # noqa: E402

LENGTHS = (128, 256, 512)
SELECTION_SEED = "phase2-human-null-v1"   # identical to the Section IV-D corpus
Z_NOMINAL = 2.3263478740408408
FIELDS = ["key_id", "hashing_key", "length", "n_texts", "exceed", "fpr",
          "green_tokens", "eligible_positions", "p_k", "mean_z", "sd_z"]


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--protocol", type=Path,
                    default=ROOT / "configs" / "phase2-confirmatory-null.json")
    ap.add_argument("--keys", type=int, default=128)
    ap.add_argument("--max-texts", type=int, default=1500)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "phase2-key-sweep")
    ap.add_argument("--device", default="cpu")
    # Scoring tokenizer. Overriding it holds the keys, the parameters and the text
    # fixed and varies only the vocabulary, as in Section IV-E.
    ap.add_argument("--tokenizer-id", default=None)
    ap.add_argument("--tokenizer-revision", default=None)
    # A second tokenizer used only to filter candidates. Passing it keeps a
    # passage only when both vocabularies support the length, so two runs with the
    # ids swapped score an identical passage set at an identical set of lengths.
    # Without it the two runs select different passages and the paired comparison
    # is not paired.
    ap.add_argument("--pair-tokenizer-id", default=None)
    ap.add_argument("--pair-tokenizer-revision", default=None)
    args = ap.parse_args(argv)

    protocol = load_json(args.protocol)
    watermark_config = load_json(ROOT / protocol["watermark_config"])
    variant = load_json(ROOT / protocol["key_schedule"]["source_config"])["variants"]["kgw"]

    if args.source:
        source_path = args.source
    else:
        from huggingface_hub import hf_hub_download
        src = protocol["source"]
        source_path = Path(hf_hub_download(repo_id=src["dataset"], filename=src["file"],
                                           repo_type="dataset", revision=src["revision"]))
    source_bytes = source_path.read_bytes()
    sha = hashlib.sha256(source_bytes).hexdigest()
    if sha != protocol["source"]["file_sha256"]:
        raise SystemExit(f"Dolly SHA-256 mismatch: {sha}")
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]

    torch, transformers = require_ml_dependencies()
    model = dict(protocol["model"])
    if args.tokenizer_id:
        if not args.tokenizer_revision:
            raise SystemExit("--tokenizer-id requires --tokenizer-revision: pins are not optional")
        model = {"id": args.tokenizer_id, "revision": args.tokenizer_revision,
                 "device": args.device}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    cfg = transformers.AutoConfig.from_pretrained(model["id"], revision=model["revision"])
    pair_tok = None
    if args.pair_tokenizer_id:
        if not args.pair_tokenizer_revision:
            raise SystemExit("--pair-tokenizer-id requires --pair-tokenizer-revision")
        pair_tok = transformers.AutoTokenizer.from_pretrained(
            args.pair_tokenizer_id, revision=args.pair_tokenizer_revision)
    runner = SimpleNamespace(model=SimpleNamespace(config=cfg), device=args.device,
                             tokenizer=tokenizer)

    # one processor per key, sharing the permutation table as build_scorers does
    keys = [derive_kgw_key(i) for i in range(args.keys)]
    scorers, shared = [], None
    for index, key in enumerate(keys):
        settings = dict(watermark_config["kgw"], hashing_key=int(key))
        keyed = canonical.build_kgw_config(settings, variant)
        proc = keyed.construct_processor(runner.model.config.vocab_size, runner.device)
        if shared is None:
            shared = proc.fixed_table
        else:
            proc.fixed_table = shared
        scorers.append(CompactKGWScorer(
            processor=proc, variant=variant, key_id=f"kgw-{index:03d}",
            hashing_key=int(key),
            greenlist_ratio=float(settings["greenlist_ratio"])))
    print(f"{len(scorers)} KGW keys; conditions 0-9 are the paper's ten", file=sys.stderr)

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for field in ("response", "context"):
            text = str(row.get(field, "") or "").strip()
            if not text:
                continue
            norm = normalized(text)
            if norm in seen:
                continue
            seen.add(norm)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < LENGTHS[0]:
                continue
            usable = len(ids)
            if pair_tok is not None:
                other = len(pair_tok(text, add_special_tokens=False)["input_ids"])
                if other < LENGTHS[0]:
                    continue
                # a length counts only if both vocabularies reach it
                usable = min(usable, other)
            rank = hashlib.sha256(f"{SELECTION_SEED}\0{row_index}\0{field}".encode()).hexdigest()
            candidates.append({"rank": rank, "ids": ids, "usable": usable})
    candidates.sort(key=lambda c: c["rank"])
    selected = candidates[: args.max_texts]
    print(f"{len(candidates)} eligible; scoring {len(selected)}", file=sys.stderr)

    # accumulate per (key, length) instead of emitting a row per score
    n_k, n_l = len(scorers), len(LENGTHS)
    acc = {name: np.zeros((n_k, n_l), dtype=np.float64)
           for name in ("n", "exceed", "green", "elig", "sz", "szz")}
    started = time.time()
    for i, c in enumerate(selected):
        ids = c["ids"]
        lengths = [L for L in LENGTHS if L <= c["usable"]]
        with torch.no_grad():
            for ki, sc in enumerate(scorers):
                res = sc.score_prefixes(ids[: lengths[-1]], lengths)
                for L in lengths:
                    li = LENGTHS.index(L)
                    s = res[L]
                    z = float(s["value"])
                    acc["n"][ki, li] += 1
                    acc["exceed"][ki, li] += z > Z_NOMINAL
                    acc["green"][ki, li] += s.get("green_tokens") or 0
                    acc["elig"][ki, li] += s.get("eligible_positions") or 0
                    acc["sz"][ki, li] += z
                    acc["szz"][ki, li] += z * z
        if (i + 1) % 50 == 0:
            el = time.time() - started
            print(f"  {i + 1}/{len(selected)}  {el / 60:.1f} min, "
                  f"~{el / (i + 1) * (len(selected) - i - 1) / 60:.1f} min left",
                  file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "sweep-cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for ki, sc in enumerate(scorers):
            for li, L in enumerate(LENGTHS):
                n = acc["n"][ki, li]
                if not n:
                    continue
                mean = acc["sz"][ki, li] / n
                var = max(acc["szz"][ki, li] / n - mean * mean, 0.0)
                w.writerow({
                    "key_id": sc.key_id, "hashing_key": sc.hashing_key, "length": L,
                    "n_texts": int(n), "exceed": int(acc["exceed"][ki, li]),
                    "fpr": acc["exceed"][ki, li] / n,
                    "green_tokens": int(acc["green"][ki, li]),
                    "eligible_positions": int(acc["elig"][ki, li]),
                    "p_k": acc["green"][ki, li] / acc["elig"][ki, li],
                    "mean_z": mean, "sd_z": float(np.sqrt(var * n / max(n - 1, 1))),
                })
    run = {
        "schema_version": 1,
        "scope": ("key-space sweep: KGW only, detector-only scoring of human-written "
                  "Dolly text under many derived keys; nothing generated"),
        "protocol": str(args.protocol.relative_to(ROOT)),
        "source": {**protocol["source"], "observed_sha256": sha},
        "tokenizer": model,
        "pair_tokenizer": ({"id": args.pair_tokenizer_id,
                            "revision": args.pair_tokenizer_revision}
                           if pair_tok is not None else None),
        "variant": variant,
        "key_derivation": "ai_watermarks_phase2.key_schedule.derive_kgw_key",
        "n_keys": len(keys),
        "paper_keys_are_prefix": True,
        "selection_seed": SELECTION_SEED,
        "texts_scored": len(selected),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps(run, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
