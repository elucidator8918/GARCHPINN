"""Ablation study driver (docs/realized_garch_reference.tex, "Ablation study design").

    python ablate.py                                    # all variants, both models and freqs, 5 seeds
    python ablate.py --variants baseline,tau0 --models egarch --freqs daily --seeds 3
    python ablate.py --grid --variants baseline         # adds the lambda_m x lambda_g x lambda_orth grid (P2)
    python ablate.py --quick --data fake.csv --out abl_smoke

Resumable: runs already in <out>/runs.jsonl are skipped. Outputs: results.csv/json (one row
per run), losses/*.npy (per-t test QLIKE, float32), P1-P5 PNGs and summary.md.
"""
import argparse
import os

import torch

torch.set_default_dtype(torch.float64)

from garchpinn.ablation import VARIANTS, all_variants, make_plots, run_ablation, summary_table  # noqa: E402
from garchpinn.config import FREQS, MODELS  # noqa: E402


def _list(s):
    return [x for x in s.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="all", help="comma list or 'all'; known: " + ",".join(VARIANTS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--freqs", default=",".join(FREQS))
    ap.add_argument("--seeds", default="5", help="N (seeds 0..N-1) or a comma list")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--grid", action="store_true", help="add the lambda grid variants (P2)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: daily, 2 seeds, 3 epochs, 30 L-BFGS iters, tiny MC")
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args(argv)

    base = {}
    if a.quick:
        base.update(stage1_iters=30, n_paths=100, patience=3)
        if a.epochs is None:
            a.epochs = 3
        if a.freqs == ",".join(FREQS):
            a.freqs = "daily"
        if a.seeds == "5":
            a.seeds = "2"
    reg = all_variants(a.grid)
    variants = list(VARIANTS) if a.variants == "all" else _list(a.variants)
    if a.grid:
        variants += [v for v in reg if v.startswith("grid_") and v not in variants]
    bad = [v for v in variants if v not in reg]
    if bad:
        ap.error(f"unknown variants {bad}")
    seeds = list(range(int(a.seeds))) if "," not in a.seeds else [int(s) for s in _list(a.seeds)]

    df, rows = run_ablation(variants, _list(a.models), _list(a.freqs), seeds, a.out, a.data,
                            a.epochs, base, a.grid)
    for p in make_plots(df, rows, a.out, a.dpi):
        print("wrote", p)
    md = summary_table(df)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    return df


if __name__ == "__main__":
    main()
