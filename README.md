# GARCHPINN / EGARCHPINN

Physics-informed Realized GARCH volatility models for FTSE 100 5-minute data. The authoritative
specification (every equation, algorithm and pitfall) is `docs/realized_garch_reference.tex`;
source papers are in `papers/` (see `papers/README.md`).

## Models

Both models filter returns `r_t = mu + sqrt(h_t) z_t` jointly with a realized measure `x_t`.
The neural network adds an F_{t-1}-measurable correction `g_{t-1} = s * tanh(NN(f_{t-1}, z_{t-1}, log h_{t-1}))`
inside the log-variance recursion, zero-initialised so training starts from the econometric model.

**GARCHPINN** (log-linear Realized GARCH, Hansen, Huang & Shek 2012)

    log h_t = omega + beta log h_{t-1} + gamma log x_{t-1} + g_{t-1}
    log x_t = xi + phi log h_t + tau(z_t) + u_t,   tau(z) = tau1 z + tau2 (z^2 - 1),  u_t ~ N(0, sigma_u^2)

**EGARCHPINN** (Realized EGARCH, Hansen & Huang 2016)

    log h_t = omega + beta log h_{t-1} + tau(z_{t-1}) + gamma u_{t-1} + g_{t-1}
    log x_t = xi + phi log h_t + delta(z_t) + u_t

Parameters are fitted by the joint quasi-likelihood (doc eq. `ll-joint`) of returns and the measurement equation.

## Four configurations

| model  | freq  | returns / measure                                                  |
|--------|-------|--------------------------------------------------------------------|
| garch  | daily | open-to-close returns (%), RV = sum of squared 5-min returns       |
| egarch | daily | same                                                               |
| garch  | 5min  | 5-min returns (bp), intraday periodicity removed, Parkinson range  |
| egarch | 5min  | same                                                               |

## Training (two stages, doc Algorithm 2)

1. **Stage 1**: QMLE of the pure Realized (E)GARCH with L-BFGS, NN off. A copy is kept as the `pure_model` benchmark.
2. **Stage 2**: PINN fine-tuning with Adam and truncated BPTT, state carried across chunks over the full sample
   (no rolling re-estimation). Loss = joint NLL + `lambda_m` * moment conditions on (z, u) + `lambda_g` * mean(g^2)
   + weight decay. Early stopping on validation NLL. Penalty weights are tuned on the validation split only.

Splits are chronological 70/15/15 (train/val/test), never shuffled.

## Evaluation

Out-of-sample on the test split at several horizons (daily 1/5/22, 5-min 1/12/102): log-scale QLIKE
(Patton 2011) and MSE against the realized measure, versus the stage-1 pure model and benchmarks (HAR etc.).
Multi-step forecasts are analytic (with Jensen / psi corrections) and checked by Monte Carlo.

Relation to GINN (Xu et al. 2024, arXiv:2410.00288): we keep their idea of staying close to GARCH
(zero-init correction, `lambda_g` shrinkage), but not their MSE-to-GARCH loss, BatchNorm, ReLU output or 90-day rolling windows.

## Run

    pip install -r requirements.txt
    python main.py                                   # all 4 configurations
    python main.py --model egarch --freq daily --epochs 50
    python main.py --no-nn                           # pure econometric models
    python main.py --data UK_Index_2000_2025.csv --out results --wandb

Per configuration `results/<name>/` holds `model.pt/.json`, `pure_model.pt/.json`, `<name>.png`
(training curves, filtered sqrt h vs sqrt x with split, forecast fan, test QLIKE by horizon) and `summary.json`;
`results/summary.json` collects the metrics of all runs.

## Running ablations

`ablate.py` runs the ablation design of the reference doc ("Ablation study design"): each variant in
`garchpinn/ablation.py:VARIANTS` (baseline, no_nn, lambda_m0, lambda_g0, lambda_orth0, no_state, no_har,
hidden8, hidden128, chunk32, chunk512, reinit_state, tau0, partial_lik) x model x freq x seed.
garch vs egarch and daily vs 5-min are the run axes.

    python ablate.py --data UK_Index_2000_2025.csv                # everything, 5 seeds
    python ablate.py --variants baseline,tau0 --models egarch --freqs daily --seeds 3 --epochs 50
    python ablate.py --grid --variants baseline                   # lambda_m x lambda_g x lambda_orth grid (P2)
    python ablate.py --quick --data fake.csv --out abl_smoke      # tiny smoke run

Outputs in `--out` (default `results/ablation`): `runs.jsonl` (resumable; finished runs are skipped),
`results.csv`/`results.json` (one row per run: test l_r/l_rx, QLIKE/MSE per horizon, params `p_*` and
stage-1 params `s1_*`, val NLL, runtime), `losses/*.npy` (per-t test QLIKE, float32), plots P1-P5
(P4 is the paired DM matrix with Holm-adjusted p-values) and `summary.md` (mean +- 95% seed t-interval).
Holm is applied within each DM matrix only. Across the whole study, control multiplicity as the doc describes.

## Layout

    garchpinn/config.py      Config dataclass
    garchpinn/data.py        loading, realized measures, Dataset
    garchpinn/models/        RealizedPINN base, GARCHPINN, EGARCHPINN, build_model
    garchpinn/losses.py      NLL, moment and shrinkage penalties
    garchpinn/train.py       two-stage fitting
    garchpinn/benchmarks.py  benchmark forecasters
    garchpinn/evaluate.py    out-of-sample evaluation
    garchpinn/plots.py       figures
    garchpinn/utils.py       seeds, save/load
    docs/                    reference document (LaTeX)
    papers/                  source papers
