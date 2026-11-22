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

Outputs in `--out` (default `results/ablation`): `runs/<run>.json` (one per run, written atomically;
resumable, finished runs are skipped), `cache_<freq>.pt` (parsed data, shared by workers),
`results.csv`/`results.json` (one row per run: test l_r/l_rx, QLIKE/MSE per horizon, params `p_*` and
stage-1 params `s1_*`, val NLL, runtime), `losses/*.npy` (per-t test QLIKE, float32), plots P1-P5
(P4 is the paired DM matrix with Holm-adjusted p-values) and `summary.md` (mean +- 95% seed t-interval).
Holm is applied within each DM matrix only. Across the whole study, control multiplicity as the doc describes.

## Running on a Linux GPU machine

    git clone -b realized-garch-rewrite <repo-url> GARCHPINN && cd GARCHPINN
    scp mac:~/GARCHPINN-1/UK_Index_2000_2025.csv .        # the CSV is gitignored: copy it by hand
    CUDA=cu124 bash scripts/setup_linux.sh                 # venv, CUDA torch, deps, lists every GPU, pytest

Recommended order (daily first, it is cheap and validates the real data):

    python main.py --freq daily --device cpu                        # both daily models
    python ablate.py --freqs daily --jobs 32                        # CPU workers, cores // jobs threads each
    python main.py --device cuda --parallel-configs                 # 4 configs at once, one process per GPU
    python ablate.py --freqs 5min --jobs 6 --device cuda --gpus 0,1,2   # 3 GPUs, 2 workers per GPU

Devices: `--device auto|cpu|cuda|cuda:K` (MPS is never used: no float64), `--gpus 0,1,2` (default: all
GPUs visible under `CUDA_VISIBLE_DEVICES`). `auto` picks CUDA only for wide workloads
(`streams * chunk_len >= 2^18` or `n_paths >= 200k`); otherwise CPU, because the recursion is sequential
and a GPU spends its time launching tiny kernels. The GARCH(1,1) benchmark always fits on CPU.
`ablate.py --jobs N` uses spawn worker processes; each is pinned to one device (`torch.cuda.set_device`),
GPUs are assigned round-robin (N may exceed the GPU count) and every result row records its `device`.

There is no DataParallel/DDP: one model is tiny and its filter is sequential in time, so the useful
parallelism is across runs (ablation cells, seeds, configurations), not within one.

Expected speed-ups: ablations scale close to linearly in `--jobs` on CPU until memory bandwidth
saturates (on an 8-core Mac the quick sweep went from 38 s with `--jobs 1` to 25 s with `--jobs 2`,
including the spawn start-up); for the full daily sweep (~5-14 h serial) expect roughly
cores/2 x. A GPU helps a single run only for the 5-min stage-1/stage-2 with many streams and for large
Monte Carlo path counts; on daily data stay on CPU. These GPU figures are untested so far: measure
on the box (`runtime_s` in `results.csv`).

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
