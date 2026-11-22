# TODO

Last updated: 2026-11-23. See also [plan.md](plan.md) for the fast-filter / parallel-ablation design.

## Done (tested on simulated + synthetic data)

- [x] `GARCHPINN` (Realized GARCH, HHS 2012) and `EGARCHPINN` (Realized EGARCH, HH 2016),
      with parameter-recovery tests
- [x] Two-stage training: stage-1 QMLE (L-BFGS), then the stage-2 PINN (Adam, TBPTT with carried state)
  - econ warm-up freeze, then the econ params train at a reduced lr
  - cosine schedule
  - standardised moment penalty, and a test that stage 2 beats stage 1
- [x] Identification penalty: g is orthogonal to log h_{t-1} and log x_{t-1} (garch),
      and to z, z^2-1 and u (egarch)
- [x] Data: daily open-to-close RV, 5-min with periodicity filter + Parkinson measure, no look-ahead,
      date-based train/val/test split
- [x] Forecasts: analytic with Gaussian/empirical MGF (Jensen) correction, and Monte Carlo with NN
- [x] Evaluation: QLIKE/MSE by horizon vs GARCH(1,1), HAR and pure RGARCH; DM test (HLN, NW bandwidth)
- [x] Diagnostics: F1–F7, R1–R5, I1–I5, E2–E6 (`--diagnostics`, `--fast`)
- [x] Ablation runner: `ablate.py`, 14 variants, resumable, plots P1–P5
- [x] Reference doc `docs/realized_garch_reference.tex/.pdf`, and the papers in `papers/`

## In progress (fix workflow running when this was written)

- [ ] 5-min overnight gap: overnight return with its own variance multiplier kappa
- [ ] Automatic bar-timestamp convention check (`bar_stamp = auto|start|end`)
- [ ] 5-min speed-up (target < 30 min per configuration on CPU)
- [ ] Integration pass: full tests, 4-config smoke run, quick ablation
- [ ] Fable math review of the above, and apply the confirmed fixes

## Next

- [ ] **Run on the real FTSE CSV** (`python main.py --data UK_Index_2000_2025.csv`). Nothing has been
      validated on real data yet.
- [ ] Check on real data: first-bar stamping, bars/session (~102?), High/Low columns present
- [ ] Parallel-scan filter for GARCHPINN (see plan.md §1)
- [ ] `ablate.py --jobs N` for parallel runs across CPU cores (plan.md §2)
- [ ] Full ablation sweep: daily first (~5–14 h serial), 5-min after the speed-up
- [ ] `--device` option; CUDA only for wide stream counts (MPS has no float64)

## Known limitations / nice-to-have

- [ ] Plot DPI is 80 (`DPI` in `garchpinn/plots.py`); raise it for publication figures
- [ ] Rolling/expanding re-estimation for out-of-sample evaluation (doc §8.2)
- [ ] Model Confidence Set is approximate (block bootstrap)
- [ ] DM / MCS / Kupiec / Christoffersen / MZ papers are cited but not in `papers/`
- [ ] Nelson (1991) EGARCH and the Raissi et al. 2019 journal version are not downloaded (paywalled)
