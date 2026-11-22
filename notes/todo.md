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

- [x] 5-min overnight gap: overnight return with a learned variance multiplier kappa
      (return likelihood only; `overnight = model|drop`)
- [x] Automatic bar-timestamp convention check (`bar_stamp = auto|start|end`)
- [x] 5-min speed-up: filter restructured (same maths, matches `forward_reference` to 1e-12)
- [x] Integration pass: 63 tests pass; 4-config smoke and quick ablation clean
- [x] Ablation switches for the new features: `overnight_drop`, `moment_raw`, `no_warmup`
- [x] `--device auto|cpu|cuda|cuda:K` and `--gpus` (main.py, ablate.py); device threaded through all
      modules, static audit test of tensor factories, meta-device forward test, mocked 1/3-GPU assignment
- [x] `ablate.py --jobs N` (spawn pool, per-run atomic `runs/*.json`, dataset cache, GPU round-robin);
      `main.py --parallel-configs`; `scripts/setup_linux.sh`; README GPU section

## Paused

- [ ] Fable math review of the overnight/orthogonality/stage-2/speed changes, then apply fixes.
      Paused at the user's request (the Fable limit resets 2026-11-29). Resume with the
      `garchpinn-fix-all` workflow script (`resumeFromRunId wf_51bf06f3-3f4`).

## Open issues from integration (synthetic data)

- [ ] GARCH(1,1) benchmark (`benchmarks.py`) is now the main 5-min cost, so a full 5-min run
      still takes more than 30 min
- [ ] EGARCH daily: the NN overfits (val NLL rises even while the econ params are frozen),
      so stage 2 is discarded. Needs more data, a stronger lambda_g, a smaller net, or
      `nn_state_inputs=False`
- [ ] HAR beats the models at the daily k=22 horizon
- [ ] F3 plot: Monte Carlo overlay barely visible, and the inset overlaps the curve
- [ ] I5 plot: the stage-1 point is hidden under the stage-2 point
- [ ] Optional day-boundary jump term in the 5-min recursion (doc lists it as an extension)

## Next

- [ ] **Run on the real FTSE CSV** (`python main.py --data UK_Index_2000_2025.csv`). Nothing has been
      validated on real data yet.
- [ ] Check on real data: first-bar stamping, bars/session (~102?), High/Low columns present
- [ ] Parallel-scan filter for GARCHPINN (see plan.md §1)
- [ ] Full ablation sweep: daily first (~5–14 h serial), 5-min after the speed-up
- [ ] On the GPU box: run `scripts/setup_linux.sh`; check the CUDA tests pass (CPU/GPU agreement 1e-8,
      2 jobs on 2 GPUs) and measure `--device cuda` vs cpu for 5-min; tune `AUTO_CUDA_MIN_WIDTH`

## Known limitations / nice-to-have

- [ ] Plot DPI is 80 (`DPI` in `garchpinn/plots.py`); raise it for publication figures
- [ ] Rolling/expanding re-estimation for out-of-sample evaluation (doc §8.2)
- [ ] Model Confidence Set is approximate (block bootstrap)
- [ ] DM / MCS / Kupiec / Christoffersen / MZ papers are cited but not in `papers/`
- [ ] Nelson (1991) EGARCH and the Raissi et al. 2019 journal version are not downloaded (paywalled)
