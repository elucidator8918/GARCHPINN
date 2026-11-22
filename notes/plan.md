# Plan: fast filtering (parallel scan) and parallel ablations

Status: proposed. Starts after the current fix workflow finishes, because that workflow is editing
`garchpinn/models/` and `data.py` now.

## Why

The variance recursion is sequential: `log h_t` needs `log h_{t-1}`. At 5-min frequency the real
data has about 650k steps. A step is a few scalar operations, so Python overhead dominates the run
time: about 290 s per epoch on 40k bars, and more than a day per full run. A GPU does not fix
this on its own, because each step is still a separate kernel launch. The fix is to change the
algorithm.

## 1. Parallel-scan filter for GARCHPINN

When the NN correction uses only exogenous inputs (`nn_state_inputs=False`), the Realized GARCH
recursion is **linear in observed data**:

    log h_t = beta * log h_{t-1} + b_t,    b_t = omega + gamma * log x_{t-1} + s * g(f_{t-1})

All `b_t` can be computed up front in one vectorised NN pass. The recursion is then a
first-order linear recurrence, with closed form

    log h_t = beta^t log h_0 + sum_{s=1..t} beta^(t-s) b_s

**Implementation.** Use a chunked Toeplitz scan rather than raw powers, since `beta^-s`
overflows.
- Split the series into blocks of length `L` (about 256 to 1024).
- Within a block, `log h = K @ b + beta^(1..L) * log h_in`, where `K[i,j] = beta^(i-j)` for
  `i >= j` (lower-triangular). That is one BLAS matmul per block.
- Carry `log h_in` between blocks. This is a sequential loop over `T/L` blocks only, about 1k
  iterations for 650k steps. A batched alternative is an associative scan over `(a, b)` pairs
  with `(a1,b1)∘(a2,b2) = (a1*a2, a2*b1 + b2)`, which takes log-depth.
- Then compute `z_t`, `u_t` and the likelihood fully vectorised, as now.
- Autograd works through the matmuls, so the training code does not change.

**Where it applies**

| Case | Fast path? |
|---|---|
| GARCHPINN, NN off (stage-1 QMLE) | yes |
| GARCHPINN, `nn_state_inputs=False` | yes |
| GARCHPINN, `nn_state_inputs=True` | no (g depends on `log h_{t-1}`, `z_{t-1}`) |
| EGARCHPINN | no (`z_t` depends on `h_t`, so the recursion is nonlinear) |

The fast path is selected automatically in `forward()`. Otherwise the code falls back to the
sequential loop.

**Caveats**
- **log h guard.** The smooth guard on `log h` is nonlinear. It is the identity in practice, so
  on the fast path apply it after the scan and `warn` if it binds anywhere, or skip it there.
- **Overnight term.** The overnight multiplier `kappa`, if added by the current workflow, only
  scales the variance in the return likelihood, so the scan is unaffected. If a day-boundary
  jump term enters the recursion, it folds into `b_t`.
- **Precision.** Stay in float64. Powers of `beta` with `beta` near 0.999 and `L=1024` are fine
  (about 0.36).

**Optional, EGARCH.** Nonlinear recurrences can be parallelised with fixed-point
Newton/parallel-scan iterations (DEER, Lim et al. 2023), which repeatedly linearise and scan.
This is research-grade; only consider it if EGARCH at 5-min stays too slow after section 2.

## 2. Run ablations in parallel across CPU cores

The 280 ablation runs are independent.
- Add `--jobs N` to `ablate.py` using `concurrent.futures.ProcessPoolExecutor`. Each worker
  calls `torch.set_num_threads(1)`, so N workers do not oversubscribe the cores.
- Each run writes its own `runs/<id>.json` plus a `.npy` of losses, and the CSV is rebuilt from
  those files. This avoids concurrent writes and keeps resumability (skip runs whose files exist).
- Cache the parsed dataset per frequency once, as a pickle or `.npz`, so workers do not each
  re-read the CSV.

## 3. GPU support (only when useful)

- **Apple MPS:** does not support float64, so skip it, or make float32 an explicit, tested option.
- **CUDA:** worthwhile only with many parallel streams (≥1024), large Monte Carlo path counts, or
  the scan path with big blocks. Add `--device {auto,cpu,cuda}` and keep `auto` = CPU unless
  `streams * chunk_len` is large.
- A 46 GB card is far more than needed. The models fit in a few hundred MB, and the benefit
  comes from parallel width, not memory.

## 4. Tests and acceptance

- **Equivalence:** the scan filter equals the sequential filter (`log_h`, `z`, `u`, NLL) to 1e-10
  on simulated data, for chunk sizes {1, 7, 256} and with carried state.
- **Gradients:** gradients w.r.t. all econ and NN parameters match the sequential path to 1e-8.
- **Fallback:** `nn_state_inputs=True` and EGARCH use the sequential path, and the selection is
  logged.
- **Parallel ablation:** `ablate.py --jobs 4 --quick` gives the same CSV as `--jobs 1`, up to
  row order.
- **Benchmarks** (record in README):
  - stage-1 QMLE on 650k 5-min bars: from hours down to minutes
  - full daily ablation sweep: from about 5–14 h down to about 1–2 h on 8 cores

## Files

- `garchpinn/models/scan.py` (new): `linear_scan(beta, b, h0, block)` with tests
- `garchpinn/models/base.py`, `garch_pinn.py`: fast-path dispatch in `forward()`
- `ablate.py`, `garchpinn/ablation.py`: `--jobs`, per-run files, dataset cache
- `garchpinn/train.py`, `main.py`: `--device`
- `tests/test_scan.py`, an extension of `tests/test_ablation.py`
- `docs/realized_garch_reference.tex`: a short "Computation" subsection (closed form, scan, and
  why EGARCH stays sequential)
