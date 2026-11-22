"""Experiment configuration.

model: "garch"  -> GARCHPINN  (log-linear Realized GARCH, Hansen, Huang & Shek 2012)
       "egarch" -> EGARCHPINN (Realized EGARCH, Hansen & Huang 2016)
freq:  "daily"  -> open-to-close daily returns, RV = sum of squared 5-min returns
       "5min"   -> 5-min returns, intraday periodicity removed, Parkinson range measure
See docs/realized_garch_reference.tex for every equation used.
"""
from dataclasses import asdict, dataclass

MODELS = ("garch", "egarch")
FREQS = ("daily", "5min")


@dataclass
class Config:
    model: str = "garch"
    freq: str = "daily"

    # ---- data ----
    data_path: str = "UK_Index_2000_2025.csv"
    start_date: str = "2000-01-01"
    end_date: str = "2025-12-31"
    session_start: str = "08:00"  # LSE continuous trading, local time
    session_end: str = "16:30"
    min_bar_frac: float = 0.8  # daily: drop days with < 80% of the median bar count
    train_frac: float = 0.70  # split by time, never shuffled
    val_frac: float = 0.15
    # 5-min bar timestamps: "auto" infers bar-start vs bar-end from the first/last stamp of the
    # day and the opening-auction |r| (data.infer_bar_stamp, logged); "start"/"end" force it
    bar_stamp: str = "auto"
    # overnight return at 5-min (doc Section 1): "model" keeps the first bar (log Last/Open) and
    # adds r_on ~ N(0, exp(kappa) h_t) at the open to the return likelihood; "drop" = legacy
    # (first bar dropped, recursion runs straight across the day boundary). None -> per-freq default
    overnight: str = None
    daily_close_to_close: bool = False  # daily: also put r_on (with kappa) in the likelihood

    # ---- model ----
    use_nn: bool = True
    hidden_dim: int = 32
    nn_state_inputs: bool = True  # feed z_{t-1}, log h_{t-1} to the network (F_{t-1}-measurable)
    nn_scale: float = 0.25  # |s * g| <= nn_scale
    log_h_bound: float = 15.0  # guard half-width around the training log variance; identity inside
    no_leverage: bool = False  # ablation: freeze the leverage terms tau1 = tau2 = 0

    # ---- stage 1: QMLE of the pure econometric model (L-BFGS) ----
    stage1_iters: int = 300
    stage1_streams: int = 1  # >1 splits the sample into parallel streams (5-min speed-up)
    stage1_burn_in: int = 22

    # ---- stage 2: PINN (Adam, truncated BPTT, carried state) ----
    epochs: int = 150
    lr: float = 3e-3
    weight_decay: float = 1e-4  # lambda_w
    streams: int = 8  # parallel contiguous segments of the training sample
    chunk_len: int = 128  # TBPTT length; state carried (detached) between chunks
    grad_clip: float = 5.0
    lambda_moment: float = 1.0
    moment_lags: int = 5
    lambda_g: float = 1e-2
    moment_standardize: bool = True  # R_moment in GMM units (each mean / its s.e., over n)
    lambda_orth: float = 1.0  # penalise mean(g), cov(g, GARCH-eq. regressors at t-1) (identification)
    econ_warmup: int = 10  # stage-2 epochs with econ params frozen at the QMLE (NN learns first)
    econ_lr_mult: float = 0.1  # econ-param lr = lr * this after the warm-up
    lr_schedule: str = "cosine"  # "cosine" (decay over epochs) or "constant"
    patience: int = 20  # counted after the warm-up
    likelihood: str = "joint"  # "joint" (l_rx) or "partial" (l_r only; ablation)
    carry_state: bool = True  # False: re-initialise the filter state at every TBPTT chunk (ablation)

    # ---- evaluation ----
    horizons: tuple = (1, 5, 22)
    har_windows: tuple = (1, 5, 22)
    forecast_horizon: int = 22
    n_paths: int = 5000
    # analytic E x MGF for evaluation and main's forecasts: "gaussian" or "empirical" (training
    # residuals). 5-min defaults to "empirical" (Gaussian MGF is biased when tau2/delta2 is large);
    # the other one is always reported alongside ("<name>_emp" or "<name>_gauss").
    forecast_mgf: str = "gaussian"

    # ---- diagnostics (doc Section figs; main.py --diagnostics/--fast/--recovery/--rolling) ----
    diagnostics: bool = True
    diag_fast: bool = False  # skip profile likelihood and MCS
    diag_recovery: bool = False  # I3 (slow)
    diag_rolling: bool = False  # I4 (slow)
    irf_paths: int = 5000  # MC overlay paths for F3 (0: analytic only)
    diag_max_obs: int = 6000  # training obs used for Hessian / sandwich s.e.
    profile_grid: int = 7
    profile_iters: int = 10  # LBFGS iterations per profile point
    profile_max_obs: int = 2000
    mcs_alpha: float = 0.10
    mcs_boot: int = 500

    # "auto" (CUDA only for wide workloads, see utils.resolve_device), "cpu", "cuda", "cuda:K"
    device: str = "auto"
    seed: int = 42
    use_wandb: bool = False
    out_dir: str = "results"

    def __post_init__(self):
        assert self.model in MODELS, self.model
        assert self.freq in FREQS, self.freq
        if self.overnight is None:
            self.overnight = "model" if self.freq == "5min" else "drop"
        assert self.overnight in ("drop", "model"), self.overnight
        assert self.bar_stamp in ("auto", "start", "end"), self.bar_stamp
        if self.freq == "5min":
            # ~102 bars per LSE session (08:00-16:30); verify against the data at load time
            self.horizons = (1, 12, 102)
            self.har_windows = (1, 12, 102)
            self.forecast_horizon = 102
            # speed (measured, see doc Section 7): the cost is ~sequential steps, almost flat in
            # the stream count, so many short streams. 256 streams x 102-bar burn-in masks ~6%
            # of 650k bars; stage 1 L-BFGS then needs ~1.8k steps per evaluation, not 455k.
            self.stage1_streams = max(self.stage1_streams, 256)
            self.stage1_burn_in = max(self.stage1_burn_in, 102)
            self.stage1_iters = min(self.stage1_iters, 150)  # 6 econ params; converges well before
            self.streams = max(self.streams, 128)
            self.chunk_len = max(self.chunk_len, 512)
            self.epochs = min(self.epochs, 30)
            self.econ_warmup = min(self.econ_warmup, 5)
            self.patience = min(self.patience, 8)
            self.forecast_mgf = "empirical"
        assert self.forecast_mgf in ("gaussian", "empirical"), self.forecast_mgf

    @property
    def name(self):
        return f"{self.model}pinn_{self.freq}" if self.use_nn else f"realized{self.model}_{self.freq}"

    @property
    def return_scale(self):
        """Percent for daily (log h in about [-2, 2.5]). For 5-min, percent times sqrt(102)
        (daily-percent-equivalent units), so per-bar log h is near 0 as well rather than 4 to 5
        in basis points. The log h guard is also centred on the training log variance."""
        return 100.0 if self.freq == "daily" else 100.0 * 102 ** 0.5

    def to_dict(self):
        return asdict(self)
