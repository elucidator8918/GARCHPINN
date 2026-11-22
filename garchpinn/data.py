"""Data pipeline: FTSE 5-min bars -> aligned (r_t, x_t) series and NN features.

Units: returns are scaled by Config.return_scale and x_t is in the same units squared, so the
measurement equation has phi ~ 1, xi ~ 0 (reference doc, Section 1).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import Config


# ---------------------------------------------------------------- loading
def infer_bar_stamp(ts: pd.Series, absr: pd.Series, cfg: Config):
    """Infer whether 5-min bar timestamps mark the bar start or the bar end (doc Section 1).

    Three votes, each from all days in the file (before the session filter):
      1. modal first stamp of the day: session_start -> start, session_start + 5 -> end;
      2. modal last stamp: session_end -> start (the 16:30 bar is the closing-auction bar
         [16:30, 16:35)), session_end + 5 -> end;
      3. the opening-auction bar has by far the largest mean |r|: if the bar labelled
         session_start carries it -> start, if the one labelled session_start + 5 does -> end.
    Returns (verdict, details). Ties resolve to "start" (the Refinitiv convention)."""
    five = pd.Timedelta(minutes=5)
    t0, t1 = pd.Timedelta(cfg.session_start + ":00"), pd.Timedelta(cfg.session_end + ":00")
    tod = ts - ts.dt.normalize()
    day = ts.dt.normalize()
    first = tod.groupby(day.values).min().mode().iloc[0]
    last = tod.groupby(day.values).max().mode().iloc[0]
    votes = {"start": 0, "end": 0}
    v1 = "start" if first == t0 else ("end" if first == t0 + five else None)
    v2 = "start" if last == t1 else ("end" if last == t1 + five else None)
    m = absr.groupby(tod.values).mean()
    a0, a1 = float(m.get(t0, np.nan)), float(m.get(t0 + five, np.nan))
    v3 = None
    if np.isfinite(a0) and np.isfinite(a1) and max(a0, a1) > 0:
        v3 = "start" if a0 > 1.2 * a1 else ("end" if a1 > 1.2 * a0 else None)
    elif np.isfinite(a1) and not np.isfinite(a0):
        v3 = "end"
    for v in (v1, v2, v3):
        if v is not None:
            votes[v] += 1
    verdict = "end" if votes["end"] > votes["start"] else "start"
    return verdict, {"first": str(first), "last": str(last), "open_absr_ratio": a0 / a1 if a1 else np.nan,
                     "votes": (v1, v2, v3)}


def load_intraday(cfg: Config) -> pd.DataFrame:
    """Session bars with intraday log returns r, stamped at the bar START internally.

    Overnight (cfg.overnight, doc Section 1):
      "drop":  legacy; the first bar of each day (its close-to-previous-close return holds the
               overnight gap) is dropped.
      "model": the first bar is kept with its own intraday return log(Last/Open) (when an Open
               column exists), and the overnight return r_on = log(Open_first / Last_prev close)
               is stored on that row (column r_on, flag on = 1). Without Open, r_on is the
               first bar's close-to-close return and is stored on the second bar instead.
    """
    df = pd.read_csv(cfg.data_path)
    df = df.dropna(axis=1, how="all")
    ts = pd.to_datetime(df["Date-Time"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    if "GMT Offset" in df:
        ts = ts + pd.to_timedelta(df["GMT Offset"], unit="h")
    df = df.assign(ts=ts).sort_values("ts")
    ts = df["ts"]
    # ---- timestamp convention (bar start vs bar end)
    stamp = getattr(cfg, "bar_stamp", "auto")
    has_open = "Open" in df and bool((df["Open"] > 0).any())
    if has_open:
        absr = np.log(df["Last"] / df["Open"]).abs()
    else:
        absr = np.log(df["Last"]).groupby(ts.dt.normalize().values).diff().abs()
    verdict, info = infer_bar_stamp(ts, absr, cfg)
    if stamp == "auto":
        stamp = verdict
        print(f"bar timestamps inferred as bar-{verdict} ({info})")
    elif stamp != verdict:
        print(f"note: bar_stamp={stamp!r} but the data look bar-{verdict} ({info})")
    if stamp == "end":
        df["ts"] = df["ts"] - pd.Timedelta(minutes=5)
    df = df.set_index("ts")
    df.attrs["bar_stamp"] = stamp
    df = df.loc[cfg.start_date:cfg.end_date].between_time(cfg.session_start, cfg.session_end)
    df = df[df["Last"] > 0].copy()
    df["date"] = df.index.normalize()
    lp = np.log(df["Last"])
    prev_close = lp.groupby(df["date"]).last().shift(1)  # last session bar of the previous day
    df["r"] = lp.groupby(df["date"]).diff()
    # a return spanning a gap (missing bars) is not a 5-min return: flagged, dropped at 5-min
    dt = df.index.to_series().groupby(df["date"].values).diff()
    df["gap_ok"] = (dt == pd.Timedelta(minutes=5)).values
    first = df["r"].isna().values  # first bar of each day
    # close-to-close return of the first bar (overnight gap + first 5 min); daily close-to-close
    df["r_c2c_first"] = np.where(first, lp.values - df["date"].map(prev_close).values, np.nan)
    df["r_c2c_day"] = df["date"].map(df.loc[first].set_index("date")["r_c2c_first"]).values
    df["on"] = 0.0
    df["r_on"] = 0.0
    mode = getattr(cfg, "overnight", "drop")
    if mode == "model" and cfg.freq == "5min":
        has_prev = np.isfinite(df["r_c2c_first"].values)
        if has_open:
            ok = first & (df["Open"].values > 0)
            df.loc[ok, "r"] = np.log(df["Last"].values[ok] / df["Open"].values[ok])
            df.loc[ok, "gap_ok"] = True
            r_on = np.log(df["Open"].values) - df["date"].map(prev_close).values
            # r_on == 0 exactly (no overnight print, e.g. Open copied from the previous close)
            # carries no scale information and makes the kappa likelihood unbounded: not an obs
            sel = ok & has_prev & np.isfinite(r_on) & (np.abs(r_on) > 0)
            df.loc[sel, "r_on"] = r_on[sel]
            df.loc[sel, "on"] = 1.0
        else:  # no Open: attach the whole first-bar return to the second bar of the day
            nxt = np.roll(first, 1) & ~first
            src = np.roll(np.where(has_prev & first, df["r_c2c_first"].values, np.nan), 1)
            sel = nxt & np.isfinite(src) & (np.abs(np.nan_to_num(src)) > 0)
            df.loc[sel, "r_on"] = src[sel]
            df.loc[sel, "on"] = 1.0
    df = df.dropna(subset=["r"]).copy()
    # clock-time bin (Andersen & Bollerslev 1997): minutes since session open / 5
    open_t = df["date"] + pd.Timedelta(cfg.session_start + ":00")
    df["bar"] = ((df.index - open_t) // pd.Timedelta(minutes=5)).astype(int)
    df.attrs["bar_stamp"] = stamp
    return df


def n_session_bars(cfg: Config) -> int:
    t0, t1 = pd.Timedelta(cfg.session_start + ":00"), pd.Timedelta(cfg.session_end + ":00")
    return int((t1 - t0) // pd.Timedelta(minutes=5)) + 1


def train_cut_date(intraday: pd.DataFrame, cfg: Config):
    """First date outside the training split (first train_frac of the trading days)."""
    d = np.sort(intraday["date"].unique())
    return d[min(int(cfg.train_frac * len(d)), len(d) - 1)]


def daily_series(intraday: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """r_t = sum of intraday returns (open-to-close), x_t = RV_t = sum of squared returns."""
    r = intraday["r"] * cfg.return_scale
    g = r.groupby(intraday["date"])
    out = pd.DataFrame({"r": g.sum(), "x": (r ** 2).groupby(intraday["date"]).sum(), "n": g.size()})
    out = out[(out["n"] >= cfg.min_bar_frac * out["n"].median()) & (out["x"] > 0)]
    out["cal"] = out.index.day_of_week / 4.0
    if getattr(cfg, "daily_close_to_close", False):
        # close-to-close variant: the overnight + first-bar return (previous close to the first
        # bar's close) enters as r_on with variance exp(kappa) h_t (doc Section 1); r stays
        # open-to-close and x stays RV, so the measurement equation is unchanged.
        c2c = (intraday["r_c2c_day"] * cfg.return_scale).groupby(intraday["date"]).first()
        c2c = c2c.reindex(out.index)
        out["on"] = (np.isfinite(c2c.values) & (np.nan_to_num(c2c.values) != 0)).astype(float)
        out["r_on"] = np.nan_to_num(c2c.values)
    return out.drop(columns="n")


def fivemin_series(intraday: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """5-min returns divided by the intraday periodicity factor s_k (Andersen & Bollerslev 1997);
    realized measure = Parkinson range (ln H/L)^2 / (4 ln 2), also divided by s_k^2."""
    if not {"High", "Low"}.issubset(intraday.columns):
        raise ValueError("freq='5min' needs High/Low columns for the range-based realized measure")
    df = intraday[(intraday["High"] > 0) & (intraday["Low"] > 0) & intraday["gap_ok"]]
    n = df.groupby("date").size()
    nbars = int(n.median())
    if abs(nbars - 102) > 3 and abs(nbars - 103) > 3:
        print(f"note: median {nbars} bars/session after filtering (expected ~102-103)")
    df = df[df["date"].map(n >= cfg.min_bar_frac * nbars).values]  # day-completeness filter
    nsb = n_session_bars(cfg)
    # periodicity factor and floor estimated on training days only (no look-ahead)
    tr = (df["date"] < train_cut_date(df, cfg)).values
    r = (df["r"] * cfg.return_scale).values
    park = (np.log(df["High"] / df["Low"]).values * cfg.return_scale) ** 2 / (4 * np.log(2))
    bar = df["bar"].values
    s2 = periodicity_factor(r[tr], bar[tr])
    s2_t = pd.Series(bar).map(s2).fillna(1.0).values
    x = park / s2_t
    xt = x[tr]
    x = np.maximum(x, np.quantile(xt[xt > 0], 0.01))  # H == L bars would give log 0
    out = pd.DataFrame({"r": r / np.sqrt(s2_t), "x": x, "cal": bar / nsb}, index=df.index)
    if getattr(cfg, "overnight", "drop") == "model":
        # overnight return (scaled, NOT divided by s_k: its own scale is exp(kappa) h_t)
        out["on"] = df["on"].values
        out["r_on"] = df["r_on"].values * cfg.return_scale
    # exposed for diagnostics (doc F6): s_k^2 by bar, the per-row factor and the train mask
    out.attrs["s2"] = s2
    out.attrs["s2_t"] = s2_t
    out.attrs["train_mask"] = tr
    out.attrs["bar"] = bar
    return out


def periodicity_factor(r, bar):
    """s_k^2 = mean(r^2 | bar k) / overall mean of the bar means (Andersen & Bollerslev 1997).
    r, bar: training-day returns (scaled) and bar indices. Returns a pd.Series indexed by bar;
    the periodicity factor plotted in F6 is s_k = sqrt of this."""
    s2 = pd.Series(np.asarray(r) ** 2).groupby(np.asarray(bar)).mean()
    return s2 / s2.mean()


def cal_angle(cal, freq):
    return 2 * np.pi * np.asarray(cal) * (4 / 5 if freq == "daily" else 1.0)


def future_cal_angles(df: pd.DataFrame, cfg: Config, horizon: int) -> np.ndarray:
    """Calendar angle of T, T+1, .. (the periods after the last observation T-1)."""
    last = float(df["cal"].values[-1])
    if cfg.freq == "daily":
        dow = (round(last * 4) + np.arange(1, horizon + 1)) % 5
        return cal_angle(dow / 4.0, cfg.freq)
    nsb = n_session_bars(cfg)
    b = round(last * nsb)
    first = int(np.min(np.round(df["cal"].values * nsb)))
    span = nsb - first
    bars = first + (b - first + np.arange(1, horizon + 1)) % span
    return cal_angle(bars / nsb, cfg.freq)


def get_series(cfg: Config) -> pd.DataFrame:
    intraday = load_intraday(cfg)
    return daily_series(intraday, cfg) if cfg.freq == "daily" else fivemin_series(intraday, cfg)


# ---------------------------------------------------------------- features / splits
def exo_features(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Row t holds F_t-measurable inputs for the correction in log h_{t+1}:
    HAR averages of log x up to t (Corsi 2009) and the calendar position of t+1."""
    lx = pd.Series(np.log(df["x"].values))
    har = [lx.rolling(w, min_periods=1).mean().values for w in cfg.har_windows]
    cal_next = np.append(df["cal"].values[1:], df["cal"].values[-1])
    ang = cal_angle(cal_next, cfg.freq)
    return np.column_stack(har + [np.sin(ang), np.cos(ang)])


ON_COLS = ("on", "r_on")  # raw (unstandardised) trailing exo columns read by the filter, not the NN


def has_overnight(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ON_COLS)


@dataclass
class Dataset:
    r: torch.Tensor
    log_x: torch.Tensor
    exo: torch.Tensor
    index: pd.Index
    n_train: int
    n_val: int
    exo_mu: np.ndarray = None  # training-split feature standardisation
    exo_sd: np.ndarray = None

    @property
    def T(self):
        return len(self.r)

    def split(self, part):
        a, b = {"train": (0, self.n_train), "val": (self.n_train, self.n_train + self.n_val),
                "test": (self.n_train + self.n_val, self.T), "fit": (0, self.n_train + self.n_val)}[part]
        return slice(a, b)


def build_dataset(df: pd.DataFrame, cfg: Config, device="cpu") -> Dataset:
    T = len(df)
    n_train, n_val = int(cfg.train_frac * T), int(cfg.val_frac * T)
    exo = exo_features(df, cfg)
    mu, sd = exo[:n_train].mean(0), exo[:n_train].std(0) + 1e-8  # training statistics only
    exo = (exo - mu) / sd
    if has_overnight(df):  # appended raw; exo_mu/exo_sd cover the NN columns only
        exo = np.column_stack([exo] + [df[c].values for c in ON_COLS])
    t = lambda a: torch.as_tensor(np.array(a, copy=True), dtype=torch.get_default_dtype(), device=device)
    return Dataset(t(df["r"].values), t(np.log(df["x"].values)), t(exo), df.index, n_train, n_val, mu, sd)


def streams(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """Split a (T, ...) series into n contiguous segments (n, T//n, ...), dropping the remainder."""
    n = max(1, min(n_streams, x.shape[0]))
    L = x.shape[0] // n
    return x[: n * L].reshape(n, L, *x.shape[1:])
