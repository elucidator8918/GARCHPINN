"""5-min data handling (bar stamps, overnight return) and the fast filter (same maths)."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from garchpinn.config import Config
from garchpinn.data import build_dataset, daily_series, fivemin_series, has_overnight, load_intraday
from garchpinn.models import EGARCHPINN, GARCHPINN, build_model, uses_overnight

torch.set_default_dtype(torch.float64)
CLS = {"garch": GARCHPINN, "egarch": EGARCHPINN}


def _csv(path, days=30, shift_min=0, seed=0):
    """Synthetic 08:00-16:30 bars (bar-start stamps, shifted by shift_min), loud opening bar."""
    rng = np.random.default_rng(seed)
    rows, p = [], 6500.0
    for d in pd.bdate_range("2020-01-06", periods=days):
        p *= math.exp(0.01 * rng.standard_normal())  # overnight move
        for k in range(103):
            s = 0.004 if k == 0 else 0.001
            o = p
            p = o * math.exp(s * rng.standard_normal())
            hi, lo = max(o, p) * (1 + 0.3 * s), min(o, p) * (1 - 0.3 * s)
            t = d + pd.Timedelta(hours=8, minutes=5 * k + shift_min)
            rows.append((t, 0, o, hi, lo, p))
    pd.DataFrame(rows, columns=["Date-Time", "GMT Offset", "Open", "High", "Low", "Last"]).to_csv(path, index=False)
    return str(path)


def _cfg(path, **kw):
    return Config(freq=kw.pop("freq", "5min"), data_path=path, start_date="2020-01-01",
                  end_date="2020-12-31", **kw)


# ---------------------------------------------------------------- timestamps
@pytest.mark.parametrize("shift,verdict", [(0, "start"), (5, "end")])
def test_bar_stamp_inferred_and_shifted(tmp_path, shift, verdict, capsys):
    a = load_intraday(_cfg(_csv(tmp_path / "a.csv"), bar_stamp="start"))
    b = load_intraday(_cfg(_csv(tmp_path / "b.csv", shift_min=shift)))
    assert verdict in capsys.readouterr().out
    assert b.attrs["bar_stamp"] == verdict
    assert b.index.equals(a.index)  # auto shift lands on the bar-start convention
    np.testing.assert_allclose(b["r"].values, a["r"].values)


def test_bar_stamp_config_validated():
    with pytest.raises(AssertionError):
        Config(bar_stamp="middle")
    with pytest.raises(AssertionError):
        Config(freq="5min", overnight="keep")


# ---------------------------------------------------------------- overnight
def test_overnight_defaults():
    assert Config(freq="5min").overnight == "model" and uses_overnight(Config(freq="5min"))
    assert Config().overnight == "drop" and not uses_overnight(Config())
    assert uses_overnight(Config(daily_close_to_close=True))
    assert not uses_overnight(Config(freq="5min", overnight="drop"))


def test_overnight_model_vs_drop(tmp_path):
    p = _csv(tmp_path / "a.csv", days=8)
    raw = pd.read_csv(p)
    drop = load_intraday(_cfg(p, overnight="drop"))
    mod = load_intraday(_cfg(p, overnight="model"))
    assert len(mod) == len(drop) + 8  # first bar of each day kept
    first = mod.groupby("date").head(1)
    # its intraday return is log(Last / Open) and r_on = log(Open_first / previous close)
    np.testing.assert_allclose(first["r"].values, np.log(raw["Last"] / raw["Open"]).values[::103])
    lastc = raw["Last"].values[102::103]
    np.testing.assert_allclose(first["r_on"].values[1:], np.log(raw["Open"].values[::103][1:] / lastc[:-1]))
    assert first["on"].values[0] == 0 and (first["on"].values[1:] == 1).all()  # no previous close
    assert (mod["on"].values.sum() == 7) and (drop["on"].values.sum() == 0)

    cfg = _cfg(p, overnight="model", min_bar_frac=0.5)
    df = fivemin_series(mod, cfg)
    assert has_overnight(df)
    ds = build_dataset(df, cfg)
    m = build_model(cfg, ds.exo.shape[-1])
    assert m.overnight and m.n_exo_nn == ds.exo.shape[-1] - 2 == len(ds.exo_mu)
    np.testing.assert_allclose(ds.exo[:, -1].numpy(), df["r_on"].values)  # raw, unstandardised


def test_daily_close_to_close_variant(tmp_path):
    p = _csv(tmp_path / "a.csv", days=8)
    cfg = _cfg(p, freq="daily", daily_close_to_close=True)
    d = daily_series(load_intraday(cfg), cfg)
    assert has_overnight(d) and d["on"].sum() == 7
    assert not has_overnight(daily_series(load_intraday(_cfg(p, freq="daily")), _cfg(p, freq="daily")))


@pytest.mark.parametrize("v", ["garch", "egarch"])
def test_overnight_likelihood_and_nn_blind(v):
    torch.manual_seed(0)
    m = CLS[v](n_exo=5, overnight=True)
    for p in m.net.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    T = 3000
    r, lx = torch.randn(T), 0.5 * torch.randn(T)
    on = (torch.arange(T) % 100 == 0).double()
    ex = torch.cat([torch.randn(T, 3), on[:, None], torch.zeros(T, 1)], 1)
    with torch.no_grad():
        out, _ = m(r, lx, ex)
        kappa = 1.3
        r_on = on * torch.randn(T) * torch.exp(0.5 * (kappa + out["log_h"]))
        ex2 = ex.clone()
        ex2[:, -1] = r_on
        out2, _ = m(r, lx, ex2)
    # the filter (and g) never read the overnight columns
    for k in ("log_h", "z", "u", "g"):
        torch.testing.assert_close(out[k], out2[k])
    # the return likelihood gains on_t * -log N(r_on; 0, exp(kappa) h_t)
    a0, _ = m.nll_parts(out2)
    with torch.no_grad():
        m.kappa.fill_(0.0)
    l0 = 0.5 * (math.log(2 * math.pi) + out2["log_h"] + out2["z"] ** 2)
    extra = on * 0.5 * (math.log(2 * math.pi) + out2["log_h"] + r_on ** 2 * torch.exp(-out2["log_h"]))
    torch.testing.assert_close(m.nll_parts(out2)[0], (l0 + extra).mean())
    # kappa's ML estimate is log mean(r_on^2 / h) over overnight rows
    opt = torch.optim.LBFGS([m.kappa], max_iter=50, line_search_fn="strong_wolfe")

    def clo():
        opt.zero_grad()
        loss = m.nll_parts(out2)[0]
        loss.backward()
        return loss

    opt.step(clo)
    sel = on > 0
    k_hat = math.log(float((r_on[sel] ** 2 * torch.exp(-out2["log_h"][sel])).mean()))
    assert abs(float(m.kappa.detach()) - k_hat) < 1e-5 and abs(k_hat - kappa) < 0.4
    assert "kappa" in m.params()


# ---------------------------------------------------------------- fast filter == reference
@pytest.mark.parametrize("v", ["garch", "egarch"])
@pytest.mark.parametrize("state_inputs", [True, False])
@pytest.mark.parametrize("B", [1, 4])
def test_fast_forward_matches_reference(v, state_inputs, B):
    torch.manual_seed(1)
    m = CLS[v](n_exo=4, nn_state_inputs=state_inputs)
    for p in m.net.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    r, lx, ex = torch.randn(B, 300), 0.5 * torch.randn(B, 300), torch.randn(B, 300, 4)
    with torch.no_grad():
        _, st = m.forward_reference(r[:, :40], lx[:, :40], ex[:, :40])
    for state in (None, st):
        ref, ref_st = m.forward_reference(r, lx, ex, state)
        fast, fast_st = m(r, lx, ex, state)
        with torch.no_grad():
            nog, nog_st = m(r, lx, ex, state)
        for o in (fast, nog):
            for k in ("log_h", "z", "u", "g", "log_h_next", "g_next"):
                torch.testing.assert_close(o[k], ref[k], rtol=0, atol=1e-12)
        for k in ref_st:
            torch.testing.assert_close(nog_st[k], ref_st[k], rtol=0, atol=1e-12)
        params = [p for p in m.parameters() if p.requires_grad]
        g_ref = torch.autograd.grad(sum(m.nll_parts(ref)), params, allow_unused=True)
        g_fast = torch.autograd.grad(sum(m.nll_parts(fast)), params, allow_unused=True)
        for a, b in zip(g_ref, g_fast):
            if a is not None:
                torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-12)


def test_nn_off_matches_reference():
    m = EGARCHPINN(n_exo=2)
    m.nn_active = False
    r, lx = torch.randn(2, 200), torch.randn(2, 200)
    ref, _ = m.forward_reference(r, lx)
    with torch.no_grad():
        fast, _ = m(r, lx)
    torch.testing.assert_close(fast["log_h"], ref["log_h"].detach(), rtol=0, atol=1e-12)
    assert float(fast["g"].abs().max()) == 0.0


def test_fivemin_speed_defaults():
    c = Config(freq="5min")
    assert c.stage1_streams >= 256 and c.streams >= 128 and c.epochs <= 30


def test_zero_overnight_return_not_an_observation(tmp_path):
    """An exactly zero r_on (Open == previous close) would make the kappa likelihood unbounded."""
    p = _csv(tmp_path / "a.csv", days=6)
    raw = pd.read_csv(p)
    first = np.arange(0, len(raw), 103)
    raw.loc[first[1:], "Open"] = raw["Last"].values[first[1:] - 1]
    raw.to_csv(p, index=False)
    d = load_intraday(_cfg(p, overnight="model"))
    assert d["on"].sum() == 0 and (d["r_on"] == 0).all()
