"""Fund-of-strategies combiner over independent sleeve PnL streams.

Three textbook combination rules are evaluated: equal-weight, inverse
realised vol (risk parity), and long-only minimum-variance. The rules
are fixed — no parameter grid, no search — so the multi-strategy view
adds zero extra degrees of freedom on top of the per-sleeve results
each sleeve notebook already documents.

Every weight at date t is built from data strictly before t-1 (rolling
windows are shifted by one day before being multiplied into the sleeve
returns), which keeps the combiner free of look-ahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

from helper import port_ret, ic_weighted_combine, stats


# ---------------------------------------------------------------------------
# Sleeve loading
# ---------------------------------------------------------------------------

# Relative to repo root. The module is imported from the repo root and from
# the `research/` notebook directory, so we resolve paths against this file.
_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"

# Costs and execution lag match the per-sleeve notebooks. The equity
# sleeve was built with exec_lag=2 and 5 bps in notebook 04, so we
# reproduce those choices here. Changing them silently would mean the
# combined PnL is no longer a faithful sum of the per-sleeve PnLs.
_EQUITY_TCOST_BPS = 5
_EQUITY_EXEC_LAG = 2


def _equity_sleeve_pnl() -> pd.Series:
    """
    Recompute the equity sleeve PnL as the IC-weighted mean of the three
    per-signal PnL streams. Notebook 04 builds a single combined-signal
    optimization, but it does not persist the result; the IC-weighted
    mean of the saved per-signal PnLs is a defensible proxy that uses
    exactly the artefacts the upstream notebook does save and inherits
    the same look-ahead guarantees from `ic_weighted_combine`.
    """
    raw = pd.read_parquet(_DATA_DIR / "sp500.parquet")
    close = raw["Close"]
    returns = close.pct_change()

    labels = ["MeanRev", "Momentum", "LowVol"]
    names = ["01_mean_reversion", "02_momentum", "03_low_volatility"]
    signals_raw = {lab: pd.read_parquet(_DATA_DIR / f"signal_{n}.parquet")
                   for lab, n in zip(labels, names)}
    weights_oos = {lab: pd.read_parquet(_DATA_DIR / f"weights_{n}_oos.parquet")
                   for lab, n in zip(labels, names)}

    # IC weights are computed on the raw per-signal panels using the same
    # 6-month lookback as notebook 04. The combiner shifts the IC by the
    # forward-return horizon to avoid leak.
    _, ic_w = ic_weighted_combine(signals_raw, close, lookback=126,
                                  min_ic=0.0, horizon=1)
    # Normalize the per-signal weight panel to a daily weight per sleeve.
    daily_w = ic_w.fillna(0.0)
    row_sum = daily_w.sum(axis=1).replace(0, np.nan)
    daily_w = daily_w.divide(row_sum, axis=0).fillna(1.0 / len(labels))

    # Per-signal PnL stream, then weighted sum. Each sleeve PnL is
    # already net of 5 bps and uses exec_lag=2 so the IC-weighted mean
    # inherits the same execution assumptions.
    pnls = pd.DataFrame({
        lab: port_ret(weights_oos[lab], returns,
                      tcost_bps=_EQUITY_TCOST_BPS,
                      exec_lag=_EQUITY_EXEC_LAG)
        for lab in labels
    })
    pnls = pnls.dropna(how="any")
    daily_w = daily_w.reindex(pnls.index).ffill().fillna(1.0 / len(labels))
    combined = (pnls * daily_w[labels]).sum(axis=1)
    combined.name = "equity_combined"
    return combined


def load_sleeve_pnls() -> dict[str, pd.Series]:
    """
    Return the three sleeve PnL streams aligned on the intersection of
    their daily indices. Equity is recomputed from saved weights so the
    module has no hidden dependency on a parquet that the upstream
    notebook never wrote.
    """
    equity = _equity_sleeve_pnl()

    crypto = pd.read_parquet(_DATA_DIR / "pnl_07_crypto_volmom.parquet").squeeze()
    crypto.name = "crypto_volmom"

    pairs = pd.read_parquet(_DATA_DIR / "pnl_09_pairs_oos.parquet").squeeze()
    pairs.name = "pairs"

    # Inner-join on the date index so every rule sees the same date set.
    # The three streams trade on different calendars (crypto is 7-day,
    # equity and pairs are business-day); the intersection is the only
    # honest common ground for daily aggregation.
    panel = pd.concat({"equity": equity, "crypto": crypto, "pairs": pairs},
                      axis=1, join="inner").dropna()
    return {name: panel[name].copy() for name in panel.columns}


# ---------------------------------------------------------------------------
# Combination rules
# ---------------------------------------------------------------------------

def _panel_from_dict(pnls: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack the sleeve dict into a single date-indexed frame, inner join."""
    df = pd.concat(pnls, axis=1, join="inner").dropna()
    # Preserve insertion order — `pd.concat` does this by default — so
    # downstream tables are stable across runs.
    return df


def combine_equal_weight(pnls: dict[str, pd.Series]) -> pd.Series:
    """
    Equal-weight combined PnL: w_i = 1/N applied to every sleeve every
    day. Weights are constants, so the one-day shift is vacuous, but we
    keep the column-mean form so this function returns the same index
    behaviour as the other two rules.
    """
    df = _panel_from_dict(pnls)
    out = df.mean(axis=1)
    out.name = "equal_weight"
    return out


def combine_inverse_vol(pnls: dict[str, pd.Series],
                        lookback: int = 60) -> pd.Series:
    """
    Inverse-vol (risk-parity) combined PnL.

    Each sleeve gets weight proportional to 1 / sigma_i, where sigma_i
    is the trailing `lookback`-day realised standard deviation of that
    sleeve's PnL. The weight panel is shifted by one day before being
    multiplied into the sleeve returns: a weight computed using returns
    up to and including day t is therefore only applied to the return
    on day t+1, which is the standard no-look-ahead convention.
    """
    df = _panel_from_dict(pnls)
    # min_periods=lookback drops the warm-up rather than silently using
    # a half-formed window; alternative was a smaller min_periods, but
    # we prefer the clean burn-in for a portfolio-level diagnostic.
    sigma = df.rolling(lookback, min_periods=lookback).std()
    inv_vol = 1.0 / sigma.replace(0, np.nan)
    w = inv_vol.divide(inv_vol.sum(axis=1), axis=0)
    held = w.shift(1)
    out = (held * df).sum(axis=1, min_count=1)
    out.name = "inverse_vol"
    return out.dropna()


def combine_min_variance(pnls: dict[str, pd.Series],
                         lookback: int = 252) -> pd.Series:
    """
    Long-only minimum-variance combined PnL.

    Solve `min w' Sigma w` subject to `sum w = 1, w >= 0` where Sigma
    is the sample covariance of sleeve PnLs over the trailing `lookback`
    days. The QP is small (N=3 here), so we solve it once per day with
    cvxpy. Weights at date t use only data up to and including t-1
    (we slice `df.iloc[i - lookback:i]`), and we additionally shift the
    weight panel by one day before applying it to returns so the
    realised weight on day t is the weight that was computed using
    returns ≤ t-1 — belt and braces.
    """
    df = _panel_from_dict(pnls)
    n = df.shape[1]
    weights = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    # Pre-declare the cvxpy problem once and update the cov parameter
    # each day. Building a fresh Problem inside the loop would burn
    # most of the runtime on parsing, not solving.
    cov_p = cp.Parameter((n, n), PSD=True)
    w_v = cp.Variable(n)
    constraints = [cp.sum(w_v) == 1, w_v >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w_v, cov_p)), constraints)

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        cov = window.cov().values
        # Symmetrize defensively; floating-point asymmetry trips up the
        # PSD parameter validation on rare days.
        cov = 0.5 * (cov + cov.T)
        cov_p.value = cov
        try:
            prob.solve(solver=cp.SCS, warm_start=True)
        except cp.SolverError:
            continue
        if w_v.value is None:
            continue
        w_today = np.clip(w_v.value, 0, None)
        s = w_today.sum()
        if s > 0:
            w_today = w_today / s
        weights.iloc[i] = w_today

    held = weights.shift(1)
    out = (held * df).sum(axis=1, min_count=1)
    out.name = "min_variance"
    return out.dropna()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_combination(combined_pnl: pd.Series,
                         label: str,
                         benchmark: pd.Series | None = None,
                         periods_per_year: int = 252) -> dict:
    """
    Headline stats for a single combined PnL stream.

    Returns a flat dict so the caller can build a comparison table with
    `pd.DataFrame.from_records`. HAC t-stat against the benchmark is
    populated when `benchmark` is supplied. Max drawdown is computed
    in compounded space to match `helper.stats`.
    """
    r = combined_pnl.dropna()
    summary = stats(r, benchmark=benchmark, plot=False,
                    periods_per_year=periods_per_year, hac_lags=5)

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    out = {
        "label": label,
        "sharpe": float(summary["sharpe"].iloc[0]),
        "ann_vol": summary["volatility_annual"].iloc[0],
        "ann_return": summary["mean_return_annual"].iloc[0],
        "max_dd": f"{dd.min() * 100:.2f}%",
        "skew": round(float(r.skew()), 3),
        "kurt": round(float(r.kurt()), 3),
        "n_days": int(len(r)),
    }
    if benchmark is not None and "alpha_tstat" in summary.columns:
        out["alpha_annual"] = summary["alpha_annual"].iloc[0]
        out["alpha_tstat_hac"] = float(summary["alpha_tstat"].iloc[0])
        out["beta"] = float(summary["beta"].iloc[0])
    return out
