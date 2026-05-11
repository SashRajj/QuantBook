"""Signal evaluation, portfolio construction, and risk diagnostics."""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as sp_stats
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """
    Per-date mean-variance optimization.

    Signals are cross-sectionally rank-normalized then standardized.
    Covariance is estimated on the training panel with Ledoit-Wolf
    shrinkage; missing return entries are replaced with each name's
    in-window mean so that delisted or newly-listed stocks do not pull
    correlations toward zero.

    A `member_mask` panel (date x ticker, boolean) can be passed to
    `run` to enforce point-in-time index membership: names that were
    not in the index on a given date are constrained to zero weight.
    """

    def __init__(self, signal_df, returns_df):
        signal_df, returns_df = signal_df.align(returns_df, join="inner", axis=1)
        if signal_df.empty or returns_df.empty:
            raise ValueError("signal_df and returns_df must share at least one column")

        clean_signal = signal_df.fillna(0).replace([np.inf, -np.inf], 0)
        ranked = clean_signal.rank(axis=1, pct=True) - 0.5
        self.alpha_df = (ranked.subtract(ranked.mean(axis=1), axis=0)
                               .divide(ranked.std(axis=1) + 1e-6, axis=0))

        ret_clean = returns_df.replace([np.inf, -np.inf], np.nan)
        col_means = ret_clean.mean()
        ret_filled = ret_clean.fillna(col_means).fillna(0).values
        self.cov = LedoitWolf().fit(ret_filled).covariance_
        self.columns = self.alpha_df.columns

    def run(self, dollar_neutral=True, fully_invested=False,
            max_position=None, long_only=False, max_leverage=None,
            target_vol=None, tcost_penalty_bps=0, subsample=1,
            member_mask=None, verbose=False):
        """
        Solve the per-date QP and return a weights DataFrame.

        Built once with `cp.Parameter` for alpha, previous weights, and
        per-name position bounds; the date loop only updates parameter
        values and re-solves with warm-starting.

        `member_mask` is an optional (date x ticker) boolean panel
        marking which names are in the eligible universe on each date.
        Non-members on a given date are pinned to zero weight via the
        per-name bound parameters.
        """
        if long_only and dollar_neutral:
            raise ValueError("long_only=True is incompatible with dollar_neutral=True")

        lam = tcost_penalty_bps / 10000
        n = len(self.columns)

        alpha_p = cp.Parameter(n)
        w_prev_p = cp.Parameter(n) if lam > 0 else None
        if w_prev_p is not None:
            w_prev_p.value = np.zeros(n)

        # Per-name bounds are always present so the same problem object
        # can serve all callers; the default cap is large enough not to
        # bind when max_position is not set.
        default_cap = 1.0 if max_position is None else max_position
        ub_p = cp.Parameter(n)
        lb_p = cp.Parameter(n)
        ub_p.value = np.full(n, default_cap)
        lb_p.value = np.full(n, 0.0 if long_only else -default_cap)

        w = cp.Variable(n)
        c = [w <= ub_p, w >= lb_p]
        if dollar_neutral:
            c.append(cp.sum(w) == 0)
        if fully_invested:
            c.append(cp.norm(w, 1) <= 1.0)
        if max_leverage:
            c.append(cp.norm(w, 1) <= max_leverage)

        obj = -alpha_p @ w + cp.quad_form(w, cp.psd_wrap(self.cov))
        if lam > 0:
            obj += lam * cp.norm(w - w_prev_p, 1)
        prob = cp.Problem(cp.Minimize(obj), c)

        if member_mask is not None:
            mask_aligned = (member_mask
                            .reindex(index=self.alpha_df.index, columns=self.columns)
                            .fillna(False)
                            .astype(bool))
        else:
            mask_aligned = None

        dates = self.alpha_df.index[::subsample]
        weights = {}
        w_old = np.zeros(n)

        for i, date in enumerate(dates):
            alpha = self.alpha_df.loc[date].values.astype(float)
            if np.any(np.isnan(alpha)) or np.sum(np.abs(alpha)) == 0:
                continue

            if mask_aligned is not None:
                mask_today = mask_aligned.loc[date].values
                if not mask_today.any():
                    continue
                # Names outside the universe today get a zero alpha and a
                # zero bound on both sides.
                alpha = np.where(mask_today, alpha, 0.0)
                ub_p.value = np.where(mask_today, default_cap, 0.0)
                lb_p.value = np.where(mask_today,
                                      0.0 if long_only else -default_cap,
                                      0.0)

            alpha_p.value = alpha
            if w_prev_p is not None:
                w_prev_p.value = w_old
            try:
                prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
            except cp.SolverError:
                continue
            if w.value is None:
                continue

            w_old = np.asarray(w.value).flatten()
            if fully_invested:
                s = np.sum(np.abs(w_old))
                if s > 0:
                    w_old = w_old / s
            if target_vol:
                port_vol = np.sqrt(w_old @ self.cov @ w_old) * np.sqrt(252)
                if port_vol > 0:
                    w_old = w_old * (target_vol / port_vol)
            weights[date] = pd.Series(w_old, index=self.columns)
            if verbose and (i + 1) % 200 == 0:
                print(f"  solved {i+1}/{len(dates)}")

        out = pd.DataFrame(weights).T
        if subsample > 1 and len(out) > 0:
            out = out.reindex(self.alpha_df.index).ffill()
        return out


# ---------------------------------------------------------------------------
# Backtest plumbing
# ---------------------------------------------------------------------------

def port_ret(weights_df, returns_df, tcost_bps=0, tcost_short_bps=None,
             borrow_bps_annual=0, exec_lag=2):
    """
    Realized portfolio return with execution lag and asymmetric costs.

    `exec_lag` is the number of days between signal observation and
    return realization. Default 2 reflects the realistic case where a
    signal computed at close of day t is acted on at close of t+1 and
    earns the t+1 to t+2 return. Use `exec_lag=1` for diagnostics that
    assume zero-latency execution.

    `tcost_bps` charges per unit of turnover on long-side position
    changes; `tcost_short_bps` (default: same as `tcost_bps`) applies to
    short-side changes. `borrow_bps_annual` is an annualised holding fee
    on the short book, paid daily.
    """
    if tcost_short_bps is None:
        tcost_short_bps = tcost_bps

    held = weights_df.shift(exec_lag)
    gross = (held * returns_df).sum(axis=1, min_count=1)

    longs_now = weights_df.clip(lower=0)
    longs_prev = weights_df.shift(1).clip(lower=0)
    long_turn = (longs_now - longs_prev).abs().sum(axis=1)

    shorts_now = (-weights_df).clip(lower=0)
    shorts_prev = (-weights_df).shift(1).clip(lower=0)
    short_turn = (shorts_now - shorts_prev).abs().sum(axis=1)

    tc = (long_turn * tcost_bps + short_turn * tcost_short_bps) / 10000

    borrow = shorts_now.sum(axis=1).shift(exec_lag) * borrow_bps_annual / 252 / 10000
    borrow = borrow.fillna(0)

    return gross - tc - borrow


def quick_weights(signal_df, dollar_neutral=True, long_only=False,
                  fully_invested=True, max_position=None):
    """Rank-based weights with no optimization. Used as a sanity check."""
    if long_only and dollar_neutral:
        raise ValueError("long_only=True is incompatible with dollar_neutral=True")

    weights = signal_df.rank(axis=1, pct=True)
    if long_only:
        weights = weights.clip(lower=0)
    if dollar_neutral:
        weights = weights.subtract(weights.mean(axis=1), axis=0)
    if max_position:
        weights = weights.clip(lower=-max_position, upper=max_position)
    if fully_invested:
        weights = weights.divide(weights.abs().sum(axis=1), axis=0)
    return weights


# ---------------------------------------------------------------------------
# Performance and risk statistics
# ---------------------------------------------------------------------------

def stats(port_ret, weights=None, benchmark=None, plot=True, periods_per_year=252):
    """
    Headline performance stats with an optional alpha/beta regression.

    `periods_per_year` controls the annualisation factor: 252 for daily
    equity returns, 365 for daily crypto.
    """
    cum = (1 + port_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    sigma = port_ret.std()
    ppy = periods_per_year
    sharpe = round(port_ret.mean() / sigma * np.sqrt(ppy), 3) if sigma > 0 else np.nan
    t_stat = round(port_ret.mean() / (sigma / np.sqrt(len(port_ret))), 3) if sigma > 0 else np.nan

    s = {
        "mean_return_annual": f"{port_ret.mean() * ppy * 100:.2f}%",
        "volatility_annual": f"{port_ret.std() * np.sqrt(ppy) * 100:.2f}%",
        "sharpe": sharpe,
        "t_stat": t_stat,
        "max_drawdown": f"{dd.min() * 100:.2f}%",
        "avg_drawdown": f"{dd.mean() * 100:.2f}%",
        "max_dd_duration": f"{(dd < 0).astype(int).groupby((dd == 0).cumsum()).sum().max()} days",
    }

    if benchmark is not None:
        df = pd.concat([port_ret, benchmark], axis=1).dropna()
        X = sm.add_constant(df.iloc[:, 1])
        model = sm.OLS(df.iloc[:, 0], X).fit()
        s["alpha_annual"] = f"{model.params.iloc[0] * ppy * 100:.2f}%"
        s["alpha_tstat"] = round(model.tvalues.iloc[0], 3)
        s["beta"] = round(model.params.iloc[1], 3)

    if weights is not None:
        s["daily_turnover"] = f"{weights.diff().abs().sum(axis=1).mean() * 100:.2f}%"

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(cum, label="Strategy")
        if benchmark is not None:
            plt.plot((1 + benchmark).cumprod(), label="Benchmark", alpha=0.7)
        plt.legend()
        plt.title("Cumulative Returns")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(s, index=["Strategy"])


def ic(signal_df, prices_df, horizons=(1, 5, 10, 20)):
    """Spearman rank IC and ICIR at each forward-return horizon."""
    results = {}
    for h in horizons:
        fwd = prices_df.pct_change(h).shift(-h)
        corr = signal_df.corrwith(fwd, axis=1, method="spearman")
        results[h] = {"IC": round(corr.mean(), 4),
                      "ICIR": round(corr.mean() / corr.std(), 4)}
    return pd.DataFrame(results).T


def ic_weighted_combine(signal_dict, prices_df, lookback=126, min_ic=0.0,
                        horizon=1):
    """
    Combine signals with weights proportional to trailing IC.

    For each date, the weight on signal s is the rolling mean of its
    cross-sectional Spearman IC over the last `lookback` days (shifted
    by one day so the IC computed today is not used to weight today's
    signal). Signals with mean IC below `min_ic` get zero weight.

    Returns a single combined signal DataFrame aligned to the union of
    the input signals' indices and columns.
    """
    fwd = prices_df.pct_change(horizon).shift(-horizon)

    # Align all signals to a shared index/columns.
    idx = None
    cols = None
    for sig in signal_dict.values():
        idx = sig.index if idx is None else idx.union(sig.index)
        cols = sig.columns if cols is None else cols.union(sig.columns)
    aligned = {name: sig.reindex(index=idx, columns=cols)
               for name, sig in signal_dict.items()}

    # Trailing IC per signal, shifted to avoid look-ahead in the weighting.
    weights = {}
    for name, sig in aligned.items():
        ic_t = sig.corrwith(fwd.reindex(index=idx, columns=cols),
                            axis=1, method="spearman")
        weights[name] = (ic_t.shift(1)
                         .rolling(lookback, min_periods=lookback // 2)
                         .mean()
                         .clip(lower=min_ic))
    w_df = pd.DataFrame(weights, index=idx).fillna(0.0)

    # Cross-sectional z-score per signal so they are on comparable scales.
    def _zscore(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        return df.subtract(mu, axis=0).divide(sd, axis=0)

    z = {name: _zscore(sig) for name, sig in aligned.items()}

    # Combine: weighted sum, then normalise weights so the combined
    # signal scale is comparable across dates.
    w_sum = w_df.sum(axis=1).replace(0, np.nan)
    w_norm = w_df.divide(w_sum, axis=0)

    combined = sum(z[name].multiply(w_norm[name], axis=0) for name in z)
    return combined, w_norm


# ---------------------------------------------------------------------------
# Adjusted Sharpe statistics
# ---------------------------------------------------------------------------

_EULER_MASCHERONI = 0.5772156649

def probabilistic_sharpe(returns, target_sharpe=0.0, periods_per_year=252):
    """
    Probabilistic Sharpe Ratio (Bailey and Lopez de Prado, 2012).

    Probability that the true annualised Sharpe exceeds `target_sharpe`,
    accounting for finite-sample noise plus the skew and excess kurtosis
    of the realised return distribution.
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return np.nan
    sr_p = r.mean() / r.std()
    target_p = target_sharpe / np.sqrt(periods_per_year)
    skew = r.skew()
    # pandas kurt returns excess kurtosis; the PSR formula uses non-excess.
    kurt = r.kurt() + 3
    var_sr = (1 - skew * sr_p + ((kurt - 1) / 4.0) * sr_p ** 2) / (T - 1)
    sigma_sr = np.sqrt(max(var_sr, 1e-12))
    return float(sp_stats.norm.cdf((sr_p - target_p) / sigma_sr))


def deflated_sharpe(returns, n_trials, trial_sharpe_std=0.5, periods_per_year=252):
    """
    Deflated Sharpe Ratio (Bailey and Lopez de Prado, 2014).

    The Probabilistic Sharpe Ratio computed against a non-zero target
    that reflects the expected maximum annualised Sharpe under the null
    hypothesis after exploring `n_trials` candidate configurations on
    the same data. `trial_sharpe_std` is the cross-trial annualised
    Sharpe standard deviation; pass the empirical value from the grid
    if available, otherwise the default 0.5 is a reasonable prior.
    """
    r = pd.Series(returns).dropna()
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0, periods_per_year)
    z1 = sp_stats.norm.ppf(1 - 1 / n_trials)
    z2 = sp_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr_null = trial_sharpe_std * ((1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)
    return probabilistic_sharpe(r, sr_null, periods_per_year)


# ---------------------------------------------------------------------------
# Distribution and tail-risk diagnostics
# ---------------------------------------------------------------------------

def dist_plot(returns, title="Return distribution"):
    """Histogram with normal overlay and a QQ plot against the normal."""
    r = returns.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.linspace(r.min(), r.max(), 400)
    axes[0].hist(r, bins=80, density=True, alpha=0.55, label="Empirical")
    axes[0].plot(x, sp_stats.norm.pdf(x, r.mean(), r.std()), "r", label="Normal")
    axes[0].set_title(f"{title}  (skew={r.skew():.2f}, kurt={r.kurt():.2f})")
    axes[0].legend()

    sp_stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ plot vs Normal")

    plt.tight_layout()
    plt.show()


def var_cvar(returns, alpha=0.05):
    """
    Historical and parametric (Gaussian) VaR and CVaR at level alpha.

    Historical is the empirical alpha-quantile and conditional mean below
    it. Parametric assumes Gaussian returns and will understate tail risk
    when skewness or excess kurtosis is large.
    """
    r = returns.dropna()
    hist_var = r.quantile(alpha)
    hist_cvar = r[r <= hist_var].mean()
    z = sp_stats.norm.ppf(alpha)
    param_var = r.mean() + z * r.std()
    param_cvar = r.mean() - r.std() * sp_stats.norm.pdf(z) / alpha
    out = pd.DataFrame({
        "VaR": [hist_var, param_var],
        "CVaR": [hist_cvar, param_cvar],
    }, index=["historical", "parametric"])
    return out.apply(lambda col: col.map(lambda x: f"{x*100:.2f}%"))


# ---------------------------------------------------------------------------
# Factor neutralization
# ---------------------------------------------------------------------------

def beta_to(returns_df, benchmark, lookback=252):
    """Rolling per-stock beta to a benchmark, cov / var."""
    b_aligned = benchmark.reindex(returns_df.index)
    cov = returns_df.rolling(lookback).cov(b_aligned)
    var = b_aligned.rolling(lookback).var()
    return cov.divide(var, axis=0)


def neutralize(signal_df, factor_df):
    """
    Cross-sectional residual of the signal after regressing on one factor.

    Per date, run OLS of the signal on the factor across stocks and
    replace the signal with the residual. Used to strip out market-beta
    exposure.
    """
    s, f = signal_df.align(factor_df, join="inner")
    out = pd.DataFrame(index=s.index, columns=s.columns, dtype=float)
    for date in s.index:
        y = s.loc[date].values
        x = f.loc[date].values
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 30:
            continue
        X = sm.add_constant(x[mask])
        beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        resid = y[mask] - X @ beta
        row = np.full_like(y, np.nan, dtype=float)
        row[mask] = resid
        out.loc[date] = row
    return out


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def walk_forward_splits(index, train_years=5, test_years=1):
    """
    Yield rolling (train_slice, test_slice) pairs.

    Test windows are disjoint from their training windows and roll
    forward by `test_years` each step.
    """
    index = pd.DatetimeIndex(index)
    start = index.min()
    end = index.max()
    cur = start + pd.DateOffset(years=train_years)
    while cur + pd.DateOffset(years=test_years) <= end:
        train = slice(cur - pd.DateOffset(years=train_years), cur - pd.Timedelta(days=1))
        test = slice(cur, cur + pd.DateOffset(years=test_years) - pd.Timedelta(days=1))
        yield train, test
        cur = cur + pd.DateOffset(years=test_years)
