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
    Covariance is estimated once on the training panel with Ledoit-Wolf
    shrinkage.
    """

    def __init__(self, signal_df, returns_df):
        signal_df, returns_df = signal_df.align(returns_df, join="inner", axis=1)
        if signal_df.empty or returns_df.empty:
            raise ValueError("signal_df and returns_df must share at least one column")

        clean_signal = signal_df.fillna(0).replace([np.inf, -np.inf], 0)
        ranked = clean_signal.rank(axis=1, pct=True) - 0.5
        self.alpha_df = (ranked.subtract(ranked.mean(axis=1), axis=0)
                               .divide(ranked.std(axis=1) + 1e-6, axis=0))
        self.cov = LedoitWolf().fit(
            returns_df.fillna(0).replace([np.inf, -np.inf], 0).values
        ).covariance_

    def run(self, dollar_neutral=True, fully_invested=False,
            max_position=None, long_only=False, max_leverage=None,
            target_vol=None, tcost_penalty_bps=0, subsample=1, verbose=False):
        """
        Solve the per-date QP and return a weights DataFrame.

        The cvxpy problem is built once with `cp.Parameter` for alpha and
        previous weights; the date loop only updates parameter values and
        re-solves with warm-starting. `tcost_penalty_bps` adds an L1
        turnover penalty inside the objective; `subsample=k` solves every
        k-th date and forward-fills weights in between.
        """
        if long_only and dollar_neutral:
            raise ValueError("long_only=True is incompatible with dollar_neutral=True")

        lam = tcost_penalty_bps / 10000
        n = len(self.alpha_df.columns)

        alpha_p = cp.Parameter(n)
        w_prev_p = cp.Parameter(n) if lam > 0 else None
        if w_prev_p is not None:
            w_prev_p.value = np.zeros(n)

        w = cp.Variable(n)
        c = []
        if dollar_neutral:
            c.append(cp.sum(w) == 0)
        if fully_invested:
            c.append(cp.norm(w, 1) <= 1.0)
        if max_leverage:
            c.append(cp.norm(w, 1) <= max_leverage)
        if max_position:
            c.append(w <= max_position)
            c.append(w >= -max_position)
        if long_only:
            c.append(w >= 0)

        obj = -alpha_p @ w + cp.quad_form(w, cp.psd_wrap(self.cov))
        if lam > 0:
            obj += lam * cp.norm(w - w_prev_p, 1)
        prob = cp.Problem(cp.Minimize(obj), c)

        dates = self.alpha_df.index[::subsample]
        weights = {}
        w_old = np.zeros(n)

        for i, date in enumerate(dates):
            alpha = self.alpha_df.loc[date].values.astype(float)
            if np.any(np.isnan(alpha)) or np.sum(np.abs(alpha)) == 0:
                continue
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
            weights[date] = pd.Series(w_old, index=self.alpha_df.columns)
            if verbose and (i + 1) % 200 == 0:
                print(f"  solved {i+1}/{len(dates)}")

        out = pd.DataFrame(weights).T
        if subsample > 1 and len(out) > 0:
            out = out.reindex(self.alpha_df.index).ffill()
        return out


# ---------------------------------------------------------------------------
# Backtest plumbing
# ---------------------------------------------------------------------------

def port_ret(weights_df, returns_df, tcost_bps=0):
    """Realized portfolio return. Weights are applied with a one-day lag."""
    ret = (weights_df.shift(1) * returns_df).sum(axis=1)
    if tcost_bps > 0:
        tcost = weights_df.diff().abs().sum(axis=1) * tcost_bps / 10000
        ret = ret - tcost
    return ret


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

def stats(port_ret, weights=None, benchmark=None, plot=True):
    """Headline performance stats with an optional alpha/beta regression."""
    cum = (1 + port_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    s = {
        "mean_return_annual": f"{port_ret.mean() * 252 * 100:.2f}%",
        "volatility_annual": f"{port_ret.std() * np.sqrt(252) * 100:.2f}%",
        "sharpe": round(port_ret.mean() / port_ret.std() * np.sqrt(252), 3),
        "t_stat": round(port_ret.mean() / (port_ret.std() / np.sqrt(len(port_ret))), 3),
        "max_drawdown": f"{dd.min() * 100:.2f}%",
        "avg_drawdown": f"{dd.mean() * 100:.2f}%",
        "max_dd_duration": f"{(dd < 0).astype(int).groupby((dd == 0).cumsum()).sum().max()} days",
    }

    if benchmark is not None:
        df = pd.concat([port_ret, benchmark], axis=1).dropna()
        X = sm.add_constant(df.iloc[:, 1])
        model = sm.OLS(df.iloc[:, 0], X).fit()
        s["alpha_annual"] = f"{model.params.iloc[0] * 252 * 100:.2f}%"
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
