"""
End-to-end evaluation of the idiosyncratic volatility signal.

Mirrors the in-sample / out-of-sample protocol used by the other
linear-factor notebooks (01-03): point-in-time membership masking,
exec_lag=2, 5 bps/side costs, default trailing-window Ledoit-Wolf cov.
No parameter tuning is done here — lookback=60 and beta_lookback=252
are fixed as specified by the task.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from helper import Optimizer, ic, port_ret, stats
from signals import idiosyncratic_volatility


IS_END = "2020-12-31"
DATA = REPO / "data"


def main():
    # ------------------------------------------------------------------
    # Load data.
    # ------------------------------------------------------------------
    sp500 = pd.read_parquet(DATA / "sp500.parquet")
    close = sp500["Close"].copy()
    spy = pd.read_parquet(DATA / "spy.parquet")["Close"].squeeze()
    members = pd.read_parquet(DATA / "members.parquet")

    # Align spy to the equity calendar.
    spy = spy.reindex(close.index).ffill()
    spy_ret = spy.pct_change()

    returns = close.pct_change()

    print(f"close: {close.shape}, spy: {spy.shape}, members: {members.shape}")
    print(f"date range: {close.index.min().date()} to {close.index.max().date()}")

    # ------------------------------------------------------------------
    # Compute signal.
    # ------------------------------------------------------------------
    signal = idiosyncratic_volatility(close, spy,
                                      lookback=60, beta_lookback=252)
    print(f"signal: {signal.shape}, first non-NaN row at "
          f"{signal.dropna(how='all').index[0].date()}")

    # ------------------------------------------------------------------
    # Information coefficient.
    # ------------------------------------------------------------------
    ic_full = ic(signal, close, horizons=(1, 5, 10, 20))
    print("\n=== IC (full sample) ===")
    print(ic_full)

    ic_is = ic(signal.loc[:IS_END], close.loc[:IS_END], horizons=(1, 5, 10, 20))
    print("\n=== IC (in-sample, <=2020-12-31) ===")
    print(ic_is)

    ic_oos = ic(signal.loc[IS_END:].iloc[1:],
                close.loc[IS_END:].iloc[1:], horizons=(1, 5, 10, 20))
    print("\n=== IC (out-of-sample, >2020-12-31) ===")
    print(ic_oos)

    # ------------------------------------------------------------------
    # IS / OOS split.
    # ------------------------------------------------------------------
    is_signal = signal.loc[:IS_END].dropna(how="all")
    is_returns = returns.reindex(is_signal.index)

    oos_signal = signal.loc[IS_END:].iloc[1:].dropna(how="all")
    oos_returns = returns.reindex(oos_signal.index)

    print(f"\nIS dates: {is_signal.index.min().date()} to {is_signal.index.max().date()} "
          f"({len(is_signal)} rows)")
    print(f"OOS dates: {oos_signal.index.min().date()} to {oos_signal.index.max().date()} "
          f"({len(oos_signal)} rows)")

    # ------------------------------------------------------------------
    # In-sample optimization.
    # ------------------------------------------------------------------
    print("\n=== Optimizing IS ===")
    opt_is = Optimizer(is_signal, is_returns)
    w_is = opt_is.run(dollar_neutral=True, max_position=0.02,
                      max_leverage=1.0, tcost_penalty_bps=5,
                      subsample=5, member_mask=members, verbose=True)
    print(f"IS weights: {w_is.shape}")

    pnl_is = port_ret(w_is, is_returns, tcost_bps=5, exec_lag=2)
    stats_is = stats(pnl_is.dropna(), weights=w_is,
                     benchmark=spy_ret.loc[:IS_END],
                     plot=False, hac_lags=5)
    print("\n=== Stats IS ===")
    print(stats_is.T)

    # ------------------------------------------------------------------
    # Out-of-sample optimization.
    # ------------------------------------------------------------------
    print("\n=== Optimizing OOS ===")
    opt_oos = Optimizer(oos_signal, oos_returns)
    w_oos = opt_oos.run(dollar_neutral=True, max_position=0.02,
                        max_leverage=1.0, tcost_penalty_bps=5,
                        subsample=5, member_mask=members, verbose=True)
    print(f"OOS weights: {w_oos.shape}")

    pnl_oos = port_ret(w_oos, oos_returns, tcost_bps=5, exec_lag=2)
    stats_oos = stats(pnl_oos.dropna(), weights=w_oos,
                      benchmark=spy_ret.loc[oos_signal.index.min():],
                      plot=False, hac_lags=5)
    print("\n=== Stats OOS ===")
    print(stats_oos.T)

    # ------------------------------------------------------------------
    # Higher moments + max DD for both halves.
    # ------------------------------------------------------------------
    pnl_is_clean = pnl_is.dropna()
    pnl_oos_clean = pnl_oos.dropna()

    def dd(p):
        cum = (1 + p).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    print("\n=== Higher moments + max DD ===")
    print(f"IS  skew={pnl_is_clean.skew():.3f}  "
          f"excess_kurt={pnl_is_clean.kurt():.3f}  "
          f"max_dd={dd(pnl_is_clean) * 100:.2f}%")
    print(f"OOS skew={pnl_oos_clean.skew():.3f}  "
          f"excess_kurt={pnl_oos_clean.kurt():.3f}  "
          f"max_dd={dd(pnl_oos_clean) * 100:.2f}%")

    # ------------------------------------------------------------------
    # Persist artifacts.
    # ------------------------------------------------------------------
    w_is.to_parquet(DATA / "weights_10_idio_vol_is.parquet")
    w_oos.to_parquet(DATA / "weights_10_idio_vol_oos.parquet")
    pnl_combined = pd.concat([pnl_is_clean, pnl_oos_clean])
    pnl_combined.name = "pnl"
    pnl_combined.to_frame().to_parquet(DATA / "pnl_10_idio_vol.parquet")

    print("\nWrote:")
    print(f"  {DATA / 'weights_10_idio_vol_is.parquet'}")
    print(f"  {DATA / 'weights_10_idio_vol_oos.parquet'}")
    print(f"  {DATA / 'pnl_10_idio_vol.parquet'}")

    # ------------------------------------------------------------------
    # Compact headline summary.
    # ------------------------------------------------------------------
    def get(df, k):
        return df.iloc[0].get(k, np.nan)

    print("\n=== HEADLINE ===")
    for h in (1, 5, 10, 20):
        print(f"  IC_{h:>2}={ic_full.loc[h, 'IC']:+.4f}  "
              f"ICIR_{h:>2}={ic_full.loc[h, 'ICIR']:+.4f}")
    print(f"  Sharpe IS  = {get(stats_is, 'sharpe')}")
    print(f"  Sharpe OOS = {get(stats_oos, 'sharpe')}")
    print(f"  HAC alpha t IS  = {get(stats_is, 'alpha_tstat')}")
    print(f"  HAC alpha t OOS = {get(stats_oos, 'alpha_tstat')}")
    print(f"  OOS skew = {pnl_oos_clean.skew():.3f}")
    print(f"  OOS excess kurt = {pnl_oos_clean.kurt():.3f}")
    print(f"  IS  max DD = {dd(pnl_is_clean) * 100:.2f}%")
    print(f"  OOS max DD = {dd(pnl_oos_clean) * 100:.2f}%")
    t_oos = get(stats_oos, "alpha_tstat")
    try:
        passes = abs(float(t_oos)) > 2.0
    except (TypeError, ValueError):
        passes = False
    print(f"  |HAC alpha t OOS| > 2 ?  {'YES' if passes else 'NO'}")


if __name__ == "__main__":
    main()
