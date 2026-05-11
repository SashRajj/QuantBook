"""
Cross-sectional equity signals.

Each function takes a price panel (dates x tickers) and returns a signal
panel of the same shape. Larger values are a stronger long view. All
signals use trailing windows only.
"""

import numpy as np
import pandas as pd


def mean_reversion(close, lookback=20):
    """Negated rolling z-score of price over `lookback` days."""
    mu = close.rolling(lookback, min_periods=lookback).mean()
    sigma = close.rolling(lookback, min_periods=lookback).std()
    return -(close - mu) / sigma


def momentum(close, lookback=252, skip=21):
    """
    Total return from t-lookback to t-skip.

    Defaults give the standard 12-1 momentum from Jegadeesh and Titman
    (1993): skip the most recent month to avoid short-term reversal.
    """
    return close.shift(skip) / close.shift(lookback) - 1


def low_volatility(close, lookback=63):
    """Negated rolling realized volatility over `lookback` days."""
    rets = close.pct_change()
    vol = rets.rolling(lookback, min_periods=lookback).std()
    return -vol


def market_residual_momentum(close, market_close, lookback=252, skip=21,
                             beta_lookback=252):
    """
    Sum of market-residual daily returns over the formation window.

    Per Blitz, Huij, and Martens (2011): regress each stock's return on
    the market, take the residual, and use the cumulative residual over
    the trailing 12 minus 1 month window as the momentum signal. Strips
    out the market-beta component that drives most momentum crashes at
    regime transitions.

    `market_close` is a Series of the benchmark adjusted close.
    """
    rets = close.pct_change()
    mkt = market_close.reindex(close.index).pct_change()

    # Rolling per-name beta to the market.
    cov = rets.rolling(beta_lookback, min_periods=beta_lookback).cov(mkt)
    var = mkt.rolling(beta_lookback, min_periods=beta_lookback).var()
    beta = cov.divide(var, axis=0)

    # Residual returns; broadcasting beta * mkt_t against each column.
    resid = rets.subtract(beta.multiply(mkt, axis=0), axis=0)

    # Cumulative residual return over the 12-1 month window.
    cum_to_skip = resid.shift(skip).rolling(lookback - skip,
                                            min_periods=lookback - skip).sum()
    return cum_to_skip
