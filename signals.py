"""
Cross-sectional equity signals.

Each function takes a price panel (dates x tickers) and returns a signal
panel of the same shape, where larger values indicate a stronger long view.

Conventions
-----------
- Signals are constructed from adjusted close prices (auto_adjust=True in yfinance),
  so corporate actions are already handled.
- We do not winsorize or rank inside the signal functions. That is the
  optimizer's job, which keeps signal construction interpretable.
- All signals use trailing windows only. There is no look-ahead.
"""

import numpy as np
import pandas as pd


def mean_reversion(close, lookback=20):
    """
    Negated rolling z-score of price.

    A stock far below its rolling mean (negative z) gets a positive signal,
    expressing the view that it will revert upward. The economic rationale
    is short-term overreaction: liquidity demand and noise traders push
    prices away from fundamental value, and these dislocations decay.

    Parameters
    ----------
    close : DataFrame
        Adjusted close prices, dates x tickers.
    lookback : int
        Window length in trading days for the rolling mean and std.
    """
    mu = close.rolling(lookback, min_periods=lookback).mean()
    sigma = close.rolling(lookback, min_periods=lookback).std()
    return -(close - mu) / sigma


def momentum(close, lookback=252, skip=21):
    """
    Cross-sectional momentum: total return over the past `lookback` days,
    skipping the most recent `skip` days.

    The skip removes the short-term reversal effect that contaminates
    the trend signal at the 1-month horizon. Defaults give the standard
    "12 minus 1 month" momentum from Jegadeesh and Titman (1993).

    Parameters
    ----------
    close : DataFrame
        Adjusted close prices, dates x tickers.
    lookback : int
        Total formation window in trading days (default ~12 months).
    skip : int
        Most-recent days to exclude (default ~1 month).
    """
    return close.shift(skip) / close.shift(lookback) - 1


def low_volatility(close, lookback=63):
    """
    Negated rolling realized volatility.

    Low-volatility stocks have historically delivered higher risk-adjusted
    returns than CAPM predicts. Leverage constraints and lottery-preference
    behaviour are the standard explanations. We rank stocks so that low
    realized vol receives the largest signal value.

    Parameters
    ----------
    close : DataFrame
        Adjusted close prices, dates x tickers.
    lookback : int
        Rolling window in trading days (default ~3 months).
    """
    rets = close.pct_change()
    vol = rets.rolling(lookback, min_periods=lookback).std()
    return -vol
