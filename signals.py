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
