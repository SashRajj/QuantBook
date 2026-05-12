"""
Live price feed and feed-health checks.

The research code (yfinance adjusted close) and the broker (Alpaca raw
last trade) are two different sources for the same number. They will
disagree on dividend ex-dates, on split days that haven't been
propagated by one side yet, and during halts. Silently using whichever
one we happened to grab is exactly how a stale-feed bug becomes a wire
transfer.

This module:
  - Pulls the latest close per symbol from yfinance (the same adjusted
    series the backtest used).
  - Reconciles against any reference set (e.g. Alpaca quotes), flagging
    divergences over a configured threshold.
  - Implements a feed-health check that distinguishes "signal is old"
    from "feed is dead". Conflating those two failure modes is how
    silent-flatten bugs land in production books.

Feed health is read at the top of every rebalance. If the latest bar
is older than the configured SLA, the rebalance is hard-rejected with
an alert rather than silently shrinking positions via alpha decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FeedHealth:
    last_bar_ts: datetime
    age_seconds: float
    is_stale: bool
    sla_seconds: float
    n_symbols_received: int
    n_symbols_expected: int


def fetch_latest_close(symbols: list[str], lookback_days: int = 5
                       ) -> pd.DataFrame:
    """
    Pull the most recent adjusted close per symbol via yfinance.

    Returns a DataFrame indexed by symbol with columns:
        close: float    -- the adjusted close
        date:  Timestamp (UTC) -- the date of that bar

    The date is the *actual* yfinance bar timestamp, not the wall-clock
    of the fetch. Without that, downstream feed-health checks measure
    "how long since I called this function" rather than "how stale is
    the data", which is the difference between a working alert and a
    decorative one.

    Symbols with no data in the window are missing from the return
    value. The caller decides how to handle missing names — usually:
    drop from the target weight vector.
    """
    if not symbols:
        return pd.DataFrame(columns=["close", "date"])
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + 7)
    df = yf.download(
        tickers=symbols, start=start, end=end + timedelta(days=1),
        progress=False, auto_adjust=False, threads=True, group_by="ticker",
    )
    out: list[dict] = []
    if isinstance(df.columns, pd.MultiIndex):
        for sym in symbols:
            if sym not in df.columns.get_level_values(0):
                continue
            s = df[sym]["Adj Close"].dropna()
            if not s.empty:
                ts = pd.Timestamp(s.index[-1])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                out.append({"symbol": sym, "close": float(s.iloc[-1]),
                            "date": ts})
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"].dropna()
            if not s.empty:
                ts = pd.Timestamp(s.index[-1])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                out.append({"symbol": symbols[0], "close": float(s.iloc[-1]),
                            "date": ts})
    if not out:
        return pd.DataFrame(columns=["close", "date"])
    return pd.DataFrame(out).set_index("symbol")


def feed_health(last_bar_ts: pd.Timestamp | None,
                n_received: int, n_expected: int,
                sla_seconds: float = 86400 * 2,
                now: datetime | None = None) -> FeedHealth:
    """
    Assess whether the latest available bar is fresh enough to trade.

    `last_bar_ts` is the timestamp of the freshest bar across all
    symbols. Pass `None` if no data came back. `now` is injected for
    testability — defaults to `datetime.now(timezone.utc)`.

    `sla_seconds` defaults to 48h (covers Friday-to-Monday). Trading
    daily, you want this set to a couple of bars beyond the expected
    cadence — long enough to ride out a holiday but short enough to
    catch a dead feed before next session's rebalance.

    A stale feed must NOT silently flatten the book via alpha decay.
    The rebalance should hard-reject and alert. This function returns
    the health record; the runner decides what to do with it.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    if last_bar_ts is None:
        return FeedHealth(
            last_bar_ts=datetime.fromtimestamp(0, tz=timezone.utc),
            age_seconds=float("inf"),
            is_stale=True, sla_seconds=sla_seconds,
            n_symbols_received=n_received, n_symbols_expected=n_expected,
        )
    ts = pd.Timestamp(last_bar_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age = (current - ts.to_pydatetime()).total_seconds()
    return FeedHealth(
        last_bar_ts=ts.to_pydatetime(),
        age_seconds=age,
        is_stale=age > sla_seconds,
        sla_seconds=sla_seconds,
        n_symbols_received=n_received,
        n_symbols_expected=n_expected,
    )


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationReport:
    price_mismatches: list[dict]
    position_mismatches: list[dict]
    cash_drift: float           # internal_cash - broker_cash
    is_clean: bool


def reconcile_prices(internal: pd.Series, broker: pd.Series,
                     threshold_pct: float = 5.0) -> list[dict]:
    """
    Compare two price sources and return symbols whose prices diverge
    by more than `threshold_pct` percent.

    Use case: yfinance adjusted close (the source the signal was
    computed against) vs Alpaca last trade (the source we are about to
    transact at). Persistent >5% divergence usually means an
    unprocessed corporate action.
    """
    common = internal.index.intersection(broker.index)
    out = []
    for sym in common:
        a = float(internal[sym])
        b = float(broker[sym])
        if a <= 0 or b <= 0:
            continue
        pct = abs(a - b) / a * 100.0
        if pct > threshold_pct:
            out.append({"symbol": sym, "internal": a, "broker": b,
                        "pct_diff": pct})
    return out


def reconcile_positions(internal_qty: dict[str, float],
                        broker_qty: dict[str, float],
                        tolerance: float = 1e-6) -> list[dict]:
    """
    Compare per-symbol position quantities between the internal book and
    the broker. Any mismatch beyond `tolerance` shares is logged.

    Position drift is the single most common silent failure in a real
    system — fills missed by the local listener, corporate actions
    booked by the broker but not by us, manual overrides — and is far
    more costly than the price-divergence case because it persists
    across rebalances.
    """
    symbols = set(internal_qty) | set(broker_qty)
    out = []
    for s in symbols:
        i = float(internal_qty.get(s, 0.0))
        b = float(broker_qty.get(s, 0.0))
        if abs(i - b) > tolerance:
            out.append({"symbol": s, "internal": i, "broker": b,
                        "diff": i - b})
    return out


def reconcile(internal_account, broker_account,
              internal_prices: pd.Series, broker_prices: pd.Series,
              price_threshold_pct: float = 5.0,
              qty_tolerance: float = 1e-6) -> ReconciliationReport:
    """
    Full reconciliation: prices, positions, cash. Called at SOD and
    EOD by the runner.
    """
    price_mm = reconcile_prices(internal_prices, broker_prices,
                                price_threshold_pct)
    int_qty = {s: p.qty for s, p in internal_account.positions.items()}
    brk_qty = {s: p.qty for s, p in broker_account.positions.items()}
    pos_mm = reconcile_positions(int_qty, brk_qty, qty_tolerance)
    cash_drift = float(internal_account.cash - broker_account.cash)
    is_clean = (not price_mm and not pos_mm
                and abs(cash_drift) < 1.0)
    return ReconciliationReport(
        price_mismatches=price_mm,
        position_mismatches=pos_mm,
        cash_drift=cash_drift,
        is_clean=is_clean,
    )
