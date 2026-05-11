"""
Download S&P 500 prices and reconstruct point-in-time index membership.

The index has ~25 changes per year. Studying only today's members and
backfilling their full history introduces survivorship bias (delisted
or removed names are missing) and index-membership look-ahead (today's
constituents are treated as members in past periods when they were not).

To correct both, we parse the historical changes table from Wikipedia
and rebuild the membership set as it stood on each trading day. The
downloader fetches every ticker that was ever in the index over the
sample period and writes a membership mask alongside the price panel.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


DATA_DIR = Path("data")
START_DATE = "2005-01-01"
END_DATE = "2026-04-01"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_wiki_tables():
    response = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))


def _norm_ticker(t):
    if pd.isna(t):
        return None
    return str(t).strip().replace(".", "-")


def get_current_members():
    """Today's S&P 500 with the date each name was added to the index."""
    tables = _fetch_wiki_tables()
    df = tables[0].copy()
    df["Symbol"] = df["Symbol"].map(_norm_ticker)
    df["Date added"] = pd.to_datetime(df["Date added"], errors="coerce")
    return df[["Symbol", "Date added"]].rename(columns={"Symbol": "ticker",
                                                        "Date added": "added"})


def get_change_history():
    """Historical add/remove events from Wikipedia's changes table."""
    tables = _fetch_wiki_tables()
    changes = tables[1].copy()
    changes.columns = ["date", "added_ticker", "added_name",
                       "removed_ticker", "removed_name", "reason"]
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes["added_ticker"] = changes["added_ticker"].map(_norm_ticker)
    changes["removed_ticker"] = changes["removed_ticker"].map(_norm_ticker)
    changes["reason"] = (changes["reason"]
                         .astype(str)
                         .str.replace(r"\s*\[\d+\]", "", regex=True)
                         .str.strip())
    return changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def reconstruct_membership(trading_days, current=None, changes=None):
    """
    Build a (date x ticker) boolean membership panel.

    Starts from today's index and walks the change log backwards to
    recover membership on each event date, then forward-fills onto
    `trading_days`. Tickers that were members at some point but later
    removed appear as columns and are True only on dates they were
    actually in the index.
    """
    if current is None:
        current = get_current_members()
    if changes is None:
        changes = get_change_history()

    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    today = pd.Timestamp.today().normalize()

    # Walk changes in reverse to find membership at the start of the trading window.
    members = set(current["ticker"].dropna())
    snapshots = {today: set(members)}
    for _, row in changes[::-1].iterrows():
        added = row["added_ticker"]
        removed = row["removed_ticker"]
        # Before this change, the added ticker wasn't a member; the removed ticker was.
        if isinstance(added, str) and added:
            members.discard(added)
        if isinstance(removed, str) and removed:
            members.add(removed)
        snapshots[row["date"]] = set(members)

    all_tickers = sorted({t for s in snapshots.values() for t in s})
    snap_dates = sorted(snapshots.keys())
    snap_panel = pd.DataFrame(
        [[t in snapshots[d] for t in all_tickers] for d in snap_dates],
        index=pd.DatetimeIndex(snap_dates),
        columns=all_tickers,
    )

    mask = snap_panel.reindex(trading_days.union(snap_panel.index)).ffill().fillna(False)
    return mask.reindex(trading_days).astype(bool)


def download_prices(tickers, start, end):
    """Bulk yfinance download. Returns a (Open/High/Low/Close/Volume) panel."""
    data = yf.download(
        sorted(set(tickers)),
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    return data


def clean_prices(close, max_abs_return=1.0):
    """
    Mask price entries that would produce implausible daily returns.

    yfinance occasionally returns adjusted-close series with stale, zero,
    or spiked values for tickers around delistings and acquisitions
    (e.g., CBE 2012). These create per-day returns of thousands of
    percent that dominate the backtest. Any close producing a same-day
    return with `|r| > max_abs_return` is replaced with NaN so that
    downstream pct_change and signal computations skip those rows.
    """
    close = close.copy()
    rets = close.pct_change()
    bad = rets.abs() > max_abs_return
    if bad.any().any():
        close = close.mask(bad)
        # The previous close that produced the spike is also suspect.
        prev_bad = bad.shift(-1, fill_value=False)
        close = close.mask(prev_bad)
    return close


def main():
    DATA_DIR.mkdir(exist_ok=True)

    current = get_current_members()
    changes = get_change_history()
    print(f"Current members: {len(current)}; historical change events: {len(changes)}")

    # Trading-day index built from SPY so the membership panel aligns with the
    # price data we will load downstream.
    spy = yf.download("SPY", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    spy.to_parquet(DATA_DIR / "spy.parquet")
    print(f"SPY: {len(spy)} trading days, saved to {DATA_DIR / 'spy.parquet'}")

    members = reconstruct_membership(spy.index, current=current, changes=changes)
    members.to_parquet(DATA_DIR / "members.parquet")
    print(f"Membership panel: {members.shape[0]} dates x {members.shape[1]} tickers, "
          f"saved to {DATA_DIR / 'members.parquet'}")

    # Download every ticker that was a member at any point in the sample window.
    in_window = members.loc[START_DATE:END_DATE]
    universe = in_window.columns[in_window.any()].tolist()
    print(f"Downloading {len(universe)} tickers (union of all ever-members)")

    data = download_prices(universe, START_DATE, END_DATE)
    print(f"Downloaded data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    close = data["Close"]
    cleaned = clean_prices(close)
    flagged = close.notna().sum().sum() - cleaned.notna().sum().sum()
    print(f"Cleaned {flagged} suspect price entries (|daily return| > 100%)")
    data["Close"] = cleaned
    data.to_parquet(DATA_DIR / "sp500.parquet")
    print(f"Saved {len(close.columns)} tickers, {len(close)} trading days")
    print(f"Tickers with data from start: {close.iloc[0].notna().sum()}")
    print(f"Tickers with any data:        {close.notna().any().sum()}")
    print(f"File saved to {DATA_DIR / 'sp500.parquet'}")


if __name__ == "__main__":
    main()
