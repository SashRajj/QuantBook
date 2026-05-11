"""
Download daily crypto OHLCV from Binance.US public klines API.

No API key is required for klines. Symbols use USDT as the quote currency
(e.g. BTCUSDT) which is the most liquid pair format on Binance. The
output panel has the same multi-index column structure as sp500.parquet
so the existing signal and optimizer code applies without modification.
"""

import time
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path("data")
BASE_URL = "https://api.binance.us/api/v3/klines"
START_DATE = "2019-09-01"
END_DATE = "2026-04-01"

# Curated universe of top-by-market-cap pairs listed on Binance.US.
# Symbols are taken without the USD/USDT suffix; the suffix is appended
# at request time.
UNIVERSE = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "DOGE", "TRX", "AVAX", "LINK",
    "DOT", "MATIC", "LTC", "BCH", "NEAR",
    "ATOM", "AAVE", "ALGO", "FIL", "XLM",
    "ETC", "VET", "MKR", "EOS", "XTZ",
    "SAND", "MANA", "GRT", "ICP", "AXS",
]


def fetch_klines(symbol, start_ms, end_ms, interval="1d", session=None):
    """Paginated kline fetch. Returns a DataFrame indexed by close-of-day UTC."""
    session = session or requests.Session()
    bars = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "limit": 1000,
                  "startTime": cursor, "endTime": end_ms}
        r = session.get(BASE_URL, params=params, timeout=15)
        if r.status_code != 200:
            return None
        chunk = r.json()
        if not chunk:
            break
        bars.extend(chunk)
        last_close = chunk[-1][6]
        if last_close <= cursor:
            break
        cursor = last_close + 1
        # Light rate-limit hygiene; Binance.US allows 1200 req/min but
        # being a good citizen costs us nothing.
        time.sleep(0.05)

    if not bars:
        return None

    df = pd.DataFrame(bars, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "_ignore",
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df = df.drop_duplicates(subset="date").set_index("date")
    cols = ["open", "high", "low", "close", "volume"]
    return df[cols].astype(float)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    session = requests.Session()
    start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(END_DATE, tz="UTC").timestamp() * 1000)

    frames = {}
    for ticker in UNIVERSE:
        symbol = f"{ticker}USDT"
        df = fetch_klines(symbol, start_ms, end_ms, session=session)
        if df is None or df.empty:
            print(f"  {symbol:10s}  SKIP (no data)")
            continue
        frames[ticker] = df
        print(f"  {symbol:10s}  {df.index.min().date()} -> {df.index.max().date()}  "
              f"({len(df)} bars)")

    if not frames:
        raise RuntimeError("No crypto data downloaded; check network and Binance.US availability")

    panel = pd.concat(frames, axis=1)
    # Reorder to (field, ticker) like yfinance multi-index downloads.
    panel.columns = panel.columns.swaplevel(0, 1)
    panel = panel.sort_index(axis=1)
    panel.columns.names = ["Field", "Ticker"]
    panel = panel.rename(columns=str.capitalize, level=0)

    out = DATA_DIR / "crypto.parquet"
    panel.to_parquet(out)
    print(f"\nSaved {panel.shape[1] // 5} tickers x {len(panel)} days to {out}")


if __name__ == "__main__":
    main()
