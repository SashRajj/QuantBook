import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


DATA_DIR = Path("data")
START_DATE = "2005-01-01"
END_DATE = "2026-04-01"


def get_sp500_tickers():
    """Fetch current S&P 500 tickers and normalize symbols for yfinance."""
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text))
    return table[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def main():
    DATA_DIR.mkdir(exist_ok=True)

    sp500 = get_sp500_tickers()
    print(f"Found {len(sp500)} S&P 500 tickers")

    data = yf.download(
        sp500,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        threads=True,
    )
    print(f"Downloaded data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    data.to_parquet(DATA_DIR / "sp500.parquet")

    close = data["Close"]
    print(f"Saved {len(close.columns)} tickers, {len(close)} trading days")
    print(f"Tickers with data from 2005: {(close.iloc[0].notna()).sum()}")
    print(f"Tickers with any data: {(close.notna().any()).sum()}")
    print(f"File saved to {DATA_DIR / 'sp500.parquet'}")

    spy = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True)
    spy.to_parquet(DATA_DIR / "spy.parquet")
    print(f"SPY: {len(spy)} trading days, saved to {DATA_DIR / 'spy.parquet'}")


if __name__ == "__main__":
    main()
