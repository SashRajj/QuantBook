"""
Research-to-execution bridge: build the combined-portfolio target weights.

The notebooks save per-strategy weights (`weights_01_*`, `weights_02_*`,
`weights_03_*`) but the combined IC-weighted portfolio from notebook 04
is computed in memory and never written to disk. The live execution layer
needs a single target-weights panel to act on, so this module reproduces
the notebook-04 combination outside the notebook and saves the result to
`data/weights_combined_oos.parquet`.

Usage:

    python combine_weights.py                       # rebuild full panel
    python combine_weights.py --asof 2026-03-31     # rebuild and report row

Or import from the daily runner:

    from combine_weights import latest_target_weights
    w = latest_target_weights()                     # most recent row
    w = latest_target_weights(asof="2026-03-31")    # specific date

Construction matches notebook 04 verbatim (5 bps cost penalty, 2% per-name
cap, dollar-neutral, point-in-time membership mask). Any deviation from
notebook 04 here is a backtest-vs-live mismatch and should be treated as
a bug, not a feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from helper import Optimizer, ic_weighted_combine
from signals import low_volatility, mean_reversion, momentum


# Match notebook 04 exactly. Changing any of these silently breaks the
# backtest-vs-live equivalence the executor relies on.
PARAMS = {
    "ic_lookback": 126,
    "ic_horizon": 1,
    "ic_min": 0.0,
    "mr_lookback": 20,
    "mom_lookback": 252,
    "mom_skip": 21,
    "lv_lookback": 63,
    "opt_max_position": 0.02,
    "opt_max_leverage": 1.0,
    "opt_tcost_penalty_bps": 5,
    "opt_subsample": 1,
    "opt_cov_lookback_days": 252,    # 1y trailing window for cov
    "opt_cov_refit_every": 21,       # monthly cov refit
}

DATA_DIR = Path(__file__).resolve().parent / "data"
SP500_PATH = DATA_DIR / "sp500.parquet"
MEMBERS_PATH = DATA_DIR / "members.parquet"
OUTPUT_PATH = DATA_DIR / "weights_combined_oos.parquet"


def apply_signal_decay(signal: pd.DataFrame, age_bars: int,
                       half_life_bars: float) -> pd.DataFrame:
    """
    Exponentially decay a *signal* panel by its age in trading bars.

    Multiplier is `0.5 ** (age_bars / half_life_bars)`. The decay is
    applied to the signal **before** the optimiser, not to the
    optimiser's weights. Scaling weights post-optimisation would
    violate the dollar-neutrality and leverage constraints the
    optimiser solved under; scaling the signal lets the optimiser
    rebalance the rest of the book to the smaller alpha view in a
    constraint-consistent way.

    `age_bars` should come from a real "last successful refresh"
    timestamp, not from a counter. If the timestamp is missing because
    the feed is dead, the caller must raise — silent zero is the bug
    `price_feed.feed_health` is designed to prevent.
    """
    if half_life_bars <= 0 or age_bars <= 0:
        return signal
    factor = 0.5 ** (age_bars / half_life_bars)
    return signal * factor


def _load_price_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return `(close, returns, members)` aligned on dates and columns.

    `Close` is the right column to read: `download_sp500.py` calls
    `yf.download(auto_adjust=True)`, which folds split- and
    dividend-adjustments into `Close` and only populates a separate
    `Adj Close` field for a handful of legacy/delisted names (175
    tickers vs the full 832-ticker universe). Reading `Adj Close`
    would silently restrict the universe and produce an empty
    combined signal once intersected with `members.parquet`.
    """
    raw = pd.read_parquet(SP500_PATH)
    close = raw["Close"].sort_index()
    members = pd.read_parquet(MEMBERS_PATH).sort_index()

    cols = sorted(set(close.columns) & set(members.columns))
    if not cols:
        raise RuntimeError(
            "no overlap between sp500.parquet Close columns and members.parquet "
            "tickers; re-run download_sp500.py to regenerate both"
        )
    close = close[cols]
    members = members.reindex(index=close.index, columns=cols).fillna(False)

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return close, returns, members.astype(bool)


def _compute_signals(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recompute the three linear factor signals from prices.

    Recomputing rather than loading the saved parquets keeps live signals
    bit-identical to the backtest formula even if the saved files become
    stale.
    """
    return {
        "mean_reversion": mean_reversion(close, lookback=PARAMS["mr_lookback"]),
        "momentum": momentum(close, lookback=PARAMS["mom_lookback"],
                             skip=PARAMS["mom_skip"]),
        "low_volatility": low_volatility(close, lookback=PARAMS["lv_lookback"]),
    }


def build_combined_weights(asof: pd.Timestamp | None = None,
                           age_bars: int = 0,
                           half_life_bars: float | None = None,
                           warmup_days: int | None = None,
                           latest_only: bool = False,
                           verbose: bool = False) -> pd.DataFrame:
    """
    Build the combined-weights panel ending at `asof`.

    `asof` truncates the input panel to `<= asof` before signal
    computation, which prevents future data leaking into the QP at
    the live row. The default is the latest available date.

    `warmup_days` truncates the input panel to the trailing
    `warmup_days` *before* `asof`. Use for live runs that only need
    the most recent row — solving the full ~5000-date panel takes
    hours on the S&P 500 universe and is unnecessary in production.
    Pass `None` (default) to keep full history; pass `~1000+` for live.

    `latest_only` further restricts the optimiser to solving only the
    last valid date in the combined-signal panel. The cov still uses
    the trailing window. With `latest_only=True`, run time drops from
    minutes to seconds on the S&P 500 universe. Use it from the daily
    runner; leave `False` only for backtest replay.

    `age_bars` + `half_life_bars` apply exponential decay to the
    *combined signal* before the optimiser runs. Decaying weights
    post-optimisation would violate the dollar-neutrality and
    leverage constraints the QP solved under; scaling the signal
    lets the optimiser rebalance the rest of the book consistently.
    Pass `age_bars=0` (default) for no decay.

    Returns a DataFrame (date x ticker) of target weights. Rows before
    the signal warm-up window are NaN.
    """
    close, returns, members = _load_price_panel()
    if asof is not None:
        asof = pd.Timestamp(asof)
        close = close.loc[:asof]
        returns = returns.loc[:asof]
        members = members.loc[:asof]
    if warmup_days is not None:
        # Minimum warmup the constituent signals and the optimiser
        # actually need: longest signal lookback + IC-weighting window
        # + cov estimation window. Asserting here turns a silent
        # all-NaN output into a clear error if someone later adds a
        # longer-lookback signal without raising `warmup_days`.
        min_required = (PARAMS["mom_lookback"] + PARAMS["mom_skip"]
                        + PARAMS["ic_lookback"]
                        + PARAMS["opt_cov_lookback_days"])
        if warmup_days < min_required:
            raise ValueError(
                f"warmup_days={warmup_days} is below the minimum "
                f"{min_required} required by mom_lookback "
                f"({PARAMS['mom_lookback']}) + mom_skip "
                f"({PARAMS['mom_skip']}) + ic_lookback "
                f"({PARAMS['ic_lookback']}) + opt_cov_lookback_days "
                f"({PARAMS['opt_cov_lookback_days']})"
            )
        if len(close) > warmup_days:
            close = close.iloc[-warmup_days:]
            returns = returns.iloc[-warmup_days:]
            members = members.iloc[-warmup_days:]

    signals = _compute_signals(close)

    combined, _ic_weights = ic_weighted_combine(
        signals, close,
        lookback=PARAMS["ic_lookback"],
        min_ic=PARAMS["ic_min"],
        horizon=PARAMS["ic_horizon"],
    )

    if age_bars > 0 and half_life_bars is not None and half_life_bars > 0:
        combined = apply_signal_decay(combined, age_bars=age_bars,
                                      half_life_bars=half_life_bars)

    # Hand the full returns history to the optimiser — it does its own
    # trailing-window cov refit per date, point-in-time.
    members_for_opt = members.reindex(index=returns.index,
                                      columns=returns.columns).fillna(False)

    combined_for_opt = combined.dropna(how="all")
    if latest_only and not combined_for_opt.empty:
        # Keep only the latest valid signal row. The cov inside the
        # Optimizer still uses the trailing-window of `returns`, which
        # we hand in unmodified.
        combined_for_opt = combined_for_opt.iloc[[-1]]

    opt = Optimizer(
        combined_for_opt, returns,
        cov_lookback_days=PARAMS["opt_cov_lookback_days"],
        cov_refit_every=PARAMS["opt_cov_refit_every"],
    )
    weights = opt.run(
        dollar_neutral=True,
        fully_invested=False,
        max_position=PARAMS["opt_max_position"],
        max_leverage=PARAMS["opt_max_leverage"],
        tcost_penalty_bps=PARAMS["opt_tcost_penalty_bps"],
        subsample=PARAMS["opt_subsample"],
        member_mask=members_for_opt,
        verbose=verbose,
    )

    full_cols = sorted(set(close.columns))
    weights = weights.reindex(columns=full_cols).fillna(0.0)
    return weights


def latest_target_weights(asof: str | pd.Timestamp | None = None,
                          age_bars: int = 0,
                          half_life_bars: float | None = None,
                          max_stale_business_days: int = 5) -> pd.Series:
    """
    Return the single target-weights row for `asof` (or the latest date).

    Raises if the freshest non-zero weights row is more than
    `max_stale_business_days` business days older than `asof` (or
    today, if `asof` is None). Silently returning a stale row is how
    "trade last week's book at today's prices" bugs land in production;
    this guard is the cheapest place to catch them.
    """
    asof_ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp.utcnow().normalize()
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_localize(None)

    # Live use: cap the panel to roughly twice the minimum warmup so
    # the cov has enough history, and solve only the last valid row.
    # End-to-end this is seconds, not minutes.
    panel = build_combined_weights(asof=asof_ts, age_bars=age_bars,
                                   half_life_bars=half_life_bars,
                                   warmup_days=1200,
                                   latest_only=True)
    if panel.empty:
        raise RuntimeError("combined weights panel is empty; check inputs")
    non_zero = panel.loc[(panel != 0).any(axis=1)]
    if non_zero.empty:
        raise RuntimeError("no non-zero weight row found in panel")

    latest_ts = pd.Timestamp(non_zero.index[-1])
    if latest_ts.tzinfo is not None:
        latest_ts = latest_ts.tz_localize(None)
    age = len(pd.bdate_range(latest_ts, asof_ts)) - 1
    if age > max_stale_business_days:
        raise RuntimeError(
            f"latest weight row ({latest_ts.date()}) is {age} business "
            f"days older than asof ({asof_ts.date()}); refusing to "
            f"return a stale book. Re-run combine_weights or investigate."
        )
    return non_zero.iloc[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", type=str, default=None,
                        help="latest date to include (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH,
                        help="output parquet path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    panel = build_combined_weights(asof=args.asof, verbose=args.verbose)
    panel.to_parquet(args.out)

    last = panel.loc[(panel != 0).any(axis=1)].iloc[-1]
    print(f"Wrote {args.out}: shape={panel.shape}")
    print(f"Last non-zero date: {last.name}")
    print(f"  Gross exposure: {last.abs().sum():.4f}")
    print(f"  Net exposure:   {last.sum():+.4f}")
    print(f"  Long names:     {(last > 0).sum()}")
    print(f"  Short names:    {(last < 0).sum()}")
    print(f"  Max long:       {last.max():+.4f}")
    print(f"  Max short:      {last.min():+.4f}")


if __name__ == "__main__":
    main()
