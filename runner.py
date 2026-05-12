"""
Daily rebalance runner. The cron entry point.

Pipeline (top of every trading day, at the configured rebalance time):

    1. Load configuration (config.yaml).
    2. Build today's target weights (combine_weights.latest_target_weights),
       optionally applying signal-side decay if the freshest bar is older
       than the rebalance cadence.
    3. Pull live prices for every name in the target plus held positions
       (yfinance), keeping the actual bar timestamps.
    4. Feed health check: if the latest bar is older than the SLA, alert
       and exit non-zero. Do NOT silently shrink the book via decay.
    5. Reconcile: positions and prices between internal book and broker.
       In paper mode there is nothing external to reconcile; the
       reconcile call still runs and the result is logged.
    6. Build the pre-trade risk gate from config + recent ADV + the
       *persisted* high-water-mark from the previous run.
    7. Call execution.rebalance with the chosen algo.
    8. Append every event to the audit JSONL.
    9. Update the persisted HWM with end-of-run equity.

This script is intentionally thin: every piece of logic lives in a
focused module (combine_weights, execution, risk, price_feed) and the
runner is just glue. That separation is what makes the system
inspectable in code review and replaceable piece-by-piece.

Usage:

    python runner.py                                # default config
    python runner.py --config config.yaml --dry-run # plan only, no submit
    python runner.py --asof 2026-03-31              # backfill one day
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from combine_weights import latest_target_weights
from execution import AuditLog, PaperBroker, rebalance
from price_feed import fetch_latest_close, feed_health, reconcile
from risk import PreTradeRiskGate, RiskLimits, adv_dollar_from_prices


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _state_path(log_dir: Path) -> Path:
    return log_dir / "state.json"


def _load_state(log_dir: Path) -> dict:
    """
    Persisted runtime state. Currently just the drawdown high-water
    mark; deliberately a tiny JSON so the file can be inspected and
    edited by hand if a run needs to be reset.
    """
    p = _state_path(log_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _save_state(log_dir: Path, state: dict) -> None:
    p = _state_path(log_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _build_broker(cfg: dict) -> PaperBroker:
    """The runner only constructs paper brokers. To go live, swap to
    `AlpacaBroker` here and ensure the audit + reconciliation steps
    actually compare against the live account."""
    bcfg = cfg["broker"]
    return PaperBroker(
        starting_cash=bcfg["starting_cash"],
        slippage_bps=bcfg["slippage_bps"],
        commission_bps=bcfg["commission_bps"],
    )


def _build_risk_gate(cfg: dict, hwm: float,
                     adv_dollar: dict[str, float]) -> PreTradeRiskGate:
    rcfg = cfg["risk"]
    limits = RiskLimits(
        max_gross_leverage=rcfg["max_gross_leverage"],
        max_net_exposure=rcfg["max_net_exposure"],
        max_position_pct=rcfg["max_position_pct"],
        max_positions=rcfg["max_positions"],
        max_adv_participation_pct=rcfg.get("max_adv_participation_pct"),
        max_daily_turnover_pct=rcfg["max_daily_turnover_pct"],
        drawdown_kill_pct=rcfg["drawdown_kill_pct"],
        blocked_symbols=set(rcfg.get("blocked_symbols", [])),
    )
    return PreTradeRiskGate(limits, high_water_mark=hwm, adv_dollar=adv_dollar)


def run(config_path: Path, asof: str | None = None,
        dry_run: bool = False) -> int:
    cfg = _load_config(config_path)
    load_dotenv(REPO_ROOT / ".env")

    log_dir = REPO_ROOT / cfg.get("paths", {}).get("log_dir", "logs")
    today = (asof or datetime.now(timezone.utc).date().isoformat())
    audit = AuditLog(log_dir / today / "journal.jsonl")
    audit.write("runner_start", {"asof": today, "dry_run": dry_run})

    state = _load_state(log_dir)
    persisted_hwm = state.get("high_water_mark")

    broker = _build_broker(cfg)
    held = set(broker.get_account().positions.keys())

    # 1. Probe the feed first with a single liquid name (SPY) so we
    # know the bar age before deciding whether to apply signal-side
    # decay to the rebuild. Cheap one-symbol fetch.
    probe = fetch_latest_close(["SPY"])
    if probe.empty:
        audit.write("aborted_no_probe_price", {})
        logging.error("probe fetch (SPY) returned no data")
        return 2
    freshest_bar = probe["date"].max()

    health = feed_health(
        last_bar_ts=freshest_bar,
        n_received=len(probe), n_expected=1,
        sla_seconds=cfg["execution"]["feed_sla_seconds"],
    )
    audit.write("feed_health", {
        "last_bar_ts": str(health.last_bar_ts),
        "age_seconds": health.age_seconds,
        "is_stale": health.is_stale,
        "sla_seconds": health.sla_seconds,
    })
    if health.is_stale:
        logging.error("feed stale: bar=%s age=%.0fs sla=%.0fs",
                      health.last_bar_ts, health.age_seconds,
                      health.sla_seconds)
        audit.write("aborted_stale_feed", {})
        return 2

    # 2. Signal-side decay (optional). `age_bars` is business days
    # between the freshest available bar and `asof`. With a 48h SLA
    # this is usually 0 (today) or 1 (Monday after a normal Friday);
    # holidays push it higher. Decay is applied to the signal inside
    # `build_combined_weights` BEFORE the optimiser, so the leverage
    # and dollar-neutrality constraints remain valid.
    asof_date = pd.Timestamp(asof or datetime.now(timezone.utc).date().isoformat())
    bar_date = pd.Timestamp(health.last_bar_ts.date())
    age_bars = max(0, len(pd.bdate_range(bar_date, asof_date)) - 1)
    half_life = cfg["execution"].get("signal_half_life_bars")

    logging.info("building target weights (age_bars=%d, half_life=%s)",
                 age_bars, half_life)
    weights = latest_target_weights(
        asof=asof, age_bars=age_bars if half_life else 0,
        half_life_bars=half_life,
    )
    audit.write("target_weights_built", {
        "asof": today, "age_bars": age_bars, "half_life": half_life,
        "n_long": int((weights > 0).sum()),
        "n_short": int((weights < 0).sum()),
        "gross": float(weights.abs().sum()),
        "net": float(weights.sum()),
    })

    # 3. Now fetch the full price set for held positions + target names.
    symbols = sorted(set(weights.index[weights != 0]) | held)
    logging.info("fetching live prices for %d symbols", len(symbols))
    latest = fetch_latest_close(symbols)
    if latest.empty:
        audit.write("aborted_no_prices", {})
        logging.error("no prices returned by feed for trading universe")
        return 2

    prices = latest["close"].to_dict()
    broker.update_prices(prices)

    # 5. ADV for the risk gate. In a real system this comes from the
    # broker's historical bars endpoint; here we approximate with the
    # last 20 sessions of the backtest data. Read `Close` (full
    # 832-ticker universe, auto-adjusted) not `Adj Close` (sparse
    # 175-ticker legacy column that is all-NaN in this parquet — would
    # silently zero out every ADV and turn the participation gate into
    # a no-op).
    sp500 = pd.read_parquet(REPO_ROOT / "data" / "sp500.parquet")
    close = sp500["Close"]
    volume = sp500["Volume"]
    adv_dollar = adv_dollar_from_prices(close, volume, lookback=20).to_dict()

    equity = broker.get_account().equity(prices, strict=False)
    # HWM: take the larger of persisted and current equity. First-run
    # gets seeded with `starting_cash`. Without persistence the
    # drawdown kill-switch would reset every cron firing.
    hwm = max(persisted_hwm or cfg["broker"]["starting_cash"], equity)
    gate = _build_risk_gate(cfg, hwm=hwm, adv_dollar=adv_dollar)
    audit.write("risk_gate_built", {"hwm": hwm, "equity": equity})

    # 6. Reconciliation. In paper mode the broker *is* our book so
    # there is nothing external to compare against; the call still
    # runs (with internal=broker) so the audit log has a clean record
    # of "reconciliation ran, no mismatches" rather than ambiguous
    # silence. When a real broker is wired in, pass its account here.
    recon = reconcile(
        internal_account=broker.get_account(),
        broker_account=broker.get_account(),
        internal_prices=latest["close"],
        broker_prices=latest["close"],
        price_threshold_pct=cfg["reconciliation"]["price_divergence_threshold_pct"],
        qty_tolerance=cfg["reconciliation"]["qty_tolerance"],
    )
    audit.write("reconciliation", {
        "is_clean": recon.is_clean,
        "n_price_mismatches": len(recon.price_mismatches),
        "n_position_mismatches": len(recon.position_mismatches),
        "cash_drift": recon.cash_drift,
    })
    if not recon.is_clean:
        logging.error("reconciliation failed; aborting rebalance")
        audit.write("aborted_reconciliation", {
            "price_mismatches": recon.price_mismatches,
            "position_mismatches": recon.position_mismatches,
        })
        return 3

    # 7. Execute.
    ecfg = cfg["execution"]
    algo = ecfg["algo"]
    if dry_run:
        from execution import build_order_list, weights_to_target_shares
        targets = weights_to_target_shares(weights, prices, equity=equity)
        orders = build_order_list(targets, broker.get_account(), prices,
                                  strict_prices=False)
        breach = gate.check(orders, broker.get_account(), prices, equity)
        audit.write("dry_run_summary", {
            "n_orders": len(orders), "equity": equity, "risk_breach": breach,
        })
        print(f"DRY RUN: {len(orders)} orders, equity={equity:.2f}, "
              f"breach={breach}")
        return 0

    result = rebalance(
        broker, weights, prices,
        audit=audit, risk_check=gate.check,
        algo=algo,
        twap_slices_n=ecfg.get("twap_slices"),
        vwap_profile_arr=None,
        allow_fractional=ecfg.get("allow_fractional", False),
        min_order_notional=ecfg.get("min_order_notional", 1.0),
    )

    # 8. Persist HWM. End-of-run equity is the candidate; the gate's
    # update_hwm only mutates upward so this is monotone.
    end_equity = broker.get_account().equity(prices, strict=False)
    gate.update_hwm(end_equity)
    state["high_water_mark"] = float(max(hwm, end_equity))
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    _save_state(log_dir, state)

    audit.write("runner_end", {
        "status": result["status"],
        "n_orders": result.get("n_orders", 0),
        "end_equity": end_equity,
        "hwm": state["high_water_mark"],
    })
    print(f"{result['status']}: orders={result.get('n_orders', 0)}, "
          f"equity={end_equity:.2f}, hwm={state['high_water_mark']:.2f}")
    return 0 if result["status"] == "executed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asof", type=str, default=None,
                        help="run as if this is the date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only, do not submit orders")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    sys.exit(run(args.config, asof=args.asof, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
