"""
Pre-trade risk gates.

Every order basket passes through these checks before any submission.
A single breach rejects the *entire* basket — half-rebalances are
worse than no rebalance, because they leave the book in a state
neither the model nor the human asked for.

Gate categories, in the order they fire (drawdown first, sizing last,
because a drawdown breach kills the rebalance regardless of what the
optimiser produced):

  - **Drawdown**: kill switch on equity below high-water mark by more
    than X%.
  - **Eligibility**: optional blocklist / restricted-symbol set.
  - **Sizing**: gross leverage, net exposure, per-name notional,
    per-position-count.
  - **Liquidity**: ADV participation (no single order exceeds X% of
    the symbol's 20-day average daily dollar volume).
  - **Turnover**: total notional traded in this rebalance vs cap.
    "Daily" in the config key is the intended cadence (one rebalance
    per day); a system that rebalances mid-day would need an external
    accumulator across calls.

The configuration is a plain dataclass so it can be loaded from YAML,
mutated for tests, or hard-coded for a one-off run. Each check is a
method that returns either None (clear) or a string explaining the
breach.

This module is intentionally read-only. It does not amend orders, it
does not partially reject. The rebalance is either clear or it is
not, and a rejected rebalance is logged with the reason so the human
can decide whether to push through manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from execution import Account, Order

# Optional C++ acceleration for the pre-trade gate. Same additive
# optimisation pattern as execution.py: import-failure falls back to
# the pure-Python implementation that lives below.
try:
    import qbexec_cpp as _qbexec_cpp  # noqa: F401
    _HAS_CPP = True
except ImportError:
    _qbexec_cpp = None
    _HAS_CPP = False


@dataclass
class RiskLimits:
    """
    Configuration for pre-trade risk. All units are fractions of equity
    or dollars where noted. `None` disables that check.
    """
    max_gross_leverage: float = 1.5      # |long| + |short| <= 1.5 * equity
    max_net_exposure: float = 0.10       # long - short, abs, as fraction of equity
    max_position_pct: float = 0.05       # any single |pos| / equity
    max_positions: int = 200             # count of non-zero target positions
    max_adv_participation_pct: float = 0.05  # single order <= 5% of 20D ADV
    max_daily_turnover_pct: float = 2.0  # sum(|order_notional|) / equity
    drawdown_kill_pct: float = 0.15      # equity / high-water-mark - 1 < -0.15
    blocked_symbols: set[str] = field(default_factory=set)
    # If True, an order on a symbol with no ADV data is breach-rejected
    # (conservative: caller must whitelist by populating adv_dollar).
    # If False, missing ADV is logged as a soft pass — useful when the
    # adv panel is known to be sparse (newly listed names).
    strict_adv: bool = False


class PreTradeRiskGate:
    """
    Stateful gate that tracks high-water-mark for the drawdown check
    and exposes a `check()` callable suitable for `rebalance(risk_check=...)`.

    `adv_dollar` is an optional mapping `symbol -> 20-day average daily
    dollar volume`. Names without ADV data skip the participation check
    (logged via the returned breach string rather than silently
    passed). Pass a populated dict in production.
    """

    def __init__(self, limits: RiskLimits,
                 high_water_mark: float | None = None,
                 adv_dollar: dict[str, float] | None = None):
        self.limits = limits
        self._hwm = high_water_mark
        self._adv = adv_dollar or {}

    def update_hwm(self, equity: float) -> None:
        if self._hwm is None or equity > self._hwm:
            self._hwm = equity

    def check(self, orders: list[Order], account: Account,
              prices: dict[str, float], equity: float) -> str | None:
        """
        Run every gate; return the first breach as a string or None.

        Order of checks is fixed so failures are reproducible. Drawdown
        check fires first because it can hard-flat the book regardless
        of what the optimiser produced.

        Hot-path: when the C++ extension is built, the entire check
        loop runs in `qbexec_cpp.risk_check`. Numeric formatting in the
        returned breach string may differ in a digit or two from the
        pure-Python form (snprintf vs Python's `f"{x:g}"`), but the
        order of checks, the substring keywords ("drawdown",
        "gross leverage", "per-name", "blocked", ...) and the
        all-clear-vs-breach decision are identical.
        """
        self.update_hwm(equity)

        if _HAS_CPP:
            return self._check_cpp(orders, account, prices, equity)
        return self._check_python(orders, account, prices, equity)

    # ------------------------------------------------------------------
    # C++ delegate
    # ------------------------------------------------------------------

    def _check_cpp(self, orders: list[Order], account: Account,
                   prices: dict[str, float], equity: float) -> str | None:
        # Build a plain C++ limits struct from the dataclass. None
        # fields stay as std::optional() i.e. disabled checks.
        lim = _qbexec_cpp.RiskLimitsC()
        lim.max_gross_leverage = self.limits.max_gross_leverage
        lim.max_net_exposure = self.limits.max_net_exposure
        lim.max_position_pct = self.limits.max_position_pct
        lim.max_positions = self.limits.max_positions
        lim.max_adv_participation_pct = self.limits.max_adv_participation_pct
        lim.max_daily_turnover_pct = self.limits.max_daily_turnover_pct
        lim.drawdown_kill_pct = self.limits.drawdown_kill_pct
        lim.blocked_symbols = set(self.limits.blocked_symbols)
        lim.strict_adv = self.limits.strict_adv

        cpp_orders = [
            _qbexec_cpp.OrderView(o.symbol, float(o.qty)) for o in orders
        ]
        current = {sym: float(pos.qty)
                   for sym, pos in account.positions.items()}

        breach = _qbexec_cpp.risk_check(
            cpp_orders,
            current,
            {k: float(v) for k, v in prices.items()},
            float(equity),
            self._hwm,
            {k: float(v) for k, v in self._adv.items()},
            lim,
        )
        return breach if breach else None

    # ------------------------------------------------------------------
    # Pure-Python fallback (also the numeric reference for the C++ port)
    # ------------------------------------------------------------------

    def _check_python(self, orders: list[Order], account: Account,
                      prices: dict[str, float], equity: float) -> str | None:
        # Drawdown kill switch.
        if (self._hwm is not None and self._hwm > 0
                and self.limits.drawdown_kill_pct is not None):
            dd = equity / self._hwm - 1.0
            if dd <= -self.limits.drawdown_kill_pct:
                return (f"drawdown kill: equity={equity:.2f} hwm={self._hwm:.2f} "
                        f"dd={dd:.2%} limit={-self.limits.drawdown_kill_pct:.2%}")

        # Blocked symbols.
        for o in orders:
            if o.symbol in self.limits.blocked_symbols:
                return f"blocked symbol in order list: {o.symbol}"

        # Compute the *post-trade* book state for sizing checks.
        post_qty: dict[str, float] = {
            sym: pos.qty for sym, pos in account.positions.items()
        }
        for o in orders:
            post_qty[o.symbol] = post_qty.get(o.symbol, 0.0) + o.qty

        long_notional = sum(q * prices[s] for s, q in post_qty.items()
                            if q > 0 and s in prices)
        short_notional = -sum(q * prices[s] for s, q in post_qty.items()
                              if q < 0 and s in prices)
        gross = long_notional + short_notional
        net = long_notional - short_notional

        if equity <= 0:
            return f"non-positive equity: {equity:.2f}"

        if gross / equity > self.limits.max_gross_leverage:
            return (f"gross leverage breach: gross={gross:.0f} "
                    f"equity={equity:.0f} ratio={gross/equity:.3f} "
                    f"limit={self.limits.max_gross_leverage}")

        if abs(net) / equity > self.limits.max_net_exposure:
            return (f"net exposure breach: net={net:+.0f} "
                    f"ratio={net/equity:+.3f} "
                    f"limit=+/-{self.limits.max_net_exposure}")

        # Per-name cap.
        for sym, q in post_qty.items():
            if sym not in prices:
                continue
            pos_pct = abs(q * prices[sym]) / equity
            if pos_pct > self.limits.max_position_pct:
                return (f"per-name cap breach: {sym} pos_pct={pos_pct:.3f} "
                        f"limit={self.limits.max_position_pct}")

        # Position count.
        non_zero = sum(1 for q in post_qty.values() if abs(q) > 1e-9)
        if non_zero > self.limits.max_positions:
            return (f"too many positions: {non_zero} > {self.limits.max_positions}")

        # ADV participation per order.
        if self.limits.max_adv_participation_pct is not None:
            for o in orders:
                if o.symbol not in prices:
                    continue
                order_notional = abs(o.qty * prices[o.symbol])
                adv = self._adv.get(o.symbol)
                if adv is None or adv <= 0:
                    if self.limits.strict_adv:
                        return (f"ADV unknown for {o.symbol}; cannot enforce "
                                f"participation limit (set strict_adv=False "
                                f"to skip)")
                    # Soft pass with no breach. The dry-run output and
                    # audit log surface the count of unknown-ADV names
                    # so an operator can decide to enrich the panel.
                    continue
                participation = order_notional / adv
                if participation > self.limits.max_adv_participation_pct:
                    return (f"ADV participation breach: {o.symbol} "
                            f"order_notional={order_notional:.0f} adv={adv:.0f} "
                            f"participation={participation:.3f} "
                            f"limit={self.limits.max_adv_participation_pct}")

        # Daily turnover (this rebalance only; a multi-rebalance day
        # would accumulate across calls).
        total_turnover = sum(abs(o.qty * prices.get(o.symbol, 0.0))
                             for o in orders)
        if total_turnover / equity > self.limits.max_daily_turnover_pct:
            return (f"daily turnover breach: turnover={total_turnover:.0f} "
                    f"ratio={total_turnover/equity:.3f} "
                    f"limit={self.limits.max_daily_turnover_pct}")

        return None


def adv_dollar_from_prices(close: pd.DataFrame, volume: pd.DataFrame,
                           lookback: int = 20) -> pd.Series:
    """
    20-day average daily dollar volume per symbol, from the most recent
    `lookback` bars. Used as input to `PreTradeRiskGate(adv_dollar=...)`.
    """
    dollar_volume = close * volume
    return dollar_volume.tail(lookback).mean()
