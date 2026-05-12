"""
Execution layer: take target weights from research and turn them into orders.

The research side (signals.py, helper.py, combine_weights.py) produces a
weights vector (per ticker) for each rebalance. This module is the bridge
from that vector to a live (or paper) book:

    weights -> target shares -> order list -> execution algo -> broker

Three concerns live here that do not belong on the research side:

  1. Portfolio state. Real positions, cash, and equity. The optimiser
     thinks in fractional weights; the broker thinks in shares.
  2. Execution. Even if the target is right, how you trade into it
     matters: market orders move the price, scheduled orders (TWAP,
     VWAP) smooth that impact at the cost of timing risk.
  3. Order management. Every order needs an idempotent client id, a
     status that moves through a defined state machine, and an audit
     record that survives process restarts.

The Broker interface is deliberately small so a `PaperBroker` (this file)
and a live broker (`AlpacaBroker`, IB) are drop-in interchangeable.

Alpha decay is *not* applied here. Decaying optimiser weights violates
the dollar-neutrality and leverage constraints the optimiser solved
under; decay belongs on the *signal* before the optimiser runs. See
`combine_weights.apply_signal_decay`.
"""

from __future__ import annotations

import abc
import enum
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Enums and primitives
# ---------------------------------------------------------------------------

class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, enum.Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


def utcnow() -> datetime:
    """Timezone-aware UTC now. Audit timestamps must carry a tz."""
    return datetime.now(timezone.utc)


def new_client_id(prefix: str = "QR") -> str:
    """Idempotent client-order id. Retries with the same id are no-ops."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float

    def market_value(self, last_price: float) -> float:
        return self.qty * last_price


@dataclass
class Account:
    """
    In-memory view of cash, open positions, and realised PnL.

    The paper broker mutates this directly; a live-broker adapter would
    refresh it from the broker's account endpoint at the top of each
    rebalance and reconcile any drift.
    """
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def equity(self, prices: dict[str, float], strict: bool = True) -> float:
        """
        Marked-to-market equity. Cash + position value.

        With `strict=True` (default), missing prices for held positions
        raise. Silently dropping unpriced positions understates equity
        and under-sizes the rest of the book — exactly the failure mode
        that turns a stale feed into a portfolio-level bug. The caller
        can opt out only by explicitly setting `strict=False`.
        """
        mv = 0.0
        for p in self.positions.values():
            if p.symbol not in prices:
                if strict:
                    raise KeyError(
                        f"missing price for held position {p.symbol}; "
                        f"refusing to compute equity on partial price set"
                    )
                continue
            mv += p.market_value(prices[p.symbol])
        return self.cash + mv

    def position_qty(self, symbol: str) -> float:
        return self.positions[symbol].qty if symbol in self.positions else 0.0


# ---------------------------------------------------------------------------
# Order and fill records
# ---------------------------------------------------------------------------

@dataclass
class Order:
    symbol: str
    qty: float                       # signed: + buy, - sell
    order_type: str = "market"       # 'market' | 'limit'
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_id: str = field(default_factory=new_client_id)
    status: OrderStatus = OrderStatus.PENDING
    submitted_qty: float = 0.0
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    broker_order_id: str | None = None
    submitted_at: datetime | None = None
    parent_id: str | None = None     # for child orders of a TWAP/VWAP parent


@dataclass
class Fill:
    symbol: str
    qty: float
    price: float
    timestamp: datetime              # tz-aware UTC
    order_id: str
    client_id: str
    commission: float = 0.0


@dataclass
class Decision:
    """Snapshot of intent at the moment a rebalance was decided."""
    symbol: str
    decision_qty: float
    decision_price: float
    decision_time: datetime
    target_weight: float


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only JSONL log for every decision, submission, fill, cancel,
    and reconciliation event. Survives process restarts. The audit log
    is the answer to "what did the system do yesterday at 14:32" and
    its absence is the single most common reason post-trade
    investigations stall.

    Each line is a self-contained JSON object with an ISO-8601 UTC
    timestamp and a `kind` discriminator.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, payload: dict) -> None:
        payload = {**payload, "ts": utcnow().isoformat(), "kind": kind}
        with self.path.open("a") as f:
            f.write(json.dumps(payload, default=self._encode) + "\n")

    @staticmethod
    def _encode(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, enum.Enum):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"unserialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# Broker interface
# ---------------------------------------------------------------------------

class Broker(abc.ABC):
    """
    Minimal broker interface. Concrete implementations:
      - PaperBroker (this file): simulated fills, no network.
      - AlpacaBroker: thin adapter around alpaca-py (stub here; live
        connection optional).
    """

    @abc.abstractmethod
    def get_account(self) -> Account: ...

    @abc.abstractmethod
    def last_price(self, symbol: str) -> float: ...

    @abc.abstractmethod
    def submit_order(self, order: Order) -> str: ...

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> None: ...


class PaperBroker(Broker):
    """
    Simulated broker that fills orders against an in-memory price feed.

    Fills happen at `last_price * (1 + slippage_bps * sign(qty) / 1e4)`
    plus a flat `commission_bps`. This is intentionally simple: the
    point is to exercise the exact same code path that a real broker
    will run, and to log implementation-shortfall components in a
    controlled way.

    Bookkeeping invariant (verified in `test_execution.py`): for any
    sequence of orders against constant prices marked at the slipped
    fill price,
        cash + mark_to_market(positions) + commissions == starting_cash
    holds to floating-point tolerance. (Realised PnL is *already*
    reflected in `cash`, so adding it would double-count.)
    """

    def __init__(self, starting_cash: float = 100_000.0,
                 slippage_bps: float = 1.0, commission_bps: float = 0.5,
                 fill_ratio: float = 1.0):
        if not (0 < fill_ratio <= 1):
            raise ValueError("fill_ratio must be in (0, 1]")
        self._account = Account(cash=starting_cash)
        self._prices: dict[str, float] = {}
        self._slippage_bps = slippage_bps
        self._commission_bps = commission_bps
        self._fill_ratio = fill_ratio
        self._fills: list[Fill] = []
        self._orders_by_id: dict[str, Order] = {}
        self._orders_by_client_id: dict[str, str] = {}
        self._next_order_id = 1

    def update_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    def get_account(self) -> Account:
        return self._account

    def last_price(self, symbol: str) -> float:
        if symbol not in self._prices:
            raise KeyError(f"no price for {symbol}; call update_prices first")
        return self._prices[symbol]

    def submit_order(self, order: Order) -> str:
        # Idempotency: a retry with the same client_id and matching qty
        # returns the prior broker_order_id and does not fill twice. A
        # collision with *different* qty is almost always a caller bug
        # (forgot to refresh the client_id between rebalances) and is
        # raised loudly rather than silently dropped — a silent dedup
        # of a different order keeps the caller's order PENDING forever.
        if order.client_id in self._orders_by_client_id:
            prior_id = self._orders_by_client_id[order.client_id]
            prior = self._orders_by_id[prior_id]
            if abs(prior.qty - order.qty) > 1e-9 or prior.symbol != order.symbol:
                raise ValueError(
                    f"client_id collision: {order.client_id} previously "
                    f"submitted as {prior.symbol} qty={prior.qty}, now "
                    f"requested as {order.symbol} qty={order.qty}"
                )
            return prior_id

        if order.qty == 0:
            order.status = OrderStatus.REJECTED
            return ""

        order_id = f"PAPER-{self._next_order_id}"
        self._next_order_id += 1
        order.broker_order_id = order_id
        order.submitted_at = utcnow()
        order.status = OrderStatus.SUBMITTED
        self._orders_by_id[order_id] = order
        self._orders_by_client_id[order.client_id] = order_id

        fill_qty = order.qty * self._fill_ratio
        px = self.last_price(order.symbol)
        slip = self._slippage_bps / 1e4 * (1 if fill_qty > 0 else -1)
        fill_px = px * (1 + slip)
        notional = abs(fill_qty) * fill_px
        commission = notional * self._commission_bps / 1e4

        self._apply_fill(order, fill_qty, fill_px, commission)

        if abs(order.filled_qty) >= abs(order.qty) - 1e-9:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        return order_id

    def _apply_fill(self, order: Order, qty: float, price: float,
                    commission: float) -> None:
        """
        Update positions, cash, and realised PnL for one fill.

        Four position-transition cases, in mutually exclusive branches:

          * Open: no prior position.
          * Add: same-side increase. Weighted-average the basis.
          * Reduce (partial close): opposite side, smaller than the
            existing position. Realise PnL on the closed portion;
            basis on the remaining shares is unchanged.
          * Flatten / flip: opposite side, equal to or larger than the
            existing position. Realise PnL on the closed portion; if
            anything is left over, that's a new position at the fill
            price.

        The four-branch form below is what every textbook ledger uses;
        the test suite covers each branch (open, add, partial-close,
        full-close-or-flip) for both long and short sides. The trap to
        avoid: a "same-direction" check via `pos.qty * new_qty > 0`
        stays positive on a partial close, so a naive two-branch form
        silently corrupts the cost basis on every interim reduction.
        """
        pos = self._account.positions.get(order.symbol)

        if pos is None:
            self._account.positions[order.symbol] = Position(
                symbol=order.symbol, qty=qty, avg_price=price,
            )
        else:
            same_direction = pos.qty * qty > 0
            new_qty = pos.qty + qty

            if same_direction:
                # Add to existing position. Sign-agnostic weighted average:
                # for shorts the signs cancel through the division.
                pos.avg_price = (pos.avg_price * pos.qty + price * qty) / new_qty
                pos.qty = new_qty
            else:
                # Opposite direction: close some or all of the position.
                # `closed_qty` is the signed quantity of the closed
                # portion (carries the sign of pos.qty), so
                # `closed_qty * (price - avg_price)` is correctly signed
                # for both long-closes and short-covers.
                full_close_or_flip = abs(qty) >= abs(pos.qty)
                closed_qty = pos.qty if full_close_or_flip else -qty
                self._account.realized_pnl += closed_qty * (price - pos.avg_price)

                if full_close_or_flip:
                    if new_qty == 0:
                        del self._account.positions[order.symbol]
                    else:
                        pos.qty = new_qty
                        pos.avg_price = price   # flipped side's new basis
                else:
                    # Partial close: keep avg_price, just reduce qty.
                    pos.qty = new_qty

        self._account.cash -= qty * price + commission

        order.filled_qty += qty
        # Avg fill price is the qty-weighted average across all fills
        # for this order id.
        if order.avg_fill_price == 0.0:
            order.avg_fill_price = price
        else:
            total_filled_prev = order.filled_qty - qty
            order.avg_fill_price = (
                (order.avg_fill_price * total_filled_prev + price * qty)
                / order.filled_qty
            )

        self._fills.append(Fill(
            symbol=order.symbol, qty=qty, price=price,
            timestamp=utcnow(), order_id=order.broker_order_id,
            client_id=order.client_id, commission=commission,
        ))

    def cancel_order(self, order_id: str) -> None:
        order = self._orders_by_id.get(order_id)
        if order is None:
            return
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED,
                            OrderStatus.REJECTED):
            return
        order.status = OrderStatus.CANCELED

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def orders(self) -> list[Order]:
        return list(self._orders_by_id.values())


# ---------------------------------------------------------------------------
# Alpaca adapter (stub)
# ---------------------------------------------------------------------------

class AlpacaBroker(Broker):
    """
    Drop-in `Broker` for Alpaca paper or live trading.

    This implementation is a stub. To enable real connectivity, install
    `alpaca-py`, supply `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
    via environment (`.env`), and uncomment the body of each method to
    wire it to `alpaca.trading.TradingClient`.

    The stub exists to make the broker-agnostic shape of the
    execution layer concrete. A code reviewer can see the exact
    `alpaca-py` calls that would be wired in, without needing
    credentials provisioned to verify the rest of the system.
    """

    def __init__(self, api_key: str, api_secret: str,
                 paper: bool = True):
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper
        # from alpaca.trading.client import TradingClient
        # from alpaca.data.historical import StockHistoricalDataClient
        # self._client = TradingClient(api_key, api_secret, paper=paper)
        # self._data = StockHistoricalDataClient(api_key, api_secret)
        self._connected = False

    def get_account(self) -> Account:
        # acct = self._client.get_account()
        # positions = self._client.get_all_positions()
        # return Account(
        #     cash=float(acct.cash),
        #     positions={
        #         p.symbol: Position(p.symbol, float(p.qty), float(p.avg_entry_price))
        #         for p in positions
        #     },
        # )
        raise NotImplementedError("AlpacaBroker is a stub; install alpaca-py and uncomment")

    def last_price(self, symbol: str) -> float:
        # from alpaca.data.requests import StockLatestQuoteRequest
        # q = self._data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        # return (q[symbol].ask_price + q[symbol].bid_price) / 2
        raise NotImplementedError("AlpacaBroker is a stub")

    def submit_order(self, order: Order) -> str:
        # from alpaca.trading.requests import MarketOrderRequest
        # from alpaca.trading.enums import OrderSide, TimeInForce as AlpacaTIF
        # req = MarketOrderRequest(
        #     symbol=order.symbol,
        #     qty=abs(order.qty),
        #     side=OrderSide.BUY if order.qty > 0 else OrderSide.SELL,
        #     time_in_force=AlpacaTIF.DAY,
        #     client_order_id=order.client_id,
        # )
        # placed = self._client.submit_order(req)
        # return placed.id
        raise NotImplementedError("AlpacaBroker is a stub")

    def cancel_order(self, order_id: str) -> None:
        # self._client.cancel_order_by_id(order_id)
        raise NotImplementedError("AlpacaBroker is a stub")


# ---------------------------------------------------------------------------
# Weights -> target shares -> order list
# ---------------------------------------------------------------------------

def weights_to_target_shares(weights: pd.Series, prices: dict[str, float],
                             equity: float, allow_fractional: bool = False
                             ) -> pd.Series:
    """
    Convert a weight vector into share targets.

    Each name's target dollar exposure is `weight * equity`, divided by
    the last price. Long weights become positive share counts, short
    weights negative.

    For integer-only brokers, we round to nearest (not truncate). Round
    toward zero would bias short positions (truncate(-0.7) == 0 zeros
    out a small short while keeping a small long at +0.7 also zero; for
    larger fractions the directionality is symmetric but the residual
    notional is lost). Round to nearest minimises share drift but can
    open positions one share larger than the target — caught downstream
    by the min-notional and per-name notional risk gates.
    """
    targets = {}
    for sym, w in weights.dropna().items():
        if sym not in prices or prices[sym] <= 0:
            continue
        target_notional = w * equity
        raw_qty = target_notional / prices[sym]
        if allow_fractional:
            targets[sym] = raw_qty
        else:
            targets[sym] = float(round(raw_qty))
    return pd.Series(targets, dtype=float)


def build_order_list(target_shares: pd.Series, account: Account,
                     prices: dict[str, float],
                     min_order_notional: float = 1.0,
                     strict_prices: bool = True) -> list[Order]:
    """
    Diff target shares against current positions to produce orders.

    Skips deltas whose absolute notional is below `min_order_notional`,
    which suppresses the tickle of tiny rebalances that eat alpha in
    commissions and avoids broker rejections for sub-$1 orders.

    With `strict_prices=True`, a position held in the account without a
    current price raises. Silently emitting a flatten order at an
    unknown price is how stale-feed bugs become wire-transfers.
    """
    orders: list[Order] = []
    symbols = set(target_shares.index) | set(account.positions.keys())
    for sym in symbols:
        target = float(target_shares.get(sym, 0.0))
        current = account.position_qty(sym)
        delta = target - current
        if delta == 0:
            continue
        if sym not in prices:
            if strict_prices:
                raise KeyError(
                    f"no price for {sym} but it appears in the order list"
                )
            continue
        if abs(delta * prices[sym]) < min_order_notional:
            continue
        orders.append(Order(symbol=sym, qty=delta))
    return orders


# ---------------------------------------------------------------------------
# Execution algorithms
# ---------------------------------------------------------------------------

def execute_market(broker: Broker, orders: list[Order]) -> list[str]:
    """Fire every order at the current market price."""
    ids = []
    for o in orders:
        if o.qty == 0:
            continue
        ids.append(broker.submit_order(o))
    return ids


def twap_slices(parent: Order, n_slices: int) -> list[Order]:
    """
    Generate `n_slices` child orders that sum to `parent.qty` exactly.

    Returns a list of child `Order` objects. The caller schedules them
    against a clock — this function does no sleeping. The last child
    absorbs the residual so the sum matches the parent quantity to
    floating-point precision.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    children: list[Order] = []
    base = parent.qty / n_slices
    sent = 0.0
    for k in range(n_slices):
        if k == n_slices - 1:
            child_qty = parent.qty - sent
        else:
            child_qty = base
        sent += child_qty
        children.append(Order(
            symbol=parent.symbol, qty=child_qty,
            order_type=parent.order_type, limit_price=parent.limit_price,
            parent_id=parent.client_id,
        ))
    return children


def vwap_slices(parent: Order, profile: np.ndarray) -> list[Order]:
    """
    Generate child orders weighted by an intraday volume profile.

    `profile` must sum to 1 (e.g. from `vwap_profile`). A flat profile
    reduces to TWAP. The realistic shape is U-curved: bigger slices at
    the open and close, smaller during the lunch lull. The last child
    snaps to the parent residual.
    """
    profile = np.asarray(profile, dtype=float)
    if not np.isclose(profile.sum(), 1.0, atol=1e-6):
        raise ValueError("profile must sum to 1")
    children: list[Order] = []
    sent = 0.0
    n = len(profile)
    for k, frac in enumerate(profile):
        if k == n - 1:
            child_qty = parent.qty - sent
        else:
            child_qty = parent.qty * frac
        sent += child_qty
        children.append(Order(
            symbol=parent.symbol, qty=child_qty,
            order_type=parent.order_type, limit_price=parent.limit_price,
            parent_id=parent.client_id,
        ))
    return children


def vwap_profile(volume_bars: pd.DataFrame, n_buckets: int = 13) -> np.ndarray:
    """
    Build a normalised intraday volume profile from historical bars.

    `volume_bars` is rows of intraday volume. The function aggregates
    into `n_buckets` equal-time slots and normalises so the result sums
    to 1. The classical shape on US equities is U-curved (heavy at the
    open and close, light midday); for a flat input the function
    returns a uniform profile.
    """
    if volume_bars.empty:
        return np.full(n_buckets, 1.0 / n_buckets)
    edges = np.linspace(0, len(volume_bars), n_buckets + 1, dtype=int)
    profile = np.array([
        volume_bars.iloc[edges[i]:edges[i + 1]].sum().sum()
        for i in range(n_buckets)
    ], dtype=float)
    total = profile.sum()
    if total <= 0:
        return np.full(n_buckets, 1.0 / n_buckets)
    return profile / total


# ---------------------------------------------------------------------------
# Implementation shortfall
# ---------------------------------------------------------------------------

def implementation_shortfall(decisions: list[Decision], fills: list[Fill]
                             ) -> pd.DataFrame:
    """
    Per-symbol implementation shortfall in basis points.

    Shortfall = sign(qty) * (avg_fill_price - decision_price) / decision_price.

    Positive bps for buys (or negative for sells) means we paid up.
    Consistently positive shortfall across many trades indicates the
    alpha is being eaten by impact and the execution algo should be
    more passive or the rebalance frequency cut.
    """
    by_symbol: dict[str, list[Fill]] = {}
    for f in fills:
        by_symbol.setdefault(f.symbol, []).append(f)

    rows = []
    for d in decisions:
        if d.decision_qty == 0:
            continue
        sym_fills = by_symbol.get(d.symbol, [])
        if not sym_fills:
            continue
        total_qty = sum(f.qty for f in sym_fills)
        if total_qty == 0:
            continue
        avg_px = sum(f.qty * f.price for f in sym_fills) / total_qty
        sign = 1 if d.decision_qty > 0 else -1
        bps = sign * (avg_px - d.decision_price) / d.decision_price * 1e4
        rows.append({
            "symbol": d.symbol,
            "decision_qty": d.decision_qty,
            "filled_qty": total_qty,
            "decision_price": d.decision_price,
            "avg_fill_price": avg_px,
            "shortfall_bps": bps,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rebalance entry point
# ---------------------------------------------------------------------------

def rebalance(broker: Broker, weights: pd.Series, prices: dict[str, float],
              audit: AuditLog | None = None,
              risk_check=None,
              algo: str = "market",
              twap_slices_n: int | None = None,
              vwap_profile_arr: np.ndarray | None = None,
              allow_fractional: bool = False,
              min_order_notional: float = 1.0) -> dict:
    """
    End-to-end rebalance: size to current equity, diff against the
    book, run pre-trade risk checks, and execute.

    `weights` is assumed to already incorporate any signal-side decay
    (handled in `combine_weights.apply_signal_decay`). This function
    does not touch decay; the optimiser's constraints would be
    violated by post-hoc weight scaling.

    `risk_check` is an optional callable that takes (orders, account,
    prices, equity) and returns either None (clear) or a string
    explaining the breach. The rebalance is rejected on any non-None
    return. See `risk.PreTradeRiskGate` for the standard
    implementation.

    Returns a dict with status, equity, orders submitted, decisions,
    and implementation-shortfall frame. The same payload is appended
    to the audit log.
    """
    account = broker.get_account()
    equity = account.equity(prices)

    targets = weights_to_target_shares(
        weights, prices, equity, allow_fractional=allow_fractional,
    )
    orders = build_order_list(
        targets, account, prices, min_order_notional=min_order_notional,
    )

    if risk_check is not None:
        breach = risk_check(orders, account, prices, equity)
        if breach is not None:
            result = {
                "status": "risk_breach", "reason": breach,
                "equity": equity, "n_orders_proposed": len(orders),
            }
            if audit is not None:
                audit.write("rebalance_rejected", result)
            return result

    decisions = [Decision(
        symbol=o.symbol, decision_qty=o.qty,
        decision_price=prices[o.symbol],
        decision_time=utcnow(),
        target_weight=float(weights.get(o.symbol, 0.0)),
    ) for o in orders]

    if algo == "market":
        ids = execute_market(broker, orders)
    elif algo == "twap":
        if twap_slices_n is None:
            raise ValueError("twap requires twap_slices_n")
        ids = []
        for parent in orders:
            for child in twap_slices(parent, twap_slices_n):
                ids.append(broker.submit_order(child))
    elif algo == "vwap":
        if vwap_profile_arr is None:
            raise ValueError("vwap requires vwap_profile_arr")
        ids = []
        for parent in orders:
            for child in vwap_slices(parent, vwap_profile_arr):
                ids.append(broker.submit_order(child))
    else:
        raise ValueError(f"unknown algo: {algo}")

    fills = getattr(broker, "fills", [])
    recent = [f for f in fills if f.order_id in ids]
    shortfall = implementation_shortfall(decisions, recent)

    result = {
        "status": "executed",
        "equity": equity,
        "n_orders": len(orders),
        "order_ids": ids,
        "decisions": decisions,
        "shortfall": shortfall.to_dict(orient="records"),
    }
    if audit is not None:
        audit.write("rebalance", result)
    return result
