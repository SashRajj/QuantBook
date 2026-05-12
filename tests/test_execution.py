"""
Execution-layer tests.

The bookkeeping invariant is non-negotiable: for any sequence of orders
against constant prices, every dollar must be accounted for as cash,
position market value, realised PnL, or commission. If this test fails,
the rest of the execution stack is unsafe to look at.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from execution import (
    Account,
    AlpacaBroker,
    AuditLog,
    Order,
    OrderStatus,
    PaperBroker,
    Position,
    build_order_list,
    execute_market,
    implementation_shortfall,
    new_client_id,
    rebalance,
    twap_slices,
    vwap_profile,
    vwap_slices,
    weights_to_target_shares,
    Decision,
)
from risk import PreTradeRiskGate, RiskLimits, adv_dollar_from_prices


# ---------------------------------------------------------------------------
# PaperBroker bookkeeping invariants
# ---------------------------------------------------------------------------

class PaperBrokerBookkeepingTests(unittest.TestCase):

    def _total_value(self, broker, prices):
        acct = broker.get_account()
        mv = sum(p.qty * prices[p.symbol] for p in acct.positions.values())
        return acct.cash + mv

    def _total_commissions(self, broker):
        return sum(f.commission for f in broker.fills)

    def _slippage_cost(self, broker):
        # Slippage moves cash unfavourably but does not destroy value -
        # the position is on the books at the slipped fill price, and
        # if prices stay constant the slipped basis is captured in the
        # position market value at the mark. The test below uses the
        # mark equal to the slipped fill price; with prices == fill_px
        # exactly, there is no slippage drift to subtract.
        return 0.0

    def test_cash_plus_mv_equals_start_minus_commissions(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=2.0)
        broker.update_prices({"AAA": 50.0, "BBB": 25.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        broker.submit_order(Order(symbol="BBB", qty=-200))
        # Mark at the same prices we submitted at: no MtM gain/loss.
        end = self._total_value(broker, {"AAA": 50.0, "BBB": 25.0})
        commissions = self._total_commissions(broker)
        # cash + mv == starting_cash - commissions exactly.
        self.assertAlmostEqual(end + commissions, 100_000.0, places=4)

    def test_long_to_flat_books_correct_pnl(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        broker.update_prices({"AAA": 60.0})
        broker.submit_order(Order(symbol="AAA", qty=-100))
        # Bought at 50, sold at 60, 100 shares -> +$1000 realised.
        self.assertAlmostEqual(broker.get_account().realized_pnl,
                               1000.0, places=4)
        self.assertEqual(broker.get_account().positions, {})

    def test_long_to_short_flip_books_correct_pnl(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        broker.update_prices({"AAA": 60.0})
        broker.submit_order(Order(symbol="AAA", qty=-150))  # flat the 100, short 50
        # Realised on the closed 100 long: 100 * (60 - 50) = +$1000.
        self.assertAlmostEqual(broker.get_account().realized_pnl,
                               1000.0, places=4)
        # Remaining position: 50 short at basis 60.
        pos = broker.get_account().positions["AAA"]
        self.assertAlmostEqual(pos.qty, -50.0)
        self.assertAlmostEqual(pos.avg_price, 60.0)

    def test_short_to_long_flip_books_correct_pnl(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=-100))  # short at 50
        broker.update_prices({"AAA": 40.0})
        broker.submit_order(Order(symbol="AAA", qty=150))   # cover 100, long 50
        # Realised on the closed 100 short: -100 * (40 - 50) = +$1000 profit.
        self.assertAlmostEqual(broker.get_account().realized_pnl,
                               1000.0, places=4)
        pos = broker.get_account().positions["AAA"]
        self.assertAlmostEqual(pos.qty, 50.0)
        self.assertAlmostEqual(pos.avg_price, 40.0)

    def test_short_to_flat_books_correct_pnl(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=-100))
        broker.update_prices({"AAA": 40.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        # Short at 50, covered at 40: 100 * (50 - 40) = +$1000.
        self.assertAlmostEqual(broker.get_account().realized_pnl,
                               1000.0, places=4)
        self.assertEqual(broker.get_account().positions, {})

    def test_partial_close_books_pnl_and_preserves_long_basis(self):
        """
        A reduction that does not cross zero (e.g. long 100, sell 30)
        must realise PnL on the closed portion and leave the basis on
        the remaining shares unchanged. Falling into the same-side-add
        branch would silently corrupt cost basis and zero realised PnL
        until the position finally flattened.
        """
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        broker.update_prices({"AAA": 60.0})
        broker.submit_order(Order(symbol="AAA", qty=-30))
        pos = broker.get_account().positions["AAA"]
        # Closed 30 shares at +$10/share = +$300 realised.
        self.assertAlmostEqual(broker.get_account().realized_pnl, 300.0)
        # Remaining 70 shares stay on the original basis.
        self.assertAlmostEqual(pos.qty, 70.0)
        self.assertAlmostEqual(pos.avg_price, 50.0)

    def test_partial_cover_books_pnl_and_preserves_short_basis(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=-100))  # short at 50
        broker.update_prices({"AAA": 40.0})
        broker.submit_order(Order(symbol="AAA", qty=30))    # partial cover
        pos = broker.get_account().positions["AAA"]
        # Covered 30 shares at -$10/share (profit on short) = +$300.
        self.assertAlmostEqual(broker.get_account().realized_pnl, 300.0)
        # Remaining 70 short shares stay on the original basis.
        self.assertAlmostEqual(pos.qty, -70.0)
        self.assertAlmostEqual(pos.avg_price, 50.0)

    def test_same_side_add_weighted_average_basis(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        broker.update_prices({"AAA": 60.0})
        broker.submit_order(Order(symbol="AAA", qty=100))
        pos = broker.get_account().positions["AAA"]
        # Basis is weighted average: (50*100 + 60*100) / 200 = 55.
        self.assertAlmostEqual(pos.qty, 200.0)
        self.assertAlmostEqual(pos.avg_price, 55.0)
        self.assertAlmostEqual(broker.get_account().realized_pnl, 0.0)

    def test_idempotent_client_id_dedupes_submission(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        cid = new_client_id()
        o1 = Order(symbol="AAA", qty=100, client_id=cid)
        first_id = broker.submit_order(o1)
        # Same client_id, identical order -> returns same broker id, no second fill.
        o2 = Order(symbol="AAA", qty=100, client_id=cid)
        second_id = broker.submit_order(o2)
        self.assertEqual(first_id, second_id)
        self.assertAlmostEqual(broker.get_account().positions["AAA"].qty,
                               100.0)

    def test_idempotency_conflict_raises_on_different_qty(self):
        """
        Silent dedup of a different order is an operational landmine:
        caller resubmits with the same client_id but a different qty,
        broker drops the new one, caller's order stays PENDING
        forever. Loud failure is mandatory.
        """
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        cid = new_client_id()
        broker.submit_order(Order(symbol="AAA", qty=100, client_id=cid))
        with self.assertRaises(ValueError):
            broker.submit_order(Order(symbol="AAA", qty=50, client_id=cid))
        with self.assertRaises(ValueError):
            broker.submit_order(Order(symbol="BBB", qty=100, client_id=cid))

    def test_order_status_transitions(self):
        broker = PaperBroker(starting_cash=100_000.0)
        broker.update_prices({"AAA": 50.0})
        o = Order(symbol="AAA", qty=100)
        self.assertEqual(o.status, OrderStatus.PENDING)
        broker.submit_order(o)
        self.assertEqual(o.status, OrderStatus.FILLED)

    def test_partial_fill_status(self):
        broker = PaperBroker(starting_cash=100_000.0, fill_ratio=0.5)
        broker.update_prices({"AAA": 50.0})
        o = Order(symbol="AAA", qty=100)
        broker.submit_order(o)
        self.assertEqual(o.status, OrderStatus.PARTIALLY_FILLED)
        self.assertAlmostEqual(o.filled_qty, 50.0)


# ---------------------------------------------------------------------------
# Sizing and order generation
# ---------------------------------------------------------------------------

class SizingTests(unittest.TestCase):

    def test_weights_to_target_shares_rounds_to_nearest(self):
        weights = pd.Series({"AAA": 0.10, "BBB": -0.05})
        prices = {"AAA": 100.0, "BBB": 33.0}
        shares = weights_to_target_shares(weights, prices, equity=10_000.0)
        # AAA: target_notional = 1000, /100 = 10 shares.
        self.assertAlmostEqual(shares["AAA"], 10.0)
        # BBB: target_notional = -500, /33 = -15.15 -> round to -15.
        self.assertAlmostEqual(shares["BBB"], -15.0)

    def test_fractional_shares_pass_through(self):
        weights = pd.Series({"AAA": 0.10})
        prices = {"AAA": 33.0}
        shares = weights_to_target_shares(weights, prices, equity=10_000.0,
                                          allow_fractional=True)
        self.assertAlmostEqual(shares["AAA"], 1000.0 / 33.0)

    def test_build_order_list_strict_prices(self):
        acct = Account(cash=10_000.0,
                       positions={"AAA": Position("AAA", 100, 50.0),
                                  "BBB": Position("BBB", 50, 25.0)})
        targets = pd.Series({"AAA": 50.0})  # missing BBB price intentionally
        with self.assertRaises(KeyError):
            build_order_list(targets, acct, prices={"AAA": 50.0},
                             strict_prices=True)

    def test_build_order_list_skips_below_min_notional(self):
        acct = Account(cash=10_000.0,
                       positions={"AAA": Position("AAA", 100, 50.0)})
        # Target only one share off -> $50 delta, above $1 min.
        targets = pd.Series({"AAA": 101.0})
        prices = {"AAA": 50.0}
        orders = build_order_list(targets, acct, prices,
                                  min_order_notional=100.0)
        # Delta notional = 1 * 50 = 50, below $100 min -> skipped.
        self.assertEqual(len(orders), 0)


# ---------------------------------------------------------------------------
# Execution algos
# ---------------------------------------------------------------------------

class ExecutionAlgoTests(unittest.TestCase):

    def test_twap_slices_sum_exactly(self):
        parent = Order(symbol="AAA", qty=103)
        children = twap_slices(parent, n_slices=10)
        self.assertEqual(len(children), 10)
        total = sum(c.qty for c in children)
        self.assertAlmostEqual(total, 103.0, places=9)

    def test_twap_slices_handle_negative_qty(self):
        parent = Order(symbol="AAA", qty=-77)
        children = twap_slices(parent, n_slices=7)
        total = sum(c.qty for c in children)
        self.assertAlmostEqual(total, -77.0, places=9)

    def test_vwap_slices_sum_exactly_with_uneven_profile(self):
        parent = Order(symbol="AAA", qty=200)
        profile = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.15])
        profile = profile / profile.sum()
        children = vwap_slices(parent, profile)
        self.assertEqual(len(children), 6)
        total = sum(c.qty for c in children)
        self.assertAlmostEqual(total, 200.0, places=6)

    def test_vwap_profile_rejects_non_normalised(self):
        parent = Order(symbol="AAA", qty=100)
        with self.assertRaises(ValueError):
            vwap_slices(parent, np.array([0.5, 0.4, 0.05]))

    def test_vwap_profile_uniform_falls_back_to_twap_shape(self):
        # A flat profile should produce equal slices, same as TWAP.
        parent = Order(symbol="AAA", qty=100)
        twap = twap_slices(parent, n_slices=5)
        vwap = vwap_slices(parent, np.full(5, 0.2))
        for a, b in zip(twap, vwap):
            self.assertAlmostEqual(a.qty, b.qty, places=9)


# ---------------------------------------------------------------------------
# Implementation shortfall
# ---------------------------------------------------------------------------

class ImplementationShortfallTests(unittest.TestCase):

    def test_shortfall_positive_when_buy_fills_above_decision_price(self):
        from datetime import datetime, timezone
        d = Decision(symbol="AAA", decision_qty=100,
                     decision_price=50.0,
                     decision_time=datetime.now(timezone.utc),
                     target_weight=0.1)
        from execution import Fill
        fills = [Fill(symbol="AAA", qty=100, price=50.05,
                      timestamp=datetime.now(timezone.utc),
                      order_id="X", client_id="c")]
        df = implementation_shortfall([d], fills)
        # Paid 5 bps over decision price: shortfall = +10 bps.
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["shortfall_bps"], 10.0, places=2)


# ---------------------------------------------------------------------------
# Risk gates
# ---------------------------------------------------------------------------

class RiskGateTests(unittest.TestCase):

    def _gate(self, **kwargs):
        limits = RiskLimits(**kwargs)
        return PreTradeRiskGate(limits, high_water_mark=100_000.0,
                                adv_dollar={"AAA": 1e9, "BBB": 1e9})

    def test_passes_under_all_limits(self):
        gate = self._gate(max_gross_leverage=1.0, max_net_exposure=0.05,
                          max_position_pct=0.1, max_positions=10,
                          max_adv_participation_pct=0.5,
                          max_daily_turnover_pct=2.0,
                          drawdown_kill_pct=0.5)
        acct = Account(cash=100_000.0)
        orders = [Order(symbol="AAA", qty=100)]
        prices = {"AAA": 50.0, "BBB": 50.0}
        self.assertIsNone(gate.check(orders, acct, prices, 100_000.0))

    def test_blocks_drawdown_breach(self):
        gate = self._gate(drawdown_kill_pct=0.10)
        # Equity has dropped 15% from HWM.
        breach = gate.check([], Account(cash=85_000.0),
                            prices={}, equity=85_000.0)
        self.assertIsNotNone(breach)
        self.assertIn("drawdown", breach)

    def test_blocks_gross_leverage(self):
        gate = self._gate(max_gross_leverage=0.5)
        acct = Account(cash=100_000.0)
        # Single huge order pushes gross above 0.5 * equity.
        orders = [Order(symbol="AAA", qty=2000)]  # 2000 * 50 = 100k
        breach = gate.check(orders, acct, {"AAA": 50.0}, 100_000.0)
        self.assertIsNotNone(breach)
        self.assertIn("gross leverage", breach)

    def test_blocks_per_name_cap(self):
        gate = self._gate(max_gross_leverage=10.0, max_position_pct=0.05,
                          max_adv_participation_pct=None,
                          max_daily_turnover_pct=10.0)
        acct = Account(cash=100_000.0)
        orders = [Order(symbol="AAA", qty=200)]  # 200 * 50 = 10k = 10% > 5% cap
        breach = gate.check(orders, acct, {"AAA": 50.0}, 100_000.0)
        self.assertIsNotNone(breach)
        self.assertIn("per-name", breach)

    def test_blocks_blocked_symbol(self):
        from risk import RiskLimits
        limits = RiskLimits(blocked_symbols={"AAA"})
        gate = PreTradeRiskGate(limits, high_water_mark=100_000.0,
                                adv_dollar={"AAA": 1e9})
        orders = [Order(symbol="AAA", qty=10)]
        breach = gate.check(orders, Account(cash=100_000.0),
                            {"AAA": 50.0}, 100_000.0)
        self.assertIsNotNone(breach)
        self.assertIn("blocked", breach)


# ---------------------------------------------------------------------------
# Rebalance end-to-end
# ---------------------------------------------------------------------------

class RebalanceTests(unittest.TestCase):

    def test_rebalance_market_executes_and_logs(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0, "BBB": 25.0})
        weights = pd.Series({"AAA": 0.1, "BBB": -0.05})
        result = rebalance(broker, weights, {"AAA": 50.0, "BBB": 25.0},
                           algo="market")
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["n_orders"], 2)
        # AAA target: 0.1 * 100k = 10k / 50 = 200 shares.
        self.assertAlmostEqual(broker.get_account().positions["AAA"].qty, 200)
        self.assertAlmostEqual(broker.get_account().positions["BBB"].qty, -200)

    def test_rebalance_rejected_on_risk_breach(self):
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        weights = pd.Series({"AAA": 0.5})  # 50% in one name
        gate = PreTradeRiskGate(
            RiskLimits(max_position_pct=0.10,
                       max_adv_participation_pct=None),
            high_water_mark=100_000.0,
        )
        result = rebalance(broker, weights, {"AAA": 50.0},
                           algo="market", risk_check=gate.check)
        self.assertEqual(result["status"], "risk_breach")
        self.assertEqual(broker.get_account().positions, {})

    def test_audit_log_appends(self):
        tmp = Path("/tmp/test_qr_audit.jsonl")
        if tmp.exists():
            tmp.unlink()
        audit = AuditLog(tmp)
        broker = PaperBroker(starting_cash=100_000.0,
                             slippage_bps=0.0, commission_bps=0.0)
        broker.update_prices({"AAA": 50.0})
        weights = pd.Series({"AAA": 0.1})
        rebalance(broker, weights, {"AAA": 50.0}, algo="market", audit=audit)
        lines = tmp.read_text().splitlines()
        self.assertGreater(len(lines), 0)
        import json
        for line in lines:
            obj = json.loads(line)
            self.assertIn("ts", obj)
            self.assertIn("kind", obj)


# ---------------------------------------------------------------------------
# Alpaca stub
# ---------------------------------------------------------------------------

class AlpacaBrokerStubTests(unittest.TestCase):

    def test_methods_raise_until_wired(self):
        b = AlpacaBroker(api_key="fake", api_secret="fake", paper=True)
        with self.assertRaises(NotImplementedError):
            b.get_account()
        with self.assertRaises(NotImplementedError):
            b.last_price("AAPL")
        with self.assertRaises(NotImplementedError):
            b.submit_order(Order(symbol="AAPL", qty=1))


# ---------------------------------------------------------------------------
# Feed health (real timestamps, not synthetic)
# ---------------------------------------------------------------------------

class FeedHealthTests(unittest.TestCase):

    def test_fresh_bar_is_not_stale(self):
        from datetime import datetime, timezone, timedelta
        from price_feed import feed_health
        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        bar = pd.Timestamp(now - timedelta(hours=4))   # 4 hours old
        h = feed_health(last_bar_ts=bar, n_received=10, n_expected=10,
                        sla_seconds=86400 * 2, now=now)
        self.assertFalse(h.is_stale)
        self.assertAlmostEqual(h.age_seconds, 4 * 3600, places=1)

    def test_old_bar_is_stale_with_real_timestamp(self):
        """
        Feed-health must measure actual data age, not wall-clock age:
        a 10-day-old bar trips is_stale=True under a 2-day SLA even if
        the runner is invoked at any time. Wrapping prices in a panel
        timestamped to "today" would silently disable the check.
        """
        from datetime import datetime, timezone, timedelta
        from price_feed import feed_health
        now = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        bar = pd.Timestamp(now - timedelta(days=10))
        h = feed_health(last_bar_ts=bar, n_received=10, n_expected=10,
                        sla_seconds=86400 * 2, now=now)
        self.assertTrue(h.is_stale)

    def test_no_bar_is_stale(self):
        from price_feed import feed_health
        h = feed_health(last_bar_ts=None, n_received=0, n_expected=10,
                        sla_seconds=86400)
        self.assertTrue(h.is_stale)
        self.assertEqual(h.age_seconds, float("inf"))


# ---------------------------------------------------------------------------
# Runner state persistence (HWM)
# ---------------------------------------------------------------------------

class RunnerStateTests(unittest.TestCase):
    """
    The runner persists the drawdown high-water mark between cron
    invocations. Without persistence the HWM resets every run and the
    drawdown kill-switch can never fire on a multi-day decline.
    """

    def test_state_round_trips_via_json(self):
        from runner import _load_state, _save_state
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log_dir = Path(d)
            self.assertEqual(_load_state(log_dir), {})
            _save_state(log_dir, {"high_water_mark": 123456.78,
                                  "last_run_iso": "2026-05-12T00:00:00+00:00"})
            state = _load_state(log_dir)
            self.assertAlmostEqual(state["high_water_mark"], 123456.78)
            self.assertEqual(state["last_run_iso"], "2026-05-12T00:00:00+00:00")

    def test_persisted_hwm_keeps_drawdown_gate_armed(self):
        """
        Simulate two runs: run 1 hits HWM=120k, run 2 has equity=100k.
        The gate must see a 16.7% drawdown and reject — not start fresh
        at the lower equity and silently let trading continue.
        """
        # Run 1: HWM is 120k.
        gate = PreTradeRiskGate(
            RiskLimits(drawdown_kill_pct=0.15,
                       max_adv_participation_pct=None),
            high_water_mark=120_000.0,
        )
        # Run 2 equity = 100k. -16.7% from HWM, beyond 15% kill.
        breach = gate.check([], Account(cash=100_000.0),
                            prices={}, equity=100_000.0)
        self.assertIsNotNone(breach)
        self.assertIn("drawdown", breach)


if __name__ == "__main__":
    unittest.main()
