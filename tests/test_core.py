import unittest

import numpy as np
import pandas as pd

import download_sp500
from helper import (
    Optimizer,
    beta_to,
    deflated_sharpe,
    ic,
    ic_weighted_combine,
    neutralize,
    port_ret,
    probabilistic_sharpe,
    quick_weights,
    stats,
    var_cvar,
)
from signals import (
    low_volatility,
    market_residual_momentum,
    mean_reversion,
    momentum,
    volume_adjusted_momentum,
)


class CoreFunctionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2020-01-01", periods=80, freq="B")
        base = np.linspace(100, 140, len(self.dates))
        self.close = pd.DataFrame(
            {
                "AAA": base,
                "BBB": base[::-1] + 20,
                "CCC": 100 + 3 * np.sin(np.arange(len(self.dates)) / 4),
                "DDD": 80 + np.arange(len(self.dates)) * 0.2,
            },
            index=self.dates,
        )
        self.returns = self.close.pct_change().fillna(0)

    def test_signals_preserve_shape_and_use_trailing_windows(self):
        mr = mean_reversion(self.close, lookback=10)
        mom = momentum(self.close, lookback=20, skip=5)
        lv = low_volatility(self.close, lookback=10)

        self.assertEqual(mr.shape, self.close.shape)
        self.assertEqual(mom.shape, self.close.shape)
        self.assertEqual(lv.shape, self.close.shape)
        self.assertTrue(mr.iloc[:9].isna().all().all())
        self.assertTrue(mom.iloc[:20].isna().all().all())
        self.assertTrue(lv.iloc[:10].isna().all().all())

    def test_portfolio_helpers_return_expected_shapes(self):
        signal = mean_reversion(self.close, lookback=10)
        weights = quick_weights(signal, dollar_neutral=True)
        pnl = port_ret(weights, self.returns, tcost_bps=1)
        summary = stats(pnl.dropna(), weights=weights, plot=False)
        risk = var_cvar(pnl, alpha=0.05)
        info_coef = ic(signal, self.close, horizons=(1, 5))

        self.assertEqual(pnl.index.tolist(), self.close.index.tolist())
        self.assertIn("sharpe", summary.columns)
        self.assertEqual(risk.index.tolist(), ["historical", "parametric"])
        self.assertEqual(info_coef.index.tolist(), [1, 5])

    def test_optimizer_produces_bounded_dollar_neutral_weights(self):
        signal = mean_reversion(self.close, lookback=10).dropna(how="all")
        returns = self.returns.reindex(signal.index)

        opt = Optimizer(signal, returns)
        weights = opt.run(dollar_neutral=True, max_position=0.25,
                          max_leverage=1.0, subsample=10)

        self.assertFalse(weights.empty)
        self.assertLessEqual(weights.abs().max().max(), 0.2501)
        self.assertLess(weights.sum(axis=1).abs().max(), 1e-3)

    def test_optimizer_respects_member_mask(self):
        signal = mean_reversion(self.close, lookback=10).dropna(how="all")
        returns = self.returns.reindex(signal.index)
        mask = pd.DataFrame(True, index=signal.index, columns=signal.columns)
        mask["DDD"] = False

        opt = Optimizer(signal, returns)
        weights = opt.run(dollar_neutral=True, max_position=0.25,
                          max_leverage=1.0, subsample=5, member_mask=mask)

        self.assertFalse(weights.empty)
        # Solver tolerance leaves a tiny residual; mask suppression below
        # 1e-4 is enough to confirm names are excluded from the universe.
        self.assertLess(weights["DDD"].abs().max(), 1e-4)
        self.assertGreater(weights[["AAA", "BBB", "CCC"]].abs().sum().sum(), 0)

    def test_factor_helpers_align_and_neutralize(self):
        benchmark = self.returns.mean(axis=1)
        betas = beta_to(self.returns, benchmark, lookback=20)
        signal = mean_reversion(self.close, lookback=10)
        neutral = neutralize(signal, betas)

        self.assertEqual(betas.shape, self.returns.shape)
        self.assertEqual(neutral.shape, signal.loc[betas.index].shape)

    def test_port_ret_applies_exec_lag_and_asymmetric_costs(self):
        signal = mean_reversion(self.close, lookback=10)
        weights = quick_weights(signal, dollar_neutral=True)

        no_cost = port_ret(weights, self.returns, tcost_bps=0, exec_lag=2)
        sym_cost = port_ret(weights, self.returns, tcost_bps=5, exec_lag=2)
        asym_cost = port_ret(weights, self.returns, tcost_bps=5,
                             tcost_short_bps=50, exec_lag=2)

        # First two rows are NaN under exec_lag=2.
        self.assertTrue(np.isnan(no_cost.iloc[:2]).all())
        # Symmetric costs strictly reduce total PnL after burn-in.
        self.assertLess(sym_cost.iloc[10:].sum(), no_cost.iloc[10:].sum())
        # Asymmetric costs (higher short side) reduce PnL further when
        # the strategy actually holds short positions.
        self.assertLessEqual(asym_cost.iloc[10:].sum(), sym_cost.iloc[10:].sum())

    def test_port_ret_borrow_fee_charged_on_short_book(self):
        # Synthetic always-short portfolio so the borrow term is non-zero.
        weights = pd.DataFrame(-0.25, index=self.dates,
                               columns=self.close.columns)
        rets = pd.DataFrame(0.0, index=self.dates, columns=self.close.columns)

        no_borrow = port_ret(weights, rets, exec_lag=2, borrow_bps_annual=0)
        with_borrow = port_ret(weights, rets, exec_lag=2, borrow_bps_annual=200)

        self.assertAlmostEqual(no_borrow.dropna().sum(), 0.0, places=8)
        self.assertLess(with_borrow.dropna().sum(), 0)

    def test_probabilistic_and_deflated_sharpe(self):
        rng = np.random.default_rng(0)
        # Mean/std chosen so the realised annual Sharpe exceeds 1.5 even
        # with sample noise, putting PSR firmly above 0.95.
        good = pd.Series(rng.normal(0.0015, 0.01, 2000))
        zero = pd.Series(rng.normal(0.0, 0.01, 2000))

        psr_good = probabilistic_sharpe(good)
        psr_zero = probabilistic_sharpe(zero)
        self.assertGreater(psr_good, 0.95)
        self.assertLess(psr_zero, 0.9)
        self.assertGreaterEqual(psr_zero, 0.0)
        self.assertLessEqual(psr_good, 1.0)

        # DSR must be no higher than PSR when n_trials > 1.
        dsr = deflated_sharpe(good, n_trials=20)
        self.assertLessEqual(dsr, psr_good + 1e-9)

    def test_volume_adjusted_momentum_shape_and_no_lookahead(self):
        rng = np.random.default_rng(3)
        dates = pd.date_range("2022-01-01", periods=120, freq="D")
        cols = ["AAA", "BBB", "CCC"]
        close = pd.DataFrame(rng.lognormal(0, 0.02, (len(dates), 3)),
                             index=dates, columns=cols).cumprod() * 100
        volume = pd.DataFrame(rng.lognormal(15, 0.5, (len(dates), 3)),
                              index=dates, columns=cols)
        sig = volume_adjusted_momentum(close, volume, lookback=10)
        self.assertEqual(sig.shape, close.shape)
        self.assertTrue(sig.iloc[:10].isna().all().all())
        # Changing a future row must not change a past signal value.
        perturbed = close.copy()
        perturbed.iloc[-1] *= 2
        sig2 = volume_adjusted_momentum(perturbed, volume, lookback=10)
        pd.testing.assert_frame_equal(sig.iloc[:-1].dropna(how="all"),
                                      sig2.iloc[:-1].dropna(how="all"))

    def test_market_residual_momentum_matches_shape_and_uses_only_past(self):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        mkt = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates).cumsum() + 100
        # Stocks: market beta 1 + idio noise
        cols = ["AAA", "BBB", "CCC", "DDD"]
        idio = pd.DataFrame(rng.normal(0, 0.005, (len(dates), len(cols))),
                            index=dates, columns=cols).cumsum()
        close = idio.add(mkt, axis=0)

        sig = market_residual_momentum(close, mkt, lookback=60, skip=5,
                                       beta_lookback=60)
        self.assertEqual(sig.shape, close.shape)
        # Before the rolling window is full, the signal is undefined.
        self.assertTrue(sig.iloc[:65].isna().all().all())
        # No look-ahead: changing a future price must not change a past signal.
        sig_before = sig.copy()
        close_perturbed = close.copy()
        close_perturbed.iloc[-1] *= 2
        sig_after = market_residual_momentum(close_perturbed, mkt,
                                             lookback=60, skip=5,
                                             beta_lookback=60)
        # Drop the last row where the perturbation legitimately changes things.
        pd.testing.assert_frame_equal(
            sig_before.iloc[:-1].dropna(how="all"),
            sig_after.iloc[:-1].dropna(how="all"),
        )

    def test_ic_weighted_combiner_downweights_pure_noise(self):
        rng = np.random.default_rng(2)
        dates = pd.date_range("2018-01-01", periods=500, freq="B")
        cols = [f"N{i}" for i in range(20)]

        # Build prices so true 1-day forward return = +1 * informative_signal + noise.
        informative = pd.DataFrame(rng.normal(0, 1, (len(dates), len(cols))),
                                   index=dates, columns=cols)
        noise = pd.DataFrame(rng.normal(0, 1, (len(dates), len(cols))),
                             index=dates, columns=cols)
        # Construct prices so that next-day return is informative.shift(-1)
        # by setting close_t+1 / close_t = 1 + 0.01 * informative_t.
        ret = 0.01 * informative.shift(1).fillna(0)
        close = (1 + ret).cumprod() * 100

        combined, weights = ic_weighted_combine(
            {"good": informative, "noise": noise},
            close,
            lookback=60,
        )

        # After warm-up the good signal should consistently win more weight.
        late_w = weights.iloc[200:].mean()
        self.assertGreater(late_w["good"], late_w["noise"])
        self.assertGreater(late_w["good"], 0.5)

    def test_downloader_helpers_are_import_safe(self):
        self.assertTrue(hasattr(download_sp500, "main"))
        self.assertTrue(hasattr(download_sp500, "reconstruct_membership"))
        self.assertTrue(hasattr(download_sp500, "get_change_history"))


if __name__ == "__main__":
    unittest.main()
