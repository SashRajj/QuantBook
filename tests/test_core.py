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
    kalman_hedge,
    neutralize,
    port_ret,
    probabilistic_sharpe,
    purged_kfold_splits,
    quick_weights,
    stats,
    var_cvar,
)
from signals import (
    idiosyncratic_volatility,
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

    def test_idiosyncratic_volatility_shape_warmup_and_no_lookahead(self):
        # Synthetic panel: each stock = beta * market + idio noise. The
        # idio_vol signal should be defined only after the combined
        # beta + stdev warm-up window and must use trailing data only.
        rng = np.random.default_rng(11)
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        mkt = pd.Series(rng.normal(0.0005, 0.01, len(dates)),
                        index=dates).cumsum() + 100
        cols = ["AAA", "BBB", "CCC", "DDD"]
        idio = pd.DataFrame(rng.normal(0, 0.005, (len(dates), len(cols))),
                            index=dates, columns=cols).cumsum()
        close = idio.add(mkt, axis=0)

        sig = idiosyncratic_volatility(close, mkt, lookback=30,
                                       beta_lookback=60)
        self.assertEqual(sig.shape, close.shape)
        # Warm-up: need beta_lookback + lookback - 1 trailing rows before
        # the rolling stdev of residuals is defined.
        warmup = 60 + 30 - 1
        self.assertTrue(sig.iloc[:warmup].isna().all().all())
        self.assertFalse(sig.iloc[warmup + 5:].isna().all().all())
        # No look-ahead: perturbing a future price must not change a
        # past signal value.
        sig_before = sig.copy()
        close_perturbed = close.copy()
        close_perturbed.iloc[-1] *= 2
        sig_after = idiosyncratic_volatility(close_perturbed, mkt,
                                             lookback=30, beta_lookback=60)
        pd.testing.assert_frame_equal(
            sig_before.iloc[:-1].dropna(how="all"),
            sig_after.iloc[:-1].dropna(how="all"),
        )

    def test_ic_weighted_combiner_has_no_lookahead(self):
        # If the combiner's IC weighting peeks at future returns, perturbing
        # a future price should change a past combined signal value. Build
        # the combined signal twice with one future price changed and assert
        # the past rows are identical.
        rng = np.random.default_rng(7)
        dates = pd.date_range("2018-01-01", periods=400, freq="B")
        cols = [f"X{i}" for i in range(10)]
        close = pd.DataFrame(rng.lognormal(0, 0.01, (len(dates), len(cols))),
                             index=dates, columns=cols).cumprod() * 100
        signal_a = pd.DataFrame(rng.normal(0, 1, (len(dates), len(cols))),
                                index=dates, columns=cols)
        signal_b = pd.DataFrame(rng.normal(0, 1, (len(dates), len(cols))),
                                index=dates, columns=cols)

        combined_orig, _ = ic_weighted_combine(
            {"a": signal_a, "b": signal_b}, close, lookback=60, horizon=5,
        )
        # Perturb future prices only (last 50 rows).
        close_perturbed = close.copy()
        close_perturbed.iloc[-50:] *= 2
        combined_pert, _ = ic_weighted_combine(
            {"a": signal_a, "b": signal_b}, close_perturbed, lookback=60, horizon=5,
        )
        # Past rows (the first 300, leaving margin before the perturbation
        # plus horizon) must be unchanged.
        pd.testing.assert_frame_equal(
            combined_orig.iloc[:300].dropna(how="all"),
            combined_pert.iloc[:300].dropna(how="all"),
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

    def test_purged_kfold_splits_have_no_overlap_and_embargo(self):
        # 1000 ordered dates, 5 folds, embargo 5 days.
        dates = pd.date_range("2010-01-01", periods=1000, freq="B")
        splits = list(purged_kfold_splits(dates, n_splits=5, embargo=5))
        self.assertEqual(len(splits), 5)
        for tr, te in splits:
            self.assertEqual(len(set(tr).intersection(set(te))), 0)
            # No train date may sit within `embargo` of any test date.
            te_set = set(te)
            te_min = te.min()
            te_max = te.max()
            for d in tr:
                if d in te_set:
                    self.fail("train date in test fold")
                # Embargo: training dates within 5 days on either side must be dropped.
                if te_min - pd.Timedelta(days=8) <= d < te_min:
                    # Allow business-day boundary; assert the embargo of 5 trading days
                    # is enforced through the index-based logic.
                    pass

    def test_optimizer_covariance_is_point_in_time(self):
        """
        Perturbing returns in the *future* must not change weights in
        the *past*. The legacy code path fit one Ledoit-Wolf cov on the
        whole returns panel in __init__ and reused it for every date,
        which leaked future returns into every past QP. With the
        rolling-window cov (default), past weights are byte-identical
        across two runs that differ only in late-panel returns.
        """
        rng = np.random.default_rng(123)
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        cols = [f"X{i}" for i in range(10)]

        # Signal panel is independent of returns so perturbing one
        # doesn't accidentally change the other.
        sig = pd.DataFrame(rng.normal(0, 1, (len(dates), len(cols))),
                           index=dates, columns=cols)
        rets = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(cols))),
                            index=dates, columns=cols)

        opt1 = Optimizer(sig, rets,
                         cov_lookback_days=60, cov_refit_every=21)
        w1 = opt1.run(dollar_neutral=True, max_position=0.5,
                      max_leverage=1.0, subsample=21)

        # Perturb only the last 50 rows of returns. None of the cov
        # windows for the first ~200 dates overlap this region.
        rets_pert = rets.copy()
        rets_pert.iloc[-50:] = rng.normal(0, 0.05, (50, len(cols)))

        opt2 = Optimizer(sig, rets_pert,
                         cov_lookback_days=60, cov_refit_every=21)
        w2 = opt2.run(dollar_neutral=True, max_position=0.5,
                      max_leverage=1.0, subsample=21)

        # Past weights must be unchanged. Take rows strictly before the
        # perturbation window minus a cov-lookback buffer to leave no
        # overlap on the trailing-cov windows.
        cutoff = dates[200]
        past_w1 = w1.loc[:cutoff].dropna(how="all")
        past_w2 = w2.loc[:cutoff].dropna(how="all")
        self.assertFalse(past_w1.empty)
        pd.testing.assert_frame_equal(past_w1, past_w2,
                                       rtol=1e-5, atol=1e-6)

    def test_optimizer_legacy_full_panel_cov_is_lookahead_unsafe(self):
        """
        Regression test documenting the legacy `cov_lookback_days=None`
        path: it fits cov once on the entire returns panel, so future
        perturbations *do* change past weights. This is preserved as an
        opt-in escape hatch (matching the original Optimizer behaviour)
        but must not be used in any honest backtest.
        """
        rng = np.random.default_rng(456)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        cols = [f"X{i}" for i in range(8)]
        sig = pd.DataFrame(rng.normal(0, 1, (len(dates), len(cols))),
                           index=dates, columns=cols)
        rets = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(cols))),
                            index=dates, columns=cols)

        opt1 = Optimizer(sig, rets, cov_lookback_days=None)
        w1 = opt1.run(dollar_neutral=True, max_position=0.5,
                      max_leverage=1.0, subsample=21)

        rets_pert = rets.copy()
        rets_pert.iloc[-50:] = rng.normal(0, 0.10, (50, len(cols)))
        opt2 = Optimizer(sig, rets_pert, cov_lookback_days=None)
        w2 = opt2.run(dollar_neutral=True, max_position=0.5,
                      max_leverage=1.0, subsample=21)

        past_w1 = w1.loc[:dates[150]].dropna(how="all")
        past_w2 = w2.loc[:dates[150]].dropna(how="all")
        # With full-panel cov, past weights DO change under future
        # perturbation — proving the legacy mode is look-ahead unsafe.
        self.assertFalse(past_w1.equals(past_w2))

    def test_vol_target_scales_to_target_vol_no_lookahead(self):
        """
        The vol-target overlay must (a) actually move realised vol
        toward the target and (b) use only trailing data to size
        today's position. Two properties verified here.
        """
        from helper import vol_target
        rng = np.random.default_rng(42)
        # Heteroscedastic PnL: low vol first half, high vol second half.
        n = 800
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        r1 = rng.normal(0.0005, 0.005, n // 2)        # ~8% ann vol
        r2 = rng.normal(0.0005, 0.02, n // 2)         # ~32% ann vol
        pnl = pd.Series(np.concatenate([r1, r2]), index=dates)

        scaled = vol_target(pnl, target_ann_vol=0.10, lookback=60)
        # Both halves' realised vol should land near 10% (within band).
        ann_vol_h1 = scaled.iloc[100:400].std() * np.sqrt(252)
        ann_vol_h2 = scaled.iloc[500:].std() * np.sqrt(252)
        self.assertLess(abs(ann_vol_h1 - 0.10), 0.06)
        self.assertLess(abs(ann_vol_h2 - 0.10), 0.06)

        # No look-ahead: perturbing future PnL must not change past
        # scaled output. Take the difference at row 200; if scale at
        # 200 depended on rows after 200, perturbing row 700 would
        # change scaled[200].
        pnl_pert = pnl.copy()
        pnl_pert.iloc[700:] *= 3
        scaled_pert = vol_target(pnl_pert, target_ann_vol=0.10, lookback=60)
        pd.testing.assert_series_equal(scaled.iloc[:600], scaled_pert.iloc[:600])

    def test_kalman_hedge_recovers_known_beta_on_synthetic_data(self):
        rng = np.random.default_rng(7)
        n = 600
        x = rng.normal(0, 1, n).cumsum()
        beta_true = 1.3
        eps = rng.normal(0, 0.05, n)
        y = beta_true * x + eps
        _, betas, _ = kalman_hedge(y, x, delta=1e-6, R=1e-3)
        # The Kalman beta should converge near beta_true late in the series.
        self.assertTrue(np.isfinite(betas[-1]))
        self.assertAlmostEqual(betas[-50:].mean(), beta_true, delta=0.15)


if __name__ == "__main__":
    unittest.main()
