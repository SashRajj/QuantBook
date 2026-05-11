import unittest

import numpy as np
import pandas as pd

import download_sp500
from helper import Optimizer, beta_to, ic, neutralize, port_ret, quick_weights, stats, var_cvar
from signals import low_volatility, mean_reversion, momentum


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
        weights = opt.run(dollar_neutral=True, max_position=0.25, max_leverage=1.0, subsample=10)

        self.assertFalse(weights.empty)
        self.assertLessEqual(weights.abs().max().max(), 0.2501)
        self.assertLess(weights.sum(axis=1).abs().max(), 1e-3)

    def test_factor_helpers_align_and_neutralize(self):
        benchmark = self.returns.mean(axis=1)
        betas = beta_to(self.returns, benchmark, lookback=20)
        signal = mean_reversion(self.close, lookback=10)
        neutral = neutralize(signal, betas)

        self.assertEqual(betas.shape, self.returns.shape)
        self.assertEqual(neutral.shape, signal.loc[betas.index].shape)

    def test_downloader_is_import_safe(self):
        self.assertTrue(hasattr(download_sp500, "main"))


if __name__ == "__main__":
    unittest.main()
