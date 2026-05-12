# Cross-Sectional Signal Research

End-to-end quant research and execution stack: **data → signals →
portfolio construction → multi-strategy combination → live execution →
audit**. Six factor / ML / pairs strategies evaluated on the S&P 500
and top 30 crypto pairs under one rigorous framework (point-in-time
membership, HAC standard errors, Deflated Sharpe, 5 bps/side costs,
two-day execution lag), then combined into a single fund-of-strategies
book and wired into a broker-agnostic order-management layer with
pre-trade risk gates and an append-only audit log.

The headline multi-strategy combination achieves **Sharpe 0.73 OOS
net of costs**, with **positive skew (+0.15)**, near-zero SPY beta
(-0.04), and -13.6% max drawdown over the 2021-2026 out-of-sample
window. The Sharpe lift comes from inverse-vol weighting across
uncorrelated equity / crypto / pairs sleeves and a 10% vol-target
overlay — not from search or signal tuning. No single strategy clears
family-wise Bonferroni significance; the diversified combination
clears `|t| > 1.7` HAC against SPY but not `|t| > 2`, and the README
is explicit about that.

## Headline result

Across all strategies, applied with consistent transaction costs (5 bps
per side on equity, 10 bps per side on crypto), HAC-corrected
(Newey-West, 5 lag) standard errors on the alpha regression, and the
realistic two-day execution lag for equities:

| Notebook | Strategy | Sharpe | Alpha (ann) | Alpha t-stat (HAC) | DSR | Passes \|t\|>2 |
|---|---|---:|---:|---:|---:|:---:|
| 01 | Rank-weighted mean rev (IS) | 0.17 | +0.28% | 0.21 | – | no |
| 01 | Optimized mean rev (IS) | 0.12 | -1.54% | -0.40 | ~0 | no |
| 01 | Optimized mean rev (OOS) | -0.45 | -11.13% | -1.80 | ~0 | no |
| 02 | Optimized momentum (IS) | -0.30 | -3.65% | -0.60 | – | no |
| 02 | Optimized momentum (OOS) | 0.09 | +2.55% | 0.27 | – | no |
| 03 | Optimized low-vol (IS) | -0.68 | -7.04% | -1.65 | – | no |
| 03 | Optimized low-vol (OOS) | -0.52 | +2.56% | 0.34 | – | no |
| 04 | Combined equity (IS, IC-weighted) | -0.30 | -2.02% | -0.36 | – | no |
| 04 | Combined equity (OOS) | -0.17 | +1.48% | 0.16 | – | no |
| 07 | Standalone crypto momentum (opt) | 0.87 | +34.1% | 1.51 | – | borderline |
| **07** | **Standalone crypto VolMom (opt)** | **0.83** | **+51.1%** | **2.15** | – | **yes** |
| 07 | Crypto combined (IC-weighted, IS 2019-23) | 0.15 | +5.3% | 0.26 | 0.16 | no |
| 07 | Crypto combined (IC-weighted, OOS 2024+) | -0.50 | -20.2% | -0.74 | 0.03 | no |
| **08** | **ML decile spread (IS, top10/bot10, 5d rebal)** | **0.99** | **+28.94%** | **2.47** | 0.70 | **yes** |
| 08 | ML decile spread (OOS) | -0.01 | -0.86% | -0.06 | 0.03 | no |
| 08 | ML optimized portfolio (full) | 0.26 | +1.46% | 0.45 | – | no |
| 09 | Pairs (selection 2010-2017 diagnostic) | 1.68 | – | 4.53 | – | n/a (in-sample) |
| 09 | Pairs IS 2018-2020 (static OLS hedge) | 0.41 | +0.7% | 0.20 | – | no |
| 09 | Pairs OOS 2021-2026 (static OLS hedge) | 0.64 | +5.4% | 1.12 | – | no |

*Numbers above were regenerated after the `Optimizer` covariance look-ahead
fix (see "Realism corrections"). The fix has near-zero impact on the
headline conclusions: low-vol IS HAC t moved from -1.91 → -1.65, crypto
vol-mom moved from 2.13 → 2.15, everything else is unchanged to two
significant figures. The cov bug existed; its numerical impact on these
particular factors was small. The fix is still required for correctness
and for any future strategy whose performance is more cov-sensitive.*

Two strategies clear the per-test `|t| > 2` HAC bar once realistic
costs and standard errors are applied: standalone optimised
volume-adjusted momentum on crypto, and the in-sample epoch of the
LightGBM decile spread on the top-300 most liquid S&P 500 names. Both
fail to reproduce their edge out-of-sample — exactly the documented
factor-decay pattern of the post-2020 era.

**Multiple-comparison caveat.** Across notebooks 01-09 the project
evaluated roughly 50 strategy configurations (4 lookback grids × 3
linear signals × IS/OOS × quick/optimised, plus crypto, ML and pairs
variants). Bonferroni-corrected at family-wise 5%, the right bar is
`|t| > 3.0`, not 2.0. **Neither of the two strategies above clears
that bar.** The honest reading is: no result in this repo survives
a family-wise multiple-testing correction. The `|t| > 2` column is
presented because it is the conventional per-test threshold, not
because it implies discovery.

**Crypto survivorship caveat.** The crypto vol-mom HAC t=2.13 is the
single positive headline. It is computed on the top 30 USDT pairs by
*today's* market cap — LUNA, FTT, UST and other 2021-22 blow-ups are
structurally excluded from the long leg. Re-running on a 2019-vintage
candidate set is on the v2 list; until then the t-stat should be read
as upper-bounded.

An earlier version of this table reported the IC-weighted combined
portfolio at HAC t = 2.55, but that result was an artifact of a
look-ahead bug in `ic_weighted_combine` — the IC weighting at
horizon = 14 was being shifted by only one day, which silently used
13 days of future returns in the weighting decision. Once the shift
is corrected to the full horizon, the combined portfolio drops to
Sharpe ≈ 0 in-sample and goes negative out-of-sample. The bug, the
catch, and the demotion of the headline result are documented in the
git history (commit fixing `ic_weighted_combine`).

A second look-ahead bug, in the `Optimizer` covariance estimation,
was caught during the execution-layer build: cov was fit once on the
full returns panel passed in `__init__`, then reused for every date
in `run()`, so each QP saw future returns. The fix is a rolling
trailing-window cov keyed off each loop date (default 252 trading
days, refit monthly). Test `test_optimizer_covariance_is_point_in_time`
verifies that perturbing late-panel returns no longer alters past
weights; a companion regression test (`..._legacy_full_panel_cov_is_lookahead_unsafe`)
asserts the legacy mode fails the same check. Notebooks 01-04, 07,
and 08 were re-run under the fix; numbers in the table above reflect
the corrected optimizer. Net change: low-vol IS HAC t moved -1.91 →
-1.65, crypto vol-mom 2.13 → 2.15, everything else within 0.01-0.03
of pre-fix values. The bug was real; its impact on these specific
factors was small because LW shrinkage on a long-history panel is
dominated by the diagonal, which is fairly time-stable on US
equities. Other strategies (especially anything with concentrated
positions or short-history universes) would have shown bigger gaps.

The classical equity factors do not pass at any significance bar.
This is the project's main result.

What the equity notebooks are good for is calibration: they confirm
that the framework is honest. A pipeline that runs survivorship-corrected
data, point-in-time membership, realistic execution lag, asymmetric
costs, Deflated Sharpe, and HAC SEs and reports `t = 0.21` for the
strategy that most public-data backtests would report as `t = 4.53` (the
no-cost version) is a pipeline you can trust when it does report a
positive result.

## Project goal

To walk through the full quant research workflow end-to-end on a public
dataset: data preparation, signal construction, information-coefficient
analysis, parameter sensitivity, in-sample / out-of-sample validation,
mean-variance portfolio optimization with realistic frictions, and
portfolio-level risk diagnostics, with statistical-significance and
multiple-testing corrections applied consistently.

## Repository layout

```
.
├── download_sp500.py         # Reconstructs point-in-time membership, fetches equity prices
├── download_crypto.py        # Fetches daily OHLCV for top 30 USDT pairs from Binance.US
├── signals.py                # Signal definitions (mean reversion, momentum, low vol, residual momentum)
├── helper.py                 # Optimizer, IC, stats, VaR, DSR, IC-weighted combiner
├── combine_weights.py        # Research-to-execution bridge: combined target weights + signal decay
├── execution.py              # Broker interface, PaperBroker, OMS, execution algos, audit log
├── risk.py                   # Pre-trade risk gates (leverage, position cap, ADV, drawdown kill)
├── price_feed.py             # Live price fetch (yfinance), feed-health check, reconciliation
├── runner.py                 # Daily rebalance entry point (cron target)
├── config.yaml               # Runtime config: broker, risk limits, execution algo, paths
├── research/
│   ├── 01_mean_reversion.ipynb
│   ├── 02_momentum.ipynb
│   ├── 03_low_volatility.ipynb
│   ├── 04_portfolio.ipynb          # Equities: combined portfolio with IC-weighted combiner
│   ├── 05_feature_selection.ipynb  # Spike-and-slab Bayesian variable selection
│   ├── 06_monte_carlo.ipynb        # HMM regime-conditional Monte Carlo
│   ├── 07_crypto.ipynb             # Same framework applied to top 30 crypto pairs
│   ├── 08_ml_alpha.ipynb           # LightGBM with purged k-fold + walk-forward retraining
│   └── 09_pairs.ipynb              # Cointegration-based pairs trading + Kalman comparison
├── data/                     # Generated by the download scripts (gitignored)
├── tests/                    # test_core.py (research), test_execution.py (execution)
├── requirements.txt
└── README.md
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_sp500.py        # populates data/ with equity prices and membership
python download_crypto.py       # populates data/crypto.parquet (optional, for notebook 07)
python -m unittest discover -s tests
```

The ML alpha notebook (`08_ml_alpha.ipynb`) additionally needs LightGBM
(`pip install lightgbm`; on macOS also `brew install libomp` for the
OpenMP runtime). Both are listed in `requirements.txt`.

Then start Jupyter from the `research/` directory and run the notebooks
in order:

```
cd research
jupyter notebook
```

The first three notebooks are independent; the portfolio and Monte Carlo
notebooks depend on intermediate parquet files written by the first three.
The sample window is fixed in `download_sp500.py` (`2005-01-01` through
`2026-04-01`) so notebook outputs remain reproducible.

## The three signals

| Notebook | Signal | Hypothesis |
|----------|--------|------------|
| `01_mean_reversion.ipynb` | Negated rolling z-score of price | Short-horizon overreaction reverts within days to weeks |
| `02_momentum.ipynb` | 12-1 month total return | Slow news diffusion produces persistent trends at the 1-12 month horizon |
| `03_low_volatility.ipynb` | Negated rolling realized volatility | Leverage constraints and lottery preferences cause low-vol stocks to be underpriced relative to CAPM |

Each notebook follows the same template: hypothesis, data, signal
construction, distribution check, IC at multiple horizons, parameter grid,
rolling IC, in-sample rank-weighted backtest, transaction-cost sensitivity,
optimized backtest, out-of-sample validation, annual returns. The mean
reversion notebook additionally demonstrates factor neutralization against
rolling SPY beta.

Each signal notebook also includes a return-distribution and tail-risk
section (histogram vs Gaussian, QQ plot, historical and parametric VaR
and CVaR at 1% and 5%) on the optimised in-sample and out-of-sample PnL.
The mean reversion notebook additionally runs a Sharpe-based grid search
alongside the IC heatmap, since IC and net Sharpe can disagree once
transaction costs and covariance structure enter.

## The portfolio notebook

Combines the three signals via cross-sectional z-score averaging and
applies the full evaluation suite at the portfolio level:

- Cross-sectional rank correlation between signals
- Time-series correlation between single-signal PnL streams
- Combined-signal optimization (dollar-neutral, 2% position cap, 5 bps cost)
- Return distribution: histogram vs Gaussian, skew, kurtosis, QQ plot
- Historical and parametric VaR / CVaR at 1% and 5%
- Drawdown depth and duration distribution
- Rolling 1-year Sharpe and rolling correlation to SPY
- Walk-forward evaluation across rolling 5-year-train / 1-year-test windows

## Feature selection (notebook 05)

A standalone notebook that runs Bayesian variable selection on a panel
of eight candidate features (mean reversion, momentum, and low-volatility
at multiple lookbacks). The Gibbs sampler is hand-implemented in NumPy
using the Stochastic Search Variable Selection (SSVS) form of
spike-and-slab. Outputs are posterior inclusion probabilities for each
feature and the posterior coefficient distributions for the selected
ones, which gives an uncertainty-aware view of which lookbacks earn
their place after accounting for everything else in the model.

## Crypto (notebook 07)

The same signal and portfolio code applied to the top 30 USDT pairs on
Binance.US (2019-09 to present). What changes versus equities:

- **No index membership concept.** The universe is a curated list of
  liquid pairs; per-date masking drops names that have no data yet.
- **365 days/year.** `stats(periods_per_year=365)`.
- **`exec_lag=1`.** Daily-close signals can be executed at the next
  close with seconds of latency on a 24/7 market.
- **10 bps per side.** Reflects Binance.US taker fees.
- **Dollar-neutral via perp futures (implicit).** Spot shorting is
  limited; the construction assumes perpetual-futures execution.
  Funding cost is acknowledged but not modelled.

The five candidate signals are mean reversion, momentum, low volatility,
market-residual momentum, and volume-adjusted momentum (`mean_return *
sqrt(volume / rolling_mean_volume)`, ported from an earlier crypto-momentum
study). Each is run through the per-signal optimizer and through an
IC-weighted combiner.

**Headline result**: standalone optimised volume-adjusted momentum
reaches Sharpe 0.83 with **HAC alpha t-stat 2.13** vs BTC. That is the
only crypto strategy that clears the `|t| > 2` bar after the look-ahead
audit. Standalone optimised cross-sectional momentum reaches HAC
t = 1.56 (borderline). The IC-weighted combined portfolio reports
Sharpe ≈ 0 in-sample (t = 0.30) and Sharpe -0.50 out-of-sample (t = -0.71)
after the combiner shift bug was corrected; the previously-reported
combined Sharpe of 1.16 was driven by that bug.

Caveats specific to crypto: only six years of history concentrated
around the 2021-2022 boom-bust, narrow breadth (30 names vs 500),
funding-rate cost on the short leg is not modelled, and the pair list
is curated by today's market-cap rank so there is some universe
survivorship bias.

## Machine-learning alpha (notebook 08)

A non-linear cross-sectional alpha model. 16 features (the linear factor
signals from notebooks 01-03 at multiple lookbacks plus engineered
statistics: skewness, kurtosis, MAX/lottery, log-ADV, price-to-200-day-MA).
Target is the forward 10-day cross-sectional rank. Model is LightGBM.

Two correctness pieces specific to ML on financial panels:

- **Purged k-fold CV with embargo** (Lopez de Prado, *Advances in Financial
  Machine Learning*, ch. 7). The forward-10-day target labels overlap by
  construction with adjacent training rows; the purge step drops training
  rows whose label window overlaps the test fold, and the embargo (`HORIZON
  + 1` trading days on each side) handles serial correlation in the features.
- **Walk-forward retraining**. Every year, train on the prior 5 years,
  embargo 11 days, predict the next year. The full panel of OOS predictions
  spans 2010-2026 and is genuinely out-of-sample relative to the model that
  produced each prediction.

Reported on the same realistic-cost framework as the linear factors (5 bps
per-side, 2-day execution lag, point-in-time membership, HAC alpha SEs):

- **Decile spread, IS (predictions for dates ≤ 2020-12-31)**: Sharpe 0.99,
  alpha **t-stat 2.47** ✓ clears the significance bar.
- **Decile spread, OOS (predictions for 2021-2026)**: Sharpe ≈ 0, t-stat
  -0.06.
- Optimized (cvxpy mean-variance with 5 bps turnover penalty, 2% per-name
  cap, dollar-neutral): Sharpe 0.26 full-sample, near zero OOS.

The pattern *the IS Sharpe survives HAC SEs and the DSR adjustment for 12
LightGBM-hyperparameter trials (`DSR_IS = 0.70`), but the OOS Sharpe is
zero* is exactly what an honest pipeline should report when classical
cross-sectional factor performance has decayed. A pipeline that reported
Sharpe > 1 in both halves would deserve suspicion.

## Pairs trading (notebook 09)

Statistical-arbitrage on the S&P 500. Selection (2010-2017) within each
GICS sub-industry: any two names with correlation > 0.65, Engle-Granger
cointegration p < 0.05 on log prices, and Ornstein-Uhlenbeck half-life
of the residual spread in `[5, 60]` trading days. A second tier admits
same-sector pairs with correlation > 0.80. 33 pairs survive.

Trading (2018-2026): static OLS hedge ratio fitted on the selection
window, 60-day rolling z-score on the spread, entry at `|z| > 2`, exit
at `z = 0`, hard stop at `|z| > 3.5`. Each pair contributes a gross-1
position; the portfolio normalises each pair's PnL by its 120-day
calibration vol so risk contributions are equalised, then scales the
aggregate to a 10% annualised vol target. 5 bps per side, per leg,
1-day execution lag, point-in-time membership masking (a pair is not
traded while either leg has been removed from the index).

Results (all forward of the selection window, after costs):

- **Selection window (2010-2017, in-sample by construction, diagnostic
  only)**: Sharpe 1.68, HAC alpha t-stat 4.53.
- **IS (2018-2020)**: Sharpe 0.41.
- **OOS (2021-2026)**: Sharpe 0.64, alpha t-stat 1.12.
- **Per-year Sharpe (2018-2026)**: -0.03, 1.34, 0.36, 2.07, -0.23, 0.63,
  0.69, 0.43, 1.83.

These numbers — positive but well below the Sharpe-2-after-costs that
Gatev, Goetzmann & Rouwenhorst (2006) reported on 1962-2002 data — match
the post-2010 decay surveyed by Krauss et al. (2017). The selection-window
diagnostic confirms the pairs do mean-revert profitably *in-sample*; the
gap between that and the IS/OOS Sharpes is what overfitting in pair
selection looks like even with a stationarity test and economic prior.
A Kalman-filter variant of the hedge ratio is run as a comparison and
underperforms the static OLS hedge on OOS, as expected — small drift
parameters track noise rather than the underlying cointegration vector.

## Multi-strategy combiner (notebook 10)

The single-sleeve numbers are conservative. Each sleeve (equity factors,
crypto vol-mom, pairs) on its own does not clear the per-test bar, but
the cross-correlations are small (equity vs crypto -0.04, equity vs
pairs -0.23, crypto vs pairs -0.02) so a fund-of-strategies view
captures real diversification.

`multi_strategy.py` exposes three **fixed** combination rules — no
parameter search:

- **Equal-weight**: `w_i = 1/N`.
- **Inverse-vol (risk parity)**: `w_i ∝ 1/σ_i` on a trailing 60-day
  realised vol, weights `shift(1)` before applying.
- **Minimum-variance**: per-date QP `min w'Σw  s.t.  sum(w)=1, w>=0`
  on a 252-day rolling cov, weights `shift(1)` before applying.

Equal-weight gets dragged down by the high-kurtosis crypto sleeve
(kurt 191 on the combined PnL). Both vol-aware rules cut crypto exposure
and let pairs contribute more, producing well-behaved tails (kurt 2-3).

A 10% annualised vol-target overlay (`helper.vol_target`, 60-day
trailing realised vol, weights `shift(1)`, leverage capped at 5x) is
then applied to the inverse-vol combination. Vol-targeting helps
inverse-vol but not min-variance — min-var already produces stable
variance by construction; vol-target on top is redundant. Inverse-vol
uses static weighting, so the time-varying overlay adds a real lift.

| Configuration | Sharpe | Skew | Kurt | HAC alpha t (SPY) | Max DD | Ann vol |
|---|---:|---:|---:|---:|---:|---:|
| Equal-weight | 0.04 | -8.12 | +191 | 0.32 | -27.9% | 18.10% |
| Min-variance | 0.70 | -0.09 | +2.41 | 1.45 | -8.3% | 8.08% |
| Inverse-vol | 0.64 | +0.07 | +2.86 | 1.63 | -11.4% | 8.56% |
| **Inverse-vol + vol-target 10%** | **0.73** | **+0.15** | **+2.23** | **1.73** | **-13.6%** | **10.62%** |

The headline combined OOS Sharpe net of 5 bps/side costs is **0.73**
with **positive skew (+0.15)** and **near-zero SPY beta (-0.04)** — the
first positive-skew portfolio in this repo. It still does not clear
per-test `|t| > 2` (HAC alpha t = 1.73), and certainly not the
family-wise Bonferroni bar — but the result is honest, diversified,
and stable enough to size a real (if small) book against.

The honest read on the Sharpe lift: most of it comes from
diversification across uncorrelated sleeves (equity ⊥ crypto ⊥ pairs),
not from new alpha. The vol-target overlay adds ~0.09 to Sharpe on
top by reallocating risk across time. None of this is a search-found
result; the three combination rules are textbook and the overlay is
the standard 60-day inverse-vol scale.

## Execution layer

The execution layer is the bridge from research to a (paper) trading
account. Its purpose is to demonstrate that the research-to-live pipeline
is in place and correct, not to claim profitability. The headline-table
results above show that the equity factors do not clear HAC `|t| > 2`
out-of-sample, so live-paper PnL on the combined equity book is expected
to be flat-to-negative net of costs. A pipeline that reports that
faithfully is more useful than one that overstates.

### What the execution layer does

```
combine_weights.latest_target_weights(asof)   # research-side, single weight vector
        │
        ▼
runner.run                                     # daily entry point, cron target
        │
        ├─ price_feed.fetch_latest_close       # live prices via yfinance
        ├─ price_feed.feed_health              # hard-fail on stale feed
        ├─ price_feed.reconcile                # broker vs internal book (positions + cash)
        │
        ├─ risk.PreTradeRiskGate.check         # gross/net, position cap, ADV, drawdown
        │
        ├─ execution.weights_to_target_shares  # weights × equity / price → shares
        ├─ execution.build_order_list          # diff vs current positions
        ├─ execution.execute_market/twap/vwap  # parent/child slicing
        │
        ├─ execution.implementation_shortfall  # bps gap from decision price
        └─ execution.AuditLog                  # append-only JSONL of every event
```

### Modules

- **`execution.py`** — the broker-agnostic OMS. Defines an abstract
  `Broker`, a `PaperBroker` simulator (deterministic fills, configurable
  slippage and commission), and an `AlpacaBroker` stub showing the exact
  shape of the live integration. Each `Order` carries an idempotent
  `client_id` (re-submitting the same id returns the prior broker id
  with no second fill, which is how a retry survives a network blip),
  a status that transitions through a defined state machine
  (`PENDING → SUBMITTED → PARTIALLY_FILLED/FILLED/CANCELED/REJECTED`),
  and an audit trail. The `PaperBroker` bookkeeping invariant — that
  cash + position market value + realised PnL stays equal to starting
  cash minus commissions across any sequence of orders — is tested
  end-to-end in `test_execution.py`, including the four cross-zero cases
  (long-to-flat, long-to-short, short-to-long, short-to-flat) where a
  miscoded realised-PnL sign is the most common bug.

- **`risk.py`** — `PreTradeRiskGate` runs every order basket through:
  drawdown kill switch (equity vs high-water mark), gross/net leverage,
  per-name notional cap, max position count, ADV-participation cap on
  every order, total daily turnover cap, and an optional blocked-symbol
  list. Any single breach rejects the *entire* basket — partial
  rebalances leave the book in a state neither the model nor the human
  asked for. Limits live in `config.yaml`, not in code.

- **`price_feed.py`** — pulls the latest adjusted close per symbol from
  yfinance and computes a feed-health record (last-bar age vs SLA).
  A stale feed hard-fails the rebalance; the runner exits non-zero so
  the cron job alerts. Importantly, signal staleness is *not* handled
  by silently shrinking positions via alpha decay — that conflates "the
  market hasn't moved in a way the model finds interesting" with "our
  data pipeline is dead", and the latter is the one that costs money.
  The module also exposes `reconcile()` which compares prices and
  positions between the internal book and the broker; persistent drift
  is the single most common silent failure in real systems.

- **`combine_weights.py`** — the research-to-execution bridge. Loads
  the price panel, computes the three linear factor signals from
  `signals.py`, runs `helper.ic_weighted_combine` to get a single
  signal, then feeds it into `helper.Optimizer` with the same
  parameters as notebook 04 (dollar-neutral, 2% per-name cap, 5 bps
  cost penalty). Exposes `latest_target_weights(asof)` for the daily
  runner. Also exposes `apply_signal_decay(signal, age_bars, half_life)`
  for the *signal-side* decay PMs ask for: weights cannot be decayed
  post-optimisation without violating the constraints the optimiser
  solved under.

- **`runner.py`** — the cron target. Loads config, builds today's
  weights, fetches prices, runs the feed-health check, builds the risk
  gate from the most recent 20-day ADV, calls `execution.rebalance`,
  and appends everything to `logs/YYYY-MM-DD/journal.jsonl`. Supports
  `--dry-run` (plan, don't submit) and `--asof YYYY-MM-DD` (backfill a
  prior day).

### Why pairs and crypto are not in the live runner

- **Pairs (notebook 09)** is structurally different from cross-sectional
  rebalancing: positions are opened on z-score crossings of a
  cointegrated spread, sized per-pair, and held until exit threshold.
  The current `Broker` and `rebalance` abstractions are date-driven, not
  event-driven, and bolting pairs onto the same loop would either
  corrupt the design or duplicate it badly. A pairs-execution module
  with stateful per-pair entry/exit logic is a clean v2.
- **Crypto (notebook 07)** uses Binance.US data and trades on a 24/7
  market with different cost and execution conventions; co-mingling it
  into the same Alpaca-targeted weight vector would create more bugs
  than it would save lines. A separate `runner_crypto.py` targeting the
  Binance.US testnet is the right separation if it gets added.

### Broker choice and credentials

The default broker in `config.yaml` is `paper` — the in-process
`PaperBroker` simulator. It needs no signup, no API keys, no KYC,
and runs entirely offline (yfinance is the only external call). To
swap in Alpaca paper trading, install `alpaca-py`, populate `.env`
with `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`, uncomment the body
of `AlpacaBroker` in `execution.py`, and change `config.yaml: broker.type`
to `alpaca`. The same `runner.py` runs unchanged — the `Broker`
interface is the point of decoupling.

### How to run it

```
# One-off: rebuild the combined target-weights panel
python combine_weights.py

# Daily rebalance (default config, paper broker)
python runner.py

# Plan only, do not submit
python runner.py --dry-run

# Backfill a specific date
python runner.py --asof 2026-03-28

# Run the full test suite (53 tests: 17 research helpers + 36 execution)
python -m unittest discover -s tests
```

### Scope and honest framing

The execution layer covers the cross-sectional equity pipeline:
data → signals → portfolio construction → risk gates → order management
→ execution → reconciliation → audit. The combined equity book exists
to exercise that pipeline; the HAC alpha t-stat on the OOS PnL is 0.16,
so live-paper PnL is expected to be flat-to-negative net of costs.
That outcome is the point: the DSR, HAC SEs, and the look-ahead audit
on `ic_weighted_combine` (commit `f6814df`) show the framework reports
unfavourable results faithfully. A framework that does not is not a
framework worth trusting on a favourable one.

## Regime-conditional Monte Carlo (notebook 06)

Fits a two-state Gaussian HMM on SPY returns to identify bull and bear
regimes, then simulates forward paths by drawing regime sequences from
the fitted transition matrix and bootstrapping strategy returns
conditional on the simulated regime. Reports the empirical distribution
of one-year Sharpe, max drawdown, and terminal wealth across thousands
of paths, and the relationship between path Sharpe and realised
bear-state share. The bootstrap preserves within-regime fat tails that
a Gaussian simulation would lose; the HMM provides the regime structure
that a plain bootstrap would lose.

## Optimizer

The optimizer in `helper.py` solves a per-date mean-variance program with
cvxpy:

```
minimise   -alpha' w + w' Σ w + λ ||w - w_prev||_1
```

subject to optional constraints: dollar neutrality, per-name position cap,
gross leverage limit, long-only, and ex-post volatility scaling.
Covariance is estimated with Ledoit-Wolf shrinkage. The transaction-cost
penalty enters the objective directly so the optimizer trades off signal
strength against rebalancing cost rather than producing weights that are
unimplementable in practice.

## Realism corrections

The project addresses the realism issues that typically inflate paper
strategies, where possible without paid data.

- **Point-in-time index membership.** `download_sp500.py` parses the
  historical changes table on Wikipedia and reconstructs the set of
  S&P 500 members active on every trading day from 2005 to today. The
  union of all ever-members forms the download universe (about 830
  tickers, versus 500 in a current-only universe). The optimizer is
  passed a date-by-date membership mask so positions are only opened in
  names that were actually in the index on that date. This removes the
  bulk of survivorship bias and index-membership look-ahead.
- **Execution lag.** `port_ret` defaults to `exec_lag=2`: a signal
  computed at the close of day t is acted on at the close of t+1 and
  earns the t+1 to t+2 return. The previous one-day lag assumed costless
  execution at the same close used to compute the signal.
- **Asymmetric transaction costs.** `port_ret` charges separate bps on
  long-side and short-side turnover and a daily holding fee on the
  short book at a configurable annualised borrow rate.
- **Deflated Sharpe Ratio.** `helper.deflated_sharpe` and
  `probabilistic_sharpe` (Bailey and Lopez de Prado 2012, 2014) penalise
  the headline Sharpe for the parameter grids explored on the same data
  and for the skew and excess kurtosis of the realised return stream.
- **Implausible-return filter.** `clean_prices` masks out any price
  entry that would produce a same-day return greater than 100% in
  absolute value, catching adjusted-close errors that yfinance returns
  for some delisted or acquired tickers.
- **Point-in-time covariance in `Optimizer`.** Cov is refit on a
  trailing-window (default 252 trading days, refit monthly via
  `cov_refit_every=21`) inside `run()`, not once on the whole panel
  in `__init__`. The previous single-shot fit silently leaked future
  returns into every backtest date. `test_optimizer_covariance_is_point_in_time`
  asserts that perturbing future returns no longer changes past
  weights; the regression test `..._legacy_full_panel_cov_is_lookahead_unsafe`
  documents the broken mode the codebase used to ship with.

## Remaining limitations

- **Some delisted names lack yfinance data.** About 175 tickers that
  were S&P 500 members at some point in the sample period have no usable
  history through the free yfinance feed. The membership mask still
  records that they should have been members on the relevant dates,
  but the strategy cannot trade them. Residual survivorship bias is
  therefore reduced but not eliminated.
- **Single-factor risk model.** Ledoit-Wolf shrinkage is reasonable
  for a few hundred names but is not a multi-factor structural risk
  model. Position limits and dollar neutrality compensate in part.
- **No market-impact cost.** Transaction cost is per-bps of turnover
  with no square-root or non-linear dependence on order size.
- **Daily, close-to-close.** Intraday execution latency, opening
  auctions, and tick-level effects are out of scope.

## Why these signals

The three span distinct theoretical categories so the portfolio notebook
has something to combine:

- Mean reversion is a *contrarian* signal driven by short-horizon noise.
- Momentum is a *trend-following* signal driven by behavioural underreaction.
- Low volatility is a *risk-based* anomaly driven by frictions on the
  marginal investor.

They are textbook factors; the point of the project is the evaluation
template around them, not the signals themselves.
