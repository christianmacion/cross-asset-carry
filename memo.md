# Memo — design rationale for `cross-asset-carry`

## What this is

An open-source implementation of cross-asset carry signals on free public data, plus a long-short carry portfolio backtest across FX, bonds, and commodities. The deliverable is methodology + an honest read of out-of-sample performance.

## What this is NOT

A live trading system or a backtest engine with proprietary data. The embedded *return* sample is a stand-in (we did not assemble a free multi-asset total-return series spanning 15 years in this build). Strategy thresholds are textbook, not tuned. The headline number is the OOS Sharpe — not the IS one.

## Why carry

Carry is one of the few theoretically-grounded return premia: assets with higher implied forward yield should, on average, outpace assets with lower yield (Fama-French-style term premium in rates, roll-yield in contango-prone commodity curves, interest-rate-differential convergence in FX). The academic literature (Burnside 2012; Lustig-Verdelhan-Roussanov 2011; Erb-Harvey 2006) places alpha at 1-3% annualized per asset class — small but exploitable in principle.

This project does not claim to add to that literature. It demonstrates **methodological literacy**: how to assemble carry signals across heterogeneous asset classes, run them through a clean vol-weighted long-short portfolio, and report OOS performance honestly.

## Carry definitions

| Asset class | Carry | Sign convention |
|---|---|---|
| FX (4 pairs) | foreign 3M − USD 3M | higher ⇒ long foreign / short USD |
| Bond | 10Y − 2Y slope | positive ⇒ long bond earns roll-down carry |
| Commodity | annualized 1M-vs-3M log-return differential | higher ⇒ backwardation-like signal |

The commodity roll-yield is an *approximation* via return differential — true roll yield needs a futures term structure which is not in the public Yahoo data. In trending commodity markets this differential tracks the term-structure proxy closely enough for a publicly-implementable signal; in sideways markets it is mostly noise.

## Portfolio methodology

- Monthly rebalance.
- Rank 8 instruments by prior-month carry.
- Long top 3, short bottom 3.
- Inverse-vol weighting on each leg (60-day realized vol / equivalent for monthly).
- 5 bps round-trip cost on turnover.
- Train 2010-2019 / Test 2020-2025 (chronological).

## A few choices worth flagging

- **Vol window of 6 monthly returns** — equivalent to ~60 trading days for daily data. Conservative for monthly series but well-defined.
- **Half book on each side** — equal gross exposure long/short, no leverage scaling.
- **Stationarity of carry** — carry is itself non-stationary in absolute units (a 4% rate differential looks small in 2010, huge in 2015). The strategy is implicitly long *high relative* carry, which is what the ranking does. We do not explicitly z-score before ranking because the cross-section ranks well in either case.
- **Top-K/Bottom-K = 3/3** — leaves 2 instruments un-touched (mid band). The 3/3 choice is the textbook long-short alternative-risk-premium formulation.

## Honest scope

- The **return series** for all 8 instruments is embedded synthetic. We did NOT assemble free public spot FX + 10Y ETF + commodity futures total returns into a single 15-year panel. Carry signals are computed from realistic embedded rate inputs; the *backtested returns* are a synthetic stand-in calibrated with mild carry-conditioned alpha consistent with academic literature. Treat the reported numbers as illustrative of *methodology*, not live performance.
- **OOS Sharpe is the only honest test.** Current OOS Sharpe is −0.78, slightly *below* the equal-weight benchmark (−0.61). The strategy did not add value in this window.
- **No firm-specific implementation.** Thresholds (top-3, vol window, 5 bps) are textbook. Live deployment would need: real multi-asset return series, walk-forward retraining, capacity-aware execution, and the standard regime / multiple-testing gates (`validation-gate-stack` in this portfolio).

## How the "ALWAYS runs" property is guaranteed

Three layers of fallbacks:
1. **FRED CSV download** for FX rates and bond yields. If it fails, fall back to `data/fred_sample.csv`.
2. **yfinance** for commodity futures. If it fails, fall back to `data/futures_sample.csv`.
3. **Embedded return sample** `data/returns_sample.csv` — there is no public-source fallback because no such composite exists in this build; the synthetic sample keeps the project offline-runnable.

The project runs `python run.py` cleanly with all three fallbacks active, which is the common case in this environment.

## What a hiring-manager-skim test should look like

1. `/opt/anaconda3/bin/python run.py` — runs in <30 s, exits 0.
2. Inspect `results/results.json` — IS Sharpe ~0.6, OOS Sharpe ~−0.8, honest verdict at the bottom.
3. Inspect `results/figure.png` — cumulative growth + carry heatmap + rolling Sharpe train-vs-test.
4. Read this memo to confirm the methodology is reasonable, the scope is bounded, and the OOS failure is reported without spin.

Author: Christian Macion. June 2026.
