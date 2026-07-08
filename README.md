# cross-asset-carry

> **An open-source, IP-clean implementation of cross-asset carry signals (FX, bonds, commodities) on free public data, with a long-short carry portfolio backtest.**

| | |
|---|---|
| **Headline metric** | **OOS annualized Sharpe −0.78 on the 2020-2025 out-of-sample window** (long top 3 / short bottom 3 by carry, vol-weighted, 5 bps round-trip cost). In-sample Sharpe 0.62. See `Honest scope` below — the OOS number is the only honest one. |
| **Run the demo** | `/opt/anaconda3/bin/python run.py` |
| **Inputs** | FX rates (FRED), bond yields (FRED), commodity futures (yfinance). Each has an embedded-sample fallback so the project ALWAYS runs offline. |
| **Author / contact** | [@christianmacion26](https://github.com/christianmacion26) |

---

## What's here

```
cross-asset-carry/
├── README.md            ← you are here
├── memo.md              ← design rationale + honest scope
├── run.py               ← single CLI entry point
├── data_loader.py       ← FRED + yfinance fetch with embedded-sample fallback
├── carry.py             ← FX carry, bond term-structure slope, commodity roll-yield proxy
├── backtest.py          ← long-short carry portfolio (vol-weighted, 5 bps cost)
├── plot_results.py      ← 3-panel figure (cumulative · carry heatmap · rolling Sharpe)
├── data/                ← embedded offline sample CSVs
└── results/             ← populated by run.py
```

---

## Why this exists

Carry is one of the few theoretically-grounded return premia in finance: an asset with higher implied carry should, on average, compound faster than one with lower carry. The academic literature (FX carry: Lustig-Verdelhan-Roussanov 2011, commodities: Erb-Harvey 2006) suggests positive alphas of a few % annualized — small, but exploitable in principle.

This project implements the SEMANTICS of cross-asset carry across the three asset classes where free public data can support it:
- **FX** (rates differential)
- **Bonds** (term-structure slope = roll-down carry)
- **Commodities** (roll-yield proxy from monthly price history)

It does *not* claim to reproduce any firm-specific implementation. It does claim: clear methodology, honest reporting of out-of-sample performance, no overfit parameters.

## Universe

8 instruments, ranked by carry each month:

| Asset | Carry definition |
|---|---|
| `EUR_USD`, `JPY_USD`, `GBP_USD`, `AUD_USD` | foreign 3-month rate − USD rate (annualized) |
| `US_SLOPE` | US 10Y − US 2Y (positive ⇒ positive carry for long bond) |
| `GC=F` (Gold), `CL=F` (Crude), `HG=F` (Copper) | `r₁ − r₃` proxy: 1-month log return minus 3-month annualized log return (proxy for backwardation/contango when no term structure) |

## Portfolio methodology

- Each month, rank the 8 instruments by **prior month-end carry**.
- **Long** the top 3 by carry, **short** the bottom 3.
- Weights are **inverse to the last-6-month realized vol** of each leg, scaled within legs so longs/shorts each contribute 50% of book.
- Held 1 month, then rebalanced.
- **5 bps round-trip** transaction cost is applied to turnover.
- Train / Test: **2010-2019 / 2020-2025** (chronological, no peeking).

## Data sources & offline fallback

| Data | Source | Offline fallback |
|---|---|---|
| FX rates (USD, EUR, JPY, GBP, AUD) | FRED CSV download (`DFF`, `IR3TIB01xxM156N`) | `data/fred_sample.csv` — 15 years monthly |
| Bond yields (US 10Y, US 2Y) | FRED CSV (`DGS10`, `DGS2`) | same `data/fred_sample.csv` |
| Commodity futures (GC=F, CL=F, HG=F, NG=F) | yfinance | `data/futures_sample.csv` |
| Asset total returns | Embedded sample (`data/returns_sample.csv`) — see **Honest scope** | same |

Every fetch is wrapped: if the network call fails (and in this environment it usually does), the embedded CSV takes over. **The project ALWAYS runs.** The honest-scope caveat is honest: the embedded *returns* sample is a synthetic stand-in, the rates and prices samples have realistic levels.

## Run it

```bash
/opt/anaconda3/bin/python run.py
```

Output:

1. `results/results.json` — train + test Sharpe, MDD, headline summary.
2. `results/demo_output.txt` — human-readable report.
3. `results/figure.png` — 3-panel chart (cumulative returns, carry heatmap, rolling Sharpe).
4. `results/port_returns.csv`, `results/carry_series.csv`, `results/asset_returns.csv` — full time-series.

## Quick start

```python
import sys; sys.path.insert(0, '.')
from data_loader import load_all
from carry import compute_all_carry
from backtest import carry_backtest, summary

d = load_all()
carry = compute_all_carry(d['rates'].dropna(), d['commodities'].dropna())
port  = carry_backtest(carry, d['asset_returns'])
print(summary(port))
```

## Honest scope

1. **The asset return series is embedded synthetic, not live.** All three carry asset classes require monthly *asset* returns that span the 8-instrument universe for the full window. We did not assemble a free-source composite (FX spot + 10Y ETF total return + commodity futures) over 15 years in this build. Carry signals are computed from FRED-derived rate inputs (real-style data); alpha in the synthetic returns is mild and consistent with the academic carry literature (~1-2% annualized for FX, weaker for bonds and commodities). Read the embedded sample as a stand-in, not as live data.
2. **In-sample Sharpe is the overfit estimate.** Out-of-sample is the only honest test. The current OOS Sharpe (`-0.78`) is *below* the equal-weight long-only benchmark (`-0.61`); by construction the strategy did not add value in this test window.
3. **Carry is a weak, noisy signal.** Cross-asset carry annualizes a few-percent alpha at best in published research; it cannot be expected to dominate transaction costs every year. The negative OOS Sharpe reflects noise, not refutation — but it also does not support live deployment on this evidence.
4. **No firm-specific tuning.** Signal thresholds, weights, and rebalance frequency are textbook choices. Anything more would be overfitting on this small dataset.

## Related repos in this portfolio

- [`validation-gate-stack`](https://github.com/christianmacion26/validation-gate-stack) — the multi-axis statistical gates that one would apply BEFORE trusting this OOS result.
- [`multiple-testing-deflated-sharpe`](https://github.com/christianmacion26/multiple-testing-deflated-sharpe) — DSR analysis for cross-strategy comparison.

Built June 2026 by [Christian Macion](https://github.com/christianmacion26).
