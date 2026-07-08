#!/usr/bin/env python3
"""
run.py — single CLI entry point.

Flow:
  1. Fetch FX rates, bond yields (FRED or sample).
  2. Fetch commodity futures (yfinance or sample).
  3. Load asset-level monthly returns (sample-based for offline reliability).
  4. Compute carry for each of the 8 instruments.
  5. Construct the long-short carry portfolio (vol-weighted, 5 bps cost).
  6. Train/test split: 2010-2020 train / 2020-2025 test (chronological, no peek).
  7. Write results/results.json with headline metrics.
  8. Invoke plot_results.py to render results/figure.png.

Usage:
    /opt/anaconda3/bin/python run.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_fred_rates, load_commodity_prices, load_asset_returns
from carry import compute_all_carry
from backtest import (
    carry_backtest, split_train_test, summary, benchmark_returns,
)

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def annualized_sharpe(returns: pd.Series, freq: int = 12) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(freq))


def main() -> int:
    print("=" * 78)
    print("  cross-asset-carry — backtest")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Step 1: load data (with embedded-sample fallbacks)
    # ------------------------------------------------------------------
    print("\n[1/6] Loading FX rates (FRED or sample)...", flush=True)
    rates = load_fred_rates()
    print(f"      rates: {rates.shape}, {rates.index.min().date()} → {rates.index.max().date()}")

    print("[2/6] Loading commodity futures (yfinance or sample)...", flush=True)
    prices = load_commodity_prices()
    print(f"      prices: {prices.shape}, {prices.index.min().date()} → {prices.index.max().date()}")

    print("[3/6] Loading asset returns sample (offline-safe)...", flush=True)
    rets = load_asset_returns()
    print(f"      rets: {rets.shape}, {rets.index.min().date()} → {rets.index.max().date()}")

    # ------------------------------------------------------------------
    # Step 2: compute carry
    # ------------------------------------------------------------------
    print("\n[4/6] Computing carry per instrument per month...", flush=True)
    carry = compute_all_carry(rates.dropna(), prices.dropna())
    carry = carry.dropna(how="any")
    print(f"      carry: {carry.shape}  ({list(carry.columns)})")

    # Align carry to returns index
    common = carry.index.intersection(rets.index)
    carry = carry.loc[common]
    rets = rets.loc[common]

    print(f"      aligned: {carry.shape[0]} months, "
          f"{carry.index.min().date()} → {carry.index.max().date()}")

    # ------------------------------------------------------------------
    # Step 3: backtest on the FULL aligned window
    # ------------------------------------------------------------------
    print("\n[5/6] Constructing long-short carry portfolio...", flush=True)
    port_ret = carry_backtest(carry, rets, start_year=2010, end_year=2025)
    print(f"      backtest: {port_ret.shape[0]} monthly observations")

    # Equal-weight benchmark (long-only, all instruments)
    bench = benchmark_returns(rets).rename("benchmark")
    # Intersect with port_ret dates
    bench = bench.loc[bench.index.intersection(port_ret.index)]

    # ------------------------------------------------------------------
    # Step 4: train / test split
    # ------------------------------------------------------------------
    SPLIT_YEAR = 2020  # chronological, no peeking
    train, test = split_train_test(port_ret, split_year=SPLIT_YEAR)

    train_summary = summary(train) if len(train) else {}
    test_summary = summary(test) if len(test) else {}

    print(f"\n      Train (pre-{SPLIT_YEAR}): "
          f"Sharpe {train_summary.get('annualized_sharpe', 0):.2f}, "
          f"return {train_summary.get('annualized_return', 0)*100:+.2f}%, "
          f"max DD {train_summary.get('max_drawdown', 0)*100:+.2f}%")
    print(f"      Test  ({SPLIT_YEAR}+):        "
          f"Sharpe {test_summary.get('annualized_sharpe', 0):.2f}, "
          f"return {test_summary.get('annualized_return', 0)*100:+.2f}%, "
          f"max DD {test_summary.get('max_drawdown', 0)*100:+.2f}%")

    # Also benchmark metrics (long-only equal weight)
    if len(train):
        bench_train = bench.loc[bench.index.year < SPLIT_YEAR]
        bench_train_sum = {
            "annualized_sharpe": round(annualized_sharpe(bench_train), 3),
            "annualized_return": round(float(bench_train.mean() * 12), 4),
            "max_drawdown": round(float(((1 + bench_train).cumprod() / (1 + bench_train).cumprod().cummax() - 1).min()), 4),
        }
    else:
        bench_train_sum = {}
    if len(test):
        bench_test = bench.loc[bench.index.year >= SPLIT_YEAR]
        bench_test_sum = {
            "annualized_sharpe": round(annualized_sharpe(bench_test), 3),
            "annualized_return": round(float(bench_test.mean() * 12), 4),
            "max_drawdown": round(float(((1 + bench_test).cumprod() / (1 + bench_test).cumprod().cummax() - 1).min()), 4),
        }
    else:
        bench_test_sum = {}

    # ------------------------------------------------------------------
    # Step 5: write results.json
    # ------------------------------------------------------------------
    is_oos_ratio = (test_summary.get("annualized_sharpe", 0) /
                    train_summary["annualized_sharpe"]
                    if train_summary.get("annualized_sharpe") else None)

    results = {
        "project": "cross-asset-carry",
        "description": (
            "Cross-asset carry long-short: long top 3 / short bottom 3 by carry, "
            "vol-weighted, 5 bps round-trip cost. 8 instruments: 4 FX, 1 bond slope, "
            "3 commodities (Gold, Crude, Copper)."
        ),
        "as_of": pd.Timestamp.now().isoformat(),
        "data_sources": {
            "fx_rates": "FRED CSV download with embedded sample fallback (data/fred_sample.csv)",
            "bond_yields": "FRED CSV download with embedded sample fallback (data/fred_sample.csv)",
            "commodity_futures": "yfinance with embedded sample fallback (data/futures_sample.csv)",
            "asset_returns": "Embedded sample: data/returns_sample.csv (FX spot + 10Y ETF total returns not assembled in this public-data build — see memo.md Honest scope)",
        },
        "config": {
            "top_k": 3,
            "bottom_k": 3,
            "weighting": "inverse 60-day realized vol",
            "transaction_cost_roundtrip_bps": 5,
            "train_years": "2010-2019",
            "test_years": "2020-2025",
        },
        "in_sample": {
            "window": train_summary.get("n_months", 0),
            "annualized_sharpe": train_summary.get("annualized_sharpe", 0),
            "annualized_return": train_summary.get("annualized_return", 0),
            "max_drawdown": train_summary.get("max_drawdown", 0),
            "win_rate": train_summary.get("win_rate", 0),
        },
        "out_of_sample": {
            "window": test_summary.get("n_months", 0),
            "annualized_sharpe": test_summary.get("annualized_sharpe", 0),
            "annualized_return": test_summary.get("annualized_return", 0),
            "max_drawdown": test_summary.get("max_drawdown", 0),
            "win_rate": test_summary.get("win_rate", 0),
        },
        "benchmark_long_only": {
            "train": bench_train_sum,
            "test": bench_test_sum,
        },
        "diagnostics": {
            "is_oos_ratio": (round(is_oos_ratio, 3) if is_oos_ratio is not None else None),
            "oos_positive": bool(test_summary.get("annualized_sharpe", 0) > 0),
            "oos_exceeds_bench": bool(
                test_summary.get("annualized_sharpe", 0) > bench_test_sum.get("annualized_sharpe", -99)
            ) if bench_test_sum else None,
        },
        "headline_metric": (
            f"Cross-asset carry portfolio: annualized Sharpe {train_summary.get('annualized_sharpe', 0):.2f} "
            f"in-sample · "
            f"OOS Sharpe {test_summary.get('annualized_sharpe', 0):.2f} · "
            f"max DD {train_summary.get('max_drawdown', 0)*100:+.1f}% "
            f"(long top 3 / short bottom 3 by carry, vol-weighted, 5 bps cost)."
        ),
    }

    out_json = RESULTS_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n      Wrote {out_json}")

    # Also persist the full backtest series for the plot
    port_ret.to_csv(RESULTS_DIR / "port_returns.csv")
    carry.to_csv(RESULTS_DIR / "carry_series.csv")
    rets.to_csv(RESULTS_DIR / "asset_returns.csv")

    # ------------------------------------------------------------------
    # Step 6: call plot_results.py
    # ------------------------------------------------------------------
    print("\n[6/6] Rendering results/figure.png...", flush=True)
    plot_script = ROOT / "plot_results.py"
    proc = subprocess.run(
        [sys.executable, str(plot_script)],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print("      ERROR: plot_results.py failed", file=sys.stderr)
        return proc.returncode

    # ------------------------------------------------------------------
    # Final report + demo_output.txt
    # ------------------------------------------------------------------
    demo = []
    demo.append("cross-asset-carry — backtest report")
    demo.append("=" * 78)
    demo.append(f"  Date run: {pd.Timestamp.now().isoformat()}")
    demo.append(f"  Window:   {carry.index.min().date()} → {carry.index.max().date()} "
                f"({len(port_ret)} monthly obs)")
    demo.append("")
    demo.append(f"  Instruments (8):")
    for c in carry.columns:
        demo.append(f"     - {c}")
    demo.append("")
    demo.append(f"  In-sample  (pre-{SPLIT_YEAR}): Sharpe {train_summary.get('annualized_sharpe', 0):.2f} · "
                f"Ann.Ret {train_summary.get('annualized_return', 0)*100:+.2f}% · "
                f"MaxDD {train_summary.get('max_drawdown', 0)*100:+.2f}% · "
                f"n = {train_summary.get('n_months', 0)}")
    demo.append(f"  Out-of-sample (post-{SPLIT_YEAR - 1}): Sharpe {test_summary.get('annualized_sharpe', 0):.2f} · "
                f"Ann.Ret {test_summary.get('annualized_return', 0)*100:+.2f}% · "
                f"MaxDD {test_summary.get('max_drawdown', 0)*100:+.2f}% · "
                f"n = {test_summary.get('n_months', 0)}")
    demo.append("")
    demo.append(f"  Benchmark long-only equal-weight:")
    demo.append(f"     train: Sharpe {bench_train_sum.get('annualized_sharpe', 0):.2f} · "
                f"Ann.Ret {bench_train_sum.get('annualized_return', 0)*100:+.2f}%")
    demo.append(f"     test : Sharpe {bench_test_sum.get('annualized_sharpe', 0):.2f} · "
                f"Ann.Ret {bench_test_sum.get('annualized_return', 0)*100:+.2f}%")
    demo.append("")
    demo.append(f"  Honest verdict:")
    is_oos = test_summary.get('annualized_sharpe', 0)
    bench_oos = bench_test_sum.get('annualized_sharpe', 0) if bench_test_sum else 0
    if is_oos > 0 and is_oos > bench_oos:
        demo.append(f"     OOS Sharpe {is_oos:.2f} > benchmark {bench_oos:.2f} — marginal edge preserved.")
    elif is_oos > 0:
        demo.append(f"     OOS Sharpe {is_oos:.2f} but below benchmark {bench_oos:.2f} — no added edge.")
    else:
        demo.append(f"     OOS Sharpe {is_oos:.2f} ≤ 0 — no edge in out-of-sample test.")
    demo_text = "\n".join(demo)
    print("\n" + demo_text)
    (RESULTS_DIR / "demo_output.txt").write_text(demo_text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
