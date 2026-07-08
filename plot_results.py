#!/usr/bin/env python3
"""
plot_results.py — render the 3-panel cross-asset carry figure.

Panels:
  1. Cumulative returns (carry portfolio vs equal-weight benchmark)
  2. Carry time-series per asset (heatmap, z-scored)
  3. Per-month rolling Sharpe (train vs test, with split marker)

Reads results/results.json + results/port_returns.csv + results/carry_series.csv
written by run.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"


def _rolling_sharpe(r: pd.Series, window: int = 12) -> pd.Series:
    """Rolling annualized Sharpe (window-month)."""
    m = r.rolling(window, min_periods=window // 2).mean()
    sd = r.rolling(window, min_periods=window // 2).std(ddof=1)
    out = (m / sd.replace(0, np.nan)) * np.sqrt(12)
    return out.fillna(0.0)


def _load():
    rj = RESULTS_DIR / "results.json"
    pr = RESULTS_DIR / "port_returns.csv"
    cr = RESULTS_DIR / "carry_series.csv"
    ar = RESULTS_DIR / "asset_returns.csv"
    if not (rj.exists() and pr.exists() and cr.exists()):
        raise FileNotFoundError("Missing run.py outputs — run run.py first.")

    with open(rj) as f:
        meta = json.load(f)

    port = pd.read_csv(pr, parse_dates=["date"]).set_index("date")
    carry = pd.read_csv(cr, parse_dates=["date"]).set_index("date")
    asset_rets = pd.read_csv(ar, parse_dates=["date"]).set_index("date")
    return meta, port, carry, asset_rets


def main() -> int:
    meta, port, carry, asset_rets = _load()

    bench = asset_rets.mean(axis=1).rename("benchmark")
    bench = bench.loc[bench.index.intersection(port.index)]
    port_eq = (1 + port["net_return"]).cumprod()
    bench_eq = (1 + bench).cumprod()

    rs = _rolling_sharpe(port["net_return"])
    split_year = meta["config"]["test_years"].split("-")[0]
    split_year = int(split_year) - 1   # last year of train
    split_date = pd.Timestamp(f"{split_year + 1}-01-01")

    # Carry z-score heatmap
    zc = (carry - carry.rolling(12, min_periods=4).mean()) / carry.rolling(12, min_periods=4).std()
    zc = zc.clip(-3, 3)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [3, 2, 2]})

    # ---- Panel 1: cumulative returns ----
    ax = axes[0]
    ax.plot(port_eq.index, port_eq.values, lw=2.0, color="#1f4e79",
            label=f"Carry portfolio (Sharpe IS={meta['in_sample']['annualized_sharpe']:.2f} | "
                  f"OOS={meta['out_of_sample']['annualized_sharpe']:.2f})")
    ax.plot(bench_eq.index, bench_eq.values, lw=1.6, color="#9c9c9c",
            ls="--", label="Equal-weight benchmark (long-only)")
    ax.axvline(split_date, color="#c0392b", ls=":", lw=1.2, label=f"Train / test split ({split_date.year})")

    # mark OOS section shading
    ax.axvspan(split_date, port_eq.index.max(), color="#c0392b", alpha=0.05)

    ax.set_ylabel("Cumulative return (×)", fontsize=10)
    ax.set_title(
        "Cross-asset carry long-short · cumulative growth of $1",
        fontsize=12, fontweight="bold", loc="left",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # ---- Panel 2: carry heatmap ----
    ax = axes[1]
    im = ax.imshow(zc.T.values, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(range(len(zc.columns)))
    ax.set_yticklabels(zc.columns, fontsize=9)
    ax.set_xticks(range(0, len(zc.index), 6))
    ax.set_xticklabels([zc.index[i].strftime("%Y") for i in range(0, len(zc.index), 6)],
                       rotation=0, fontsize=9)
    ax.set_title("Carry z-scored by 12-month rolling window (truncated ±2.5)",
                 fontsize=11, loc="left")
    ax.set_xlabel("")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("z-score", fontsize=9)
    ax.axvline(x=zc.index.get_loc(carry.index[carry.index >= split_date][0]) - 0.5,
               color="black", ls=":", lw=1.0)
    ax.text(0.01, 0.95, "TRAIN", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color="#1f4e79")
    ax.text(0.95, 0.95, "TEST", transform=ax.transAxes, fontsize=9,
            ha="right", fontweight="bold", color="#c0392b")

    # ---- Panel 3: rolling Sharpe train vs test ----
    ax = axes[2]
    rs = _rolling_sharpe(port["net_return"])
    train_rs = rs.loc[rs.index <= split_date]
    test_rs = rs.loc[rs.index > split_date]

    ax.plot(train_rs.index, train_rs.values, lw=1.6, color="#1f4e79",
            label=f"Rolling 12-mo Sharpe — train (mean={train_rs.mean():.2f})")
    ax.plot(test_rs.index, test_rs.values, lw=1.6, color="#c0392b",
            label=f"Rolling 12-mo Sharpe — test  (mean={test_rs.mean():.2f})")
    ax.axhline(0, color="gray", ls="-", lw=0.8)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.axhline(-1.0, color="gray", ls=":", lw=0.8)
    ax.axvline(split_date, color="#c0392b", ls=":", lw=1.0)
    ax.axvspan(split_date, rs.index.max(), color="#c0392b", alpha=0.05)
    ax.set_title("Per-month rolling 12-month Sharpe (annualized)", fontsize=11, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Sharpe", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    out = RESULTS_DIR / "figure.png"
    fig.suptitle("cross-asset-carry · 8-instrument long-short, vol-weighted, 5 bps TC",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
