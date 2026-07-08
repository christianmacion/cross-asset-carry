"""
carry.py — compute carry per asset class.

Three carry families:
  1. FX carry        : foreign_rate − USD_rate  (annualized, %)
  2. Bond carry      : US10Y − US2Y (term-structure slope, positive ⇒ roll-down earner)
  3. Commodity carry : approximation of roll yield from monthly price history.

A positive carry value means the asset rewards holding; negative means a
holding cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FX carry
# ---------------------------------------------------------------------------

def fx_carry(rates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FX carry for each non-USD currency vs USD.

    Parameters
    ----------
    rates : DataFrame
        Columns include 'USD', 'EUR', 'JPY', 'GBP', 'AUD'. Values are
        annualized rates in percent.

    Returns
    -------
    DataFrame: each column is the carry of the corresponding pair
        (e.g. 'EUR_USD' = EUR_rate − USD_rate).
        Positive ⇒ long the foreign currency earns carry.
    """
    usd = rates["USD"]
    pairs = ["EUR", "JPY", "GBP", "AUD"]
    out = {}
    for ccy in pairs:
        out[f"{ccy}_USD"] = rates[ccy] - usd
    return pd.DataFrame(out, index=rates.index)


# ---------------------------------------------------------------------------
# Bond term-structure carry
# ---------------------------------------------------------------------------

def bond_term_carry(rates: pd.DataFrame) -> pd.Series:
    """
    Carry from holding 10Y over 2Y. Positive slope ⇒ positive carry
    (roll-down earns positive carry for long bonds).
    """
    return rates["US_10Y"] - rates["US_2Y"]


# ---------------------------------------------------------------------------
# Commodity roll-yield approximation
# ---------------------------------------------------------------------------

def commodity_roll_carry(prices: pd.DataFrame,
                         short_window: int = 1,
                         long_window: int = 3) -> pd.DataFrame:
    """
    Approximate the roll yield of a futures contract without the term structure.

    Methodology
    -----------
    For each commodity price column, compute the short-window monthly log-return
    and the long-window monthly log-return, then take their annualized differential:

        carry ≈ ( r_long / long_window − r_short / short_window )

    Sign convention
    ---------------
    A positive value indicates backwardation-pushing prices higher in the
    near-month relative to recent history, a negative roll-yield-like signal
    indicates contango-like behavior.

    Note
    ----
    This is a *proxy*. True roll yield requires a multi-tenor futures curve.
    The differential captures whether the most recent monthly move is leading
    or lagging longer history, which empirically correlates with backwardation/contango
    structure in trending commodity markets.
    """
    out = {}
    for col in prices.columns:
        p = prices[col].astype(float)
        # log returns over short and long windows
        s = np.log(p / p.shift(short_window)) / short_window  # per-period, sign-positive if up
        l = np.log(p / p.shift(long_window)) / long_window
        # Carry proxy: how much faster is the recent move than the longer window
        # If the recent monthly change is *stronger* than the 3-month drift, that
        # corresponds to "near-month out-front" (a backwardation-like signal).
        out[col] = (s - l) * 12.0  # annualized (months → years)
    return pd.DataFrame(out, index=prices.index)


# ---------------------------------------------------------------------------
# Combined carry panel
# ---------------------------------------------------------------------------

def compute_all_carry(rates: pd.DataFrame,
                      prices: pd.DataFrame) -> pd.DataFrame:
    """
    Return a single panel of all 8 carry signals, indexed by month.
    Columns:
        EUR_USD, JPY_USD, GBP_USD, AUD_USD,
        US_SLOPE,
        GC=F, CL=F, HG=F
    """
    fx = fx_carry(rates)
    slope = bond_term_carry(rates).rename("US_SLOPE")
    commod = commodity_roll_carry(prices)
    # Reorder to spec
    carry = pd.concat([fx, slope, commod], axis=1)
    # Standard column ordering
    desired = ["EUR_USD", "JPY_USD", "GBP_USD", "AUD_USD",
               "US_SLOPE", "GC=F", "CL=F", "HG=F"]
    return carry[desired]


if __name__ == "__main__":
    # Smoke test
    from data_loader import load_all
    d = load_all()
    panel = compute_all_carry(d["rates"].dropna(), d["commodities"].dropna())
    print(panel.tail(6))
    print(panel.describe().T[["mean", "std"]])
