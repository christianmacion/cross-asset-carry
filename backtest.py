"""
backtest.py — cross-asset long-short carry portfolio.

Methodology
-----------
  - Each month rank all instruments by their carry value (computed at the
    prior month-end, no peeking).
  - Long the top K_L (default 3) by carry; short the bottom K_S (default 3).
  - Vol-weighted: weight each leg inversely to its 60-day realized volatility
    (so lower-vol assets get more portfolio risk).
  - Hold one month, then rebalance.
  - 5 bps round-trip transaction cost applied to turnover each rebalance.

Train (2010-2020) / Test (2020-2025) chronological split, NO peeking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TOP_K = 3
BOTTOM_K = 3
TC_ROUNDTRIP = 0.0005  # 5 bps per round trip, one-way implied


def _annualized_sharpe(daily_returns: pd.Series, freq: int = 12) -> float:
    """Annualized Sharpe ratio. freq=12 for monthly returns."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = np.std(r, ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(freq))


def _max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of cumulative returns."""
    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def realized_vol(prices: pd.Series, window: int = 6) -> pd.Series:
    """Rolling realized vol (std of monthly log returns * sqrt(12))."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(12)


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly simple returns from a price DataFrame."""
    return price_df.pct_change()


def carry_backtest(carry_panel: pd.DataFrame,
                   returns_panel: pd.DataFrame,
                   start_year: int = 2010,
                   end_year: int = 2025,
                   top_k: int = TOP_K,
                   bottom_k: int = BOTTOM_K,
                   tc: float = TC_ROUNDTRIP,
                   vol_window: int = 6) -> pd.DataFrame:
    """
    Run the cross-asset carry long-short backtest.

    Parameters
    ----------
    carry_panel : DataFrame, indexed by month
        Carry value per instrument (higher ⇒ better to hold long).
    returns_panel : DataFrame, indexed by month
        Forward 1-month simple returns per instrument. Same columns as carry_panel.
    start_year, end_year : int
        Calendar bounds for the backtest.

    Returns
    -------
    DataFrame with columns:
        date, long_legs (list[str]), short_legs (list[str]),
        net_return (the portfolio monthly return net of TC).
    """
    dates = carry_panel.index
    out_rows = []
    last_weights = None

    for i in range(len(dates) - 1):
        t = dates[i]
        t_next = dates[i + 1]
        if t.year < start_year or t.year > end_year:
            continue

        # Carry ranking at month-end t (use info available at t)
        c = carry_panel.iloc[i].dropna()
        if len(c) < top_k + bottom_k:
            continue
        ranked = c.sort_values(ascending=False)

        # 60-day realized vol per instrument (approx: use last vol_window monthly returns)
        rets_up_to_t = returns_panel.loc[:t].iloc[-vol_window:]
        vol = rets_up_to_t.std()

        # Inverse-vol weights (only on assets with a valid vol)
        inv_vol_full = (1.0 / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()

        # Keep only top/bottom legs that have a realized vol
        long_candidates = [x for x in ranked.index if x in inv_vol_full.index]
        short_candidates = [x for x in ranked.index[::-1] if x in inv_vol_full.index]
        long_legs = long_candidates[:top_k]
        short_legs = short_candidates[:bottom_k]
        if len(long_legs) < top_k or len(short_legs) < bottom_k:
            continue

        # Renormalize within legs
        long_w = inv_vol_full[long_legs]
        short_w = inv_vol_full[short_legs]
        if long_w.sum() == 0 or short_w.sum() == 0:
            continue
        long_w = long_w / long_w.sum() * 0.5      # half allocation to longs
        short_w = short_w / short_w.sum() * 0.5   # half allocation to shorts

        # Forward returns (month t+1)
        rets_next = returns_panel.loc[t_next]
        gross_long = (long_w * rets_next[long_legs]).sum()
        gross_short = (-short_w * rets_next[short_legs]).sum()
        gross = gross_long + gross_short

        # Transaction cost (turnover vs prior month weights)
        # For simplicity: cost = tc per leg that turned over
        # Approximation: tc per rebalance when the leg set changes
        current_weights_long = long_w
        current_weights_short = short_w
        if last_weights is not None:
            # Fraction of weights that changed
            prev_long = last_weights.get("long", {})
            prev_short = last_weights.get("short", {})
            turnover = 0.0
            for leg in long_legs:
                if prev_long.get(leg) is None:
                    turnover += current_weights_long[leg]
                else:
                    turnover += abs(current_weights_long[leg] - prev_long[leg])
            for leg in short_legs:
                if prev_short.get(leg) is None:
                    turnover += abs(current_weights_short[leg])
                else:
                    turnover += abs(current_weights_short[leg] - prev_short[leg])
            cost = turnover * tc
        else:
            cost = (top_k + bottom_k) / (top_k + bottom_k) * tc

        net = gross - cost
        out_rows.append({
            "date": t_next,
            "long_legs": long_legs,
            "short_legs": short_legs,
            "gross_return": gross,
            "tc_cost": cost,
            "net_return": net,
        })

        last_weights = {
            "long": current_weights_long.to_dict(),
            "short": current_weights_short.to_dict(),
        }

    return pd.DataFrame(out_rows).set_index("date")


def split_train_test(port_ret: pd.DataFrame,
                     split_year: int = 2020) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split (NO peeking)."""
    train = port_ret[port_ret.index.year < split_year].copy()
    test = port_ret[port_ret.index.year >= split_year].copy()
    return train, test


def summary(port_ret: pd.DataFrame) -> dict:
    """Compute headline metrics from a returns DataFrame with 'net_return'."""
    r = port_ret["net_return"]
    sharpe = _annualized_sharpe(r, freq=12)
    cum = (1.0 + r).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0)
    annual_ret = float(cum.iloc[-1] ** (12.0 / max(len(r), 1)) - 1.0)
    mdd = _max_drawdown(r)
    return {
        "n_months": int(len(r)),
        "annualized_sharpe": round(sharpe, 3),
        "annualized_return": round(annual_ret, 4),
        "total_return": round(total_ret, 4),
        "max_drawdown": round(mdd, 4),
        "win_rate": round(float((r > 0).mean()), 3) if len(r) else 0.0,
    }


def benchmark_returns(returns_panel: pd.DataFrame) -> pd.Series:
    """Equal-weight long-only benchmark on the same instruments."""
    return returns_panel.mean(axis=1)


if __name__ == "__main__":
    # Smoke test
    from data_loader import load_all
    from carry import compute_all_carry
    d = load_all()
    carry = compute_all_carry(d["rates"].dropna(), d["commodities"].dropna())
    # We need forward returns — include rates carry as constant cost/income via FX rate channel
    rets = d["commodities"].pct_change()
    # For FX and bond SLOPE, use rates differentials as proxy by year-end changes
    # For brevity in this smoke test, just trade commodities for the smoke
    carry_simple = carry.iloc[:, 5:8]
    rets_simple = rets
    port = carry_backtest(carry_simple, rets_simple)
    s = summary(port)
    print(s)
