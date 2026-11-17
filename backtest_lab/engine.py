"""Backtest engine and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365  # crypto trades every day

# Cost per unit of turnover (one side of a trade). 0.10% taker fee plus 0.05%
# assumed slippage. Entering and later exiting a position costs 2 x this.
DEFAULT_COST = 0.0015


@dataclass(frozen=True)
class Result:
    returns: pd.Series  # daily strategy returns after costs
    position: pd.Series  # position held during each day (0 or 1)
    equity: pd.Series  # growth of 1 unit of capital


def run(close: pd.Series, signal: pd.Series, cost: float = DEFAULT_COST) -> Result:
    """Apply a signal to prices.

    The signal decided at the close of day t is held during day t+1
    (`signal.shift(1)`), so a trade can never use the price it is trading on.
    Costs are charged whenever the position changes.
    """
    asset_ret = close.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat = position * asset_ret - turnover * cost
    return Result(strat, position, (1 + strat).cumprod())


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall, as a negative fraction (e.g. -0.55)."""
    return float((equity / equity.cummax() - 1).min())


def metrics(res: Result) -> dict:
    r = res.returns
    n = len(r)
    total = float(res.equity.iloc[-1] - 1)
    years = n / DAYS_PER_YEAR
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    vol = float(r.std(ddof=0) * np.sqrt(DAYS_PER_YEAR))
    sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(DAYS_PER_YEAR)) if r.std() > 0 else 0.0
    downside = r[r < 0].std(ddof=0)
    sortino = float(r.mean() / downside * np.sqrt(DAYS_PER_YEAR)) if downside > 0 else 0.0
    mdd = max_drawdown(res.equity)
    entries = int(((res.position == 1) & (res.position.shift(1, fill_value=0) == 0)).sum())
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "exposure": float(res.position.mean()),
        "trades": entries,
    }
