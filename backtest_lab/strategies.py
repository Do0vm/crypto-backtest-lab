"""Long/flat signals. Each function returns a Series of 0/1 where the value on
day t uses only information available at the close of day t. The engine then
applies it to day t+1's return, so no strategy can see the future."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def sma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Trend following: long while the fast moving average is above the slow one."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    return (f > s).astype(float).where(s.notna(), 0.0)


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Time-series momentum: long if the price is higher than `lookback` days ago."""
    past = close.shift(lookback)
    return (close > past).astype(float).where(past.notna(), 0.0)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def rsi_reversion(close: pd.Series, entry: float, exit: float, period: int = 14) -> pd.Series:
    """Mean reversion: buy when RSI drops below `entry`, sell when it rises above `exit`."""
    if entry >= exit:
        raise ValueError("entry level must be below exit level")
    r = rsi(close, period).to_numpy()
    pos = np.zeros(len(r))
    holding = False
    for i, v in enumerate(r):
        if np.isnan(v):
            pos[i] = 0.0
            continue
        if not holding and v < entry:
            holding = True
        elif holding and v > exit:
            holding = False
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=close.index)


# Parameter grids searched on the in-sample period only.
GRIDS = {
    "sma_cross": [{"fast": f, "slow": s} for f in (5, 10, 20, 30, 50)
                  for s in (50, 100, 150, 200) if f < s],
    "momentum": [{"lookback": n} for n in (7, 14, 30, 60, 90, 120, 180)],
    "rsi_reversion": [{"entry": e, "exit": x} for e in (20, 25, 30, 35)
                      for x in (50, 55, 60, 65, 70)],
}
FUNCS = {"buy_and_hold": buy_and_hold, "sma_cross": sma_cross,
         "momentum": momentum, "rsi_reversion": rsi_reversion}
