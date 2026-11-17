"""Download and cache daily OHLCV candles from Binance's public klines API."""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KLINES_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


def _fetch(symbol: str, start_ms: int, interval: str = "1d", limit: int = 1000) -> list:
    url = f"{KLINES_URL}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-backtest-lab/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def download(symbol: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Page through the API 1000 candles at a time until we reach today.
    The still-open candle for today is dropped so every row is final."""
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows: list = []
    while True:
        batch = _fetch(symbol, start_ms)
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)  # be polite to the public endpoint
    df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns="open_time").set_index("date").astype(float)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return df[df.index < today]


def load(symbol: str, refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1d.csv"
    if refresh or not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        download(symbol).to_csv(path)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    validate(df, symbol)
    return df


def validate(df: pd.DataFrame, symbol: str) -> None:
    """Fail loudly on data problems instead of silently backtesting garbage."""
    if df.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate dates")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: dates out of order")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive prices")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: high below open/close")
    gaps = df.index.to_series().diff().dt.days.dropna()
    if (gaps > 1).any():
        raise ValueError(f"{symbol}: {int((gaps > 1).sum())} missing days")
