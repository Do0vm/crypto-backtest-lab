import unittest

import numpy as np
import pandas as pd

from backtest_lab import engine, strategies
from backtest_lab.data import validate


def prices(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


class NoLookAheadTests(unittest.TestCase):
    def test_changing_the_future_does_not_change_past_positions(self):
        rng = np.random.default_rng(0)
        base = prices(100 * np.cumprod(1 + rng.normal(0, 0.03, 400)))
        shocked = base.copy()
        shocked.iloc[300:] *= 3  # rewrite the future

        for name, fn, kw in [("sma", strategies.sma_cross, {"fast": 10, "slow": 50}),
                             ("mom", strategies.momentum, {"lookback": 30}),
                             ("rsi", strategies.rsi_reversion, {"entry": 30, "exit": 60})]:
            with self.subTest(name):
                a = engine.run(base, fn(base, **kw)).position.iloc[:301]
                b = engine.run(shocked, fn(shocked, **kw)).position.iloc[:301]
                pd.testing.assert_series_equal(a, b)

    def test_signal_is_applied_next_day(self):
        close = prices([100, 110, 121])
        signal = pd.Series([1.0, 0.0, 0.0], index=close.index)
        res = engine.run(close, signal, cost=0)
        # long decided at day 0 close -> earns day 1's +10%, nothing on day 2
        self.assertAlmostEqual(res.returns.iloc[1], 0.10)
        self.assertAlmostEqual(res.returns.iloc[2], 0.0)


class CostTests(unittest.TestCase):
    def test_flat_price_round_trip_costs_two_sides(self):
        close = prices([100] * 5)
        signal = pd.Series([1, 1, 0, 0, 0], index=close.index, dtype=float)
        res = engine.run(close, signal, cost=0.001)
        self.assertAlmostEqual(res.returns.sum(), -0.002)

    def test_buy_and_hold_pays_entry_once(self):
        close = prices([100, 100, 100])
        res = engine.run(close, strategies.buy_and_hold(close), cost=0.01)
        self.assertAlmostEqual(res.returns.sum(), -0.01)


class MetricTests(unittest.TestCase):
    def test_max_drawdown(self):
        eq = pd.Series([1.0, 2.0, 1.0, 1.5, 0.5, 3.0])
        self.assertAlmostEqual(engine.max_drawdown(eq), -0.75)

    def test_cagr_of_doubling_in_a_year(self):
        n = engine.DAYS_PER_YEAR
        daily = 2 ** (1 / n) - 1
        r = pd.Series(daily, index=pd.date_range("2020-01-01", periods=n))
        res = engine.Result(r, pd.Series(1.0, index=r.index), (1 + r).cumprod())
        self.assertAlmostEqual(engine.metrics(res)["cagr"], 1.0, places=6)

    def test_trade_count(self):
        pos = pd.Series([0, 1, 1, 0, 1, 0, 0, 1], dtype=float)
        res = engine.Result(pos * 0, pos, pos * 0 + 1)
        self.assertEqual(engine.metrics(res)["trades"], 3)


class StrategyTests(unittest.TestCase):
    def test_rsi_bounds_and_extremes(self):
        up = prices(np.arange(1, 60))
        r = strategies.rsi(up).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())
        self.assertAlmostEqual(r.iloc[-1], 100.0)

    def test_sma_rejects_bad_windows(self):
        with self.assertRaises(ValueError):
            strategies.sma_cross(prices([1, 2, 3]), fast=50, slow=20)

    def test_momentum_long_in_uptrend(self):
        s = strategies.momentum(prices(np.linspace(100, 200, 50)), lookback=10)
        self.assertTrue((s.iloc[10:] == 1).all())
        self.assertTrue((s.iloc[:10] == 0).all())


class DataValidationTests(unittest.TestCase):
    def frame(self, closes, dates=None):
        idx = dates or pd.date_range("2020-01-01", periods=len(closes))
        c = pd.Series(closes, index=idx, dtype=float)
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0})

    def test_good_data_passes(self):
        validate(self.frame([1, 2, 3]), "X")

    def test_missing_day_rejected(self):
        dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-05"])
        with self.assertRaises(ValueError):
            validate(self.frame([1, 2, 3], dates), "X")

    def test_bad_price_rejected(self):
        with self.assertRaises(ValueError):
            validate(self.frame([1, 0, 3]), "X")


if __name__ == "__main__":
    unittest.main()
