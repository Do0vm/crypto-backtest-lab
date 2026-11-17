"""The study: pick each strategy's parameters on 2018-2022 only, then measure
them on 2023-onwards data they never saw. Writes tables and charts to reports/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import data, engine, strategies  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent / "reports"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SPLIT = pd.Timestamp("2023-01-01")  # first out-of-sample day

COLORS = {"buy_and_hold": "#8a94a6", "sma_cross": "#2563eb",
          "momentum": "#16a34a", "rsi_reversion": "#d97706"}


def period(res: engine.Result, start=None, end=None) -> engine.Result:
    """Slice a full-history result and restart its equity curve at 1."""
    mask = pd.Series(True, index=res.returns.index)
    if start is not None:
        mask &= res.returns.index >= start
    if end is not None:
        mask &= res.returns.index < end
    r, p = res.returns[mask], res.position[mask]
    return engine.Result(r, p, (1 + r).cumprod())


def evaluate(close: pd.Series, name: str, params: dict):
    full = engine.run(close, strategies.FUNCS[name](close, **params))
    return period(full, end=SPLIT), period(full, start=SPLIT)


def study(symbol: str) -> dict:
    df = data.load(symbol)
    close = df["close"]
    out = {"symbol": symbol, "start": str(close.index[0].date()),
           "end": str(close.index[-1].date()), "rows": {}, "grid": {}}

    is_bh, oos_bh = evaluate(close, "buy_and_hold", {})
    out["rows"]["buy_and_hold"] = {"params": {}, "is": engine.metrics(is_bh),
                                   "oos": engine.metrics(oos_bh)}
    curves = {"buy_and_hold": oos_bh.equity}

    for name, grid in strategies.GRIDS.items():
        scored = []
        for params in grid:
            is_res, oos_res = evaluate(close, name, params)
            scored.append((params, engine.metrics(is_res), engine.metrics(oos_res)))
        best = max(scored, key=lambda s: s[1]["sharpe"])  # chosen on in-sample ONLY
        out["rows"][name] = {"params": best[0], "is": best[1], "oos": best[2]}
        out["grid"][name] = [{"params": p, "is_sharpe": i["sharpe"], "oos_sharpe": o["sharpe"]}
                             for p, i, o in scored]
        curves[name] = evaluate(close, name, best[0])[1].equity

    plot_equity(symbol, curves)
    return out


def plot_equity(symbol: str, curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    for name, eq in curves.items():
        lw = 2.4 if name == "buy_and_hold" else 1.6
        ax.plot(eq.index, eq.values, label=name.replace("_", " "), color=COLORS[name], lw=lw)
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_title(f"{symbol}: out-of-sample growth of $1 (params chosen on 2018–2022)",
                 fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{symbol}_oos_equity.png")
    plt.close(fig)


def plot_sma_heatmap(result: dict) -> None:
    grid = result["grid"]["sma_cross"]
    fasts = sorted({g["params"]["fast"] for g in grid})
    slows = sorted({g["params"]["slow"] for g in grid})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=130, sharey=True)
    for ax, key, title in zip(axes, ("is_sharpe", "oos_sharpe"),
                              ("In-sample Sharpe (2018–2022)", "Out-of-sample Sharpe (2023–)")):
        m = np.full((len(fasts), len(slows)), np.nan)
        for g in grid:
            m[fasts.index(g["params"]["fast"]), slows.index(g["params"]["slow"])] = g[key]
        im = ax.imshow(m, cmap="RdYlGn", vmin=-1, vmax=2, aspect="auto")
        for i in range(len(fasts)):
            for j in range(len(slows)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(slows)), slows)
        ax.set_yticks(range(len(fasts)), fasts)
        ax.set_xlabel("slow SMA (days)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("fast SMA (days)")
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.suptitle(f"{result['symbol']} SMA crossover: does in-sample skill survive?",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(REPORTS / f"{result['symbol']}_sma_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fmt_params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "–"


def write_markdown(results: list[dict]) -> str:
    lines = []
    for res in results:
        lines += [f"### {res['symbol']}  ({res['start']} to {res['end']})", "",
                  "| Strategy | Params (picked in-sample) | IS Sharpe | OOS Sharpe | OOS CAGR "
                  "| OOS max DD | OOS trades | OOS time in market |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in res["rows"].items():
            i, o = row["is"], row["oos"]
            lines.append(
                f"| {name.replace('_', ' ')} | {fmt_params(row['params'])} | {i['sharpe']:.2f} "
                f"| **{o['sharpe']:.2f}** | {o['cagr']:.1%} | {o['max_drawdown']:.1%} "
                f"| {o['trades']} | {o['exposure']:.0%} |")
        lines.append("")
    return "\n".join(lines)


def main(refresh: bool = False) -> list[dict]:
    REPORTS.mkdir(exist_ok=True)
    if refresh:
        for s in SYMBOLS:
            data.load(s, refresh=True)
    results = [study(s) for s in SYMBOLS]
    plot_sma_heatmap(results[0])
    (REPORTS / "results.md").write_text(write_markdown(results), encoding="utf-8")
    (REPORTS / "results.json").write_text(json.dumps(results, indent=1, default=float),
                                          encoding="utf-8")
    print(write_markdown(results))
    return results
