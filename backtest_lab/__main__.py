import argparse

from .research import main

p = argparse.ArgumentParser(prog="backtest_lab", description="Run the in/out-of-sample study.")
p.add_argument("--refresh", action="store_true", help="re-download candles from Binance")
main(refresh=p.parse_args().refresh)
