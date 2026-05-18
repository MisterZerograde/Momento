"""
Fetches XAUUSD H1 historical data from MetaTrader 5 and saves to CSV.
Run this once (or refresh periodically) to update the cached dataset.
"""
import sys
import os
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd

MT5_PATH = r"C:\Program Files\Vantage International MT5\terminal64.exe"
SYMBOL = "XAUUSD.sc"
TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 50000  # ~5+ years of hourly bars
OUTPUT = os.path.join(os.path.dirname(__file__), "data", "xauusd_1h.csv")


def fetch():
    if not mt5.initialize(path=MT5_PATH):
        print(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"MT5 connected — build {mt5.version()}")
    print(f"Fetching {BARS} H1 bars for {SYMBOL}...")

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("No data returned. Is the symbol available in Market Watch?")
        sys.exit(1)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("Etc/GMT-3")  # Vantage server = UTC+3
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    return df


if __name__ == "__main__":
    fetch()
