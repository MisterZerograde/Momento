"""
Fetches H1 historical data from a user-configured MT5 terminal and saves to CSV.

Two modes:
  • default       — pulls the last N bars (config.bars)         via copy_rates_from_pos
  • --from-date X — pulls every bar from date X to now (MAX)   via copy_rates_range

`copy_rates_range` bypasses MT5's "Max bars in chart" cap (default 100k), so it
returns the full history the broker keeps for the symbol — even back to 2010+.

Reads MT5 path, symbol, bars, and broker UTC offset from config.json
(managed via the Momento Settings tab).
"""
import argparse
import sys
import os
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd

from config_io import load_config, broker_tz_string

TIMEFRAME = mt5.TIMEFRAME_H1
OUTPUT = os.path.join(os.path.dirname(__file__), "data", "xauusd_1h.csv")


def fetch(from_date=None):
    cfg = load_config()
    mt5_path = cfg["mt5_path"]
    symbol   = cfg["symbol"]
    bars     = int(cfg["bars"])
    tz       = broker_tz_string(cfg["broker_utc_offset"])

    if not mt5.initialize(path=mt5_path):
        err = mt5.last_error()
        print(f"MT5 initialize failed: {err}")
        sys.exit(1)

    print(f"MT5 connected — build {mt5.version()}")

    if from_date:
        # Date-range mode — full history from `from_date` to now.
        start_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.now(timezone.utc)
        print(f"Fetching H1 bars for {symbol} from {from_date} to {end_dt:%Y-%m-%d}…")
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, start_dt, end_dt)
    else:
        # Tail mode — last N bars from the current bar backwards.
        print(f"Fetching last {bars} H1 bars for {symbol}…")
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"No data returned for '{symbol}'. Is it available in Market Watch?")
        sys.exit(1)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize(tz)
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    return df


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from-date", dest="from_date",
                   help="YYYY-MM-DD; if set, fetches via copy_rates_range from this date to now.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch(from_date=args.from_date)
