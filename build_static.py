"""
Pre-compute static assets for Netlify deployment.

Reads data/xauusd_1h.csv, runs all analysis, and writes:
  static/stats.json   — full historical stats + regime snapshot
  static/index.html   — copy of templates/index.html

Does NOT require MetaTrader5 (Windows-only) — only reads the CSV.
Run locally to preview, or Netlify runs it on every push.
"""
import json, math, os, shutil, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    load_data, enrich, compute_stats,
    current_market_hours, BKK, SESSIONS,
)
from config_io import load_config

# ── helpers ──────────────────────────────────────────────────────────────────

def _clean(obj):
    """Recursively replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    return obj


# ── load & enrich ─────────────────────────────────────────────────────────────

cfg = load_config()
df, err = load_data()
if err:
    print(f"ERROR: {err}")
    sys.exit(1)

mh = current_market_hours()
enr = enrich(df, market_hours=mh)
thin, dead, mon_gap = mh

# ── historical stats ──────────────────────────────────────────────────────────

stats = compute_stats(enr, market_hours=mh)

stats["meta"] = {
    "symbol":           cfg.get("symbol", "XAUUSD"),
    "bars":             len(df),
    "from":             str(df["time"].dt.tz_convert(BKK).min()),
    "to":               str(df["time"].dt.tz_convert(BKK).max()),
    "tz":               "BKK (UTC+7)",
    "broker_utc_offset": cfg.get("broker_utc_offset", 3),
    "period_days":      None,
    "built_at":         pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
}

stats["market_flags"] = {
    "thin_hours":    sorted(thin),
    "dead_hours":    sorted(dead),
    "mon_gap_hours": sorted(mon_gap),
}

stats["sessions"] = SESSIONS

# Last-known price (shown in static NOW panel)
last = enr.iloc[-1]
stats["last"] = {
    "price":    round(float(last["close"]), 4),
    "bar_time": str(last["local_time"]),
}

# ── regime snapshot (30-day lookback pre-computed at build time) ──────────────

LOOKBACK = 30
cutoff   = enr["local_time"].max() - pd.Timedelta(days=LOOKBACK)
recent   = enr[enr["local_time"] >= cutoff]

overall_avg_move  = float(enr["move_pct"].mean())
overall_avg_range = float(enr["range"].mean())
rec_move = float(recent["move_pct"].mean()) if len(recent) else None

stats["regime_snapshot"] = {
    "recent_avg_range":        round(float(recent["range"].mean()), 4) if len(recent) else None,
    "historical_avg_range":    round(overall_avg_range, 4),
    "recent_avg_move_pct":     round(rec_move, 4) if rec_move is not None else None,
    "historical_avg_move_pct": round(overall_avg_move, 4),
    "delta_pct":               round((rec_move / overall_avg_move - 1) * 100, 1) if rec_move else None,
    "recent_bull_pct":         round(float(recent["bull"].mean()) * 100, 1) if len(recent) else None,
    "historical_bull_pct":     round(float(enr["bull"].mean()) * 100, 1),
    "recent_bear_pct":         round(float(recent["bear"].mean()) * 100, 1) if len(recent) else None,
    "historical_bear_pct":     round(float(enr["bear"].mean()) * 100, 1),
    "lookback_days":           LOOKBACK,
    "recent_bars":             int(len(recent)),
}

# ── hourly breakdown for regime chart (needed by drawOverview in static mode) ─

stats["recent_by_hour"] = (
    recent.groupby("hour")
    .agg(avg_range=("range", "mean"), bull_pct=("bull", "mean"), count=("range", "count"))
    .reset_index()
    .assign(bull_pct=lambda x: (x["bull_pct"] * 100).round(1))
    .round(5)
    .to_dict(orient="records")
)

stats["historical_by_hour"] = (
    enr.groupby("hour")
    .agg(avg_range=("range", "mean"), bull_pct=("bull", "mean"))
    .reset_index()
    .round(5)
    .to_dict(orient="records")
)

# ── write output ──────────────────────────────────────────────────────────────

os.makedirs("static", exist_ok=True)

stats_path = os.path.join("static", "stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(_clean(stats), f, default=str)

shutil.copy("templates/index.html", "static/index.html")

sz = os.path.getsize(stats_path) / 1024
print(f"Built {stats_path}  ({len(df)} bars, {sz:.0f} KB)")
print(f"Copied templates/index.html -> static/index.html")
print(f"Data range: {stats['meta']['from'][:10]} to {stats['meta']['to'][:10]}")
