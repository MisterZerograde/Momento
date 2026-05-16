import os, json, math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "xauusd_1h.csv")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BKK = "Asia/Bangkok"

# Trading sessions (BKK / UTC+7). XAUUSD is most active during overlaps.
# Asia: 06–15  London: 14–23  NY: 19–04
SESSIONS = [
    {"name": "Asia",          "start": 6,  "end": 15, "color": "#f5a524"},
    {"name": "London",        "start": 14, "end": 23, "color": "#2f5fff"},
    {"name": "NY",            "start": 19, "end": 28, "color": "#2fb96d"},  # 28 means 4am next day
    {"name": "London-NY OVL", "start": 19, "end": 23, "color": "#9a4dff"},  # killer overlap
]

# NY session closes ~5 PM ET = 04:00 BKK (EDT) / 05:00 BKK (EST).
# Vantage MT5 daily reset happens at midnight UTC = 07:00 BKK — NO bar is generated
# at hour 7 (zero rows in dataset). Dead zone is effectively 04:00–07:59 BKK.
THIN_HOURS    = {4, 5, 6}    # BKK hours — NY close / illiquid pre-reset
DEAD_HOURS    = {7}           # BKK hours — broker reset, bar literally missing
MON_GAP_HOURS = {8}          # BKK hour on Monday (dow=0) — first bar after weekend


def load_data():
    if not os.path.exists(DATA_PATH):
        return None, "Data file not found. Run fetch_data.py first."
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df, None


def enrich(df):
    """Add derived columns used everywhere."""
    df = df.copy()
    local = df["time"].dt.tz_convert(BKK)
    df["local_time"] = local
    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["date"] = local.dt.date
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["move_pct"] = df["body"] / df["open"] * 100
    df["bull"] = (df["close"] > df["open"]).astype(int)
    df["bear"] = (df["close"] < df["open"]).astype(int)
    df["direction_pct"] = (df["close"] - df["open"]) / df["open"] * 100  # signed

    # Mark unreliable bars so callers can filter or flag them.
    # thin_market: NY close / broker rollover (4–5 AM BKK) — spreads widen, fills are bad.
    # monday_gap:  first bars of the week (Mon 5–6 AM BKK) include the weekend gap.
    df["thin_market"] = df["hour"].isin(THIN_HOURS)
    df["dead_hour"]   = df["hour"].isin(DEAD_HOURS)   # broker reset — bar never generated
    df["monday_gap"]  = (df["dow"] == 0) & df["hour"].isin(MON_GAP_HOURS)
    df["skip_bar"]    = df["thin_market"] | df["monday_gap"]
    return df


def session_for(hour):
    """Return the most relevant session for a given BKK hour."""
    # priority: overlap > NY > London > Asia
    if 19 <= hour <= 22:
        return "London-NY OVL"
    if 19 <= hour or hour <= 3:
        return "NY"
    if 14 <= hour <= 18:
        return "London"
    if 6 <= hour <= 13:
        return "Asia"
    return "Off-hours"


def confidence(n):
    """Map sample count to a 0-100 confidence score."""
    if n < 10:  return 10
    if n < 30:  return 35
    if n < 60:  return 55
    if n < 120: return 70
    if n < 250: return 85
    return 95


def compute_stats(df_enriched):
    df = df_enriched
    df_clean = df[~df["skip_bar"]]   # exclude thin market + Monday gap bars

    by_hour = (
        df.groupby("hour")
        .agg(
            avg_range=("range", "mean"),
            avg_body=("body", "mean"),
            avg_move_pct=("move_pct", "mean"),
            bull_pct=("bull", "mean"),
            count=("range", "count"),
        )
        .reset_index()
    )
    by_hour["bull_pct"] = (by_hour["bull_pct"] * 100).round(1)
    by_hour["thin_market"] = by_hour["hour"].isin(THIN_HOURS)
    by_hour["monday_gap_included"] = by_hour["hour"].isin(MON_GAP_HOURS)

    # Clean stats (thin+gap bars removed) — more tradeable picture
    by_hour_clean = (
        df_clean.groupby("hour")
        .agg(
            avg_range_clean=("range", "mean"),
            bull_pct_clean=("bull", "mean"),
            count_clean=("range", "count"),
        )
        .reset_index()
    )
    by_hour_clean["bull_pct_clean"] = (by_hour_clean["bull_pct_clean"] * 100).round(1)
    by_hour = by_hour.merge(by_hour_clean, on="hour", how="left")
    by_hour = by_hour.round(5)

    by_day = (
        df.groupby("dow")
        .agg(
            avg_range=("range", "mean"),
            avg_move_pct=("move_pct", "mean"),
            bull_pct=("bull", "mean"),
            count=("range", "count"),
        )
        .reset_index()
    )
    by_day["bull_pct"] = (by_day["bull_pct"] * 100).round(1)
    by_day["day_name"] = by_day["dow"].apply(lambda x: DAYS[x])
    by_day = by_day.round(5)

    pivot_range = (
        df.groupby(["dow", "hour"])["range"].mean()
        .unstack(level="hour").reindex(index=range(7), columns=range(24))
    )
    pivot_move = (
        df.groupby(["dow", "hour"])["move_pct"].mean()
        .unstack(level="hour").reindex(index=range(7), columns=range(24))
    )
    pivot_bull = (
        df.groupby(["dow", "hour"])["bull"].mean()
        .unstack(level="hour").reindex(index=range(7), columns=range(24))
    ) * 100
    pivot_count = (
        df.groupby(["dow", "hour"])["range"].count()
        .unstack(level="hour").reindex(index=range(7), columns=range(24))
    )

    def pivot_to_list(p):
        rows = p.round(4).values.tolist()
        return [[None if isinstance(v, float) and math.isnan(v) else v for v in row] for row in rows]

    # Top slots exclude thin-market and Monday-gap bars entirely
    hd = (
        df_clean.groupby(["dow", "hour"])
        .agg(avg_range=("range", "mean"), avg_move_pct=("move_pct", "mean"),
             bull_pct=("bull", "mean"), count=("range", "count"))
        .reset_index()
    )
    hd["bull_pct"] = (hd["bull_pct"] * 100).round(1)
    hd["day_name"] = hd["dow"].apply(lambda x: DAYS[x])
    hd = hd.sort_values("avg_range", ascending=False).head(20).round(5)

    return {
        "by_hour": by_hour.to_dict(orient="records"),
        "by_day": by_day.to_dict(orient="records"),
        "heatmap": {
            "range": pivot_to_list(pivot_range),
            "move_pct": pivot_to_list(pivot_move),
            "bull_pct": pivot_to_list(pivot_bull),
            "count": pivot_to_list(pivot_count),
            "days": DAY_SHORT,
            "hours": list(range(24)),
        },
        "top_slots": hd.to_dict(orient="records"),
    }


def compute_now(df, lookback_days=30):
    """Trading-decision payload: current hour, today's forecast, recent regime."""
    now_utc = pd.Timestamp.utcnow().tz_convert(BKK) if pd.Timestamp.utcnow().tz else \
              pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(BKK)
    now = now_utc
    cur_hour = int(now.hour)
    cur_dow = int(now.dayofweek)
    cur_day_name = DAYS[cur_dow]
    minutes_to_next = 60 - now.minute

    last_price = float(df.iloc[-1]["close"])
    last_bar_time = df.iloc[-1]["local_time"]

    is_thin   = cur_hour in THIN_HOURS
    is_dead   = cur_hour in DEAD_HOURS
    is_mongap = cur_dow == 0 and cur_hour in MON_GAP_HOURS
    is_skip   = is_thin or is_dead or is_mongap

    # ── This hour historically (this DOW + this hour) ──────────────────
    # Always use clean data (skip thin/gap bars) for actionable stats
    df_clean = df[~df["skip_bar"]]
    same_slot     = df_clean[(df_clean["dow"] == cur_dow) & (df_clean["hour"] == cur_hour)]
    all_this_hour = df_clean[df_clean["hour"] == cur_hour]

    def slot_stats(sub):
        if len(sub) == 0:
            return {"avg_range": None, "bull_pct": None, "avg_move_pct": None, "count": 0, "confidence": 0}
        return {
            "avg_range": round(float(sub["range"].mean()), 4),
            "bull_pct": round(float(sub["bull"].mean()) * 100, 1),
            "avg_move_pct": round(float(sub["move_pct"].mean()), 4),
            "max_range": round(float(sub["range"].max()), 4),
            "count": int(len(sub)),
            "confidence": confidence(len(sub)),
        }

    this_slot = slot_stats(same_slot)
    this_hour_overall = slot_stats(all_this_hour)

    # ── Today's remaining-hours forecast ───────────────────────────────
    today_forecast = []
    for h in range(cur_hour, 24):
        h_thin   = h in THIN_HOURS
        h_dead   = h in DEAD_HOURS
        h_mongap = cur_dow == 0 and h in MON_GAP_HOURS
        h_skip   = h_thin or h_mongap
        if h_dead:
            today_forecast.append({
                "hour": h, "session": "Reset", "is_now": (h == cur_hour),
                "thin_market": False, "dead_hour": True, "monday_gap": False,
                "skip_bar": False, "avg_range": None, "bull_pct": None,
                "avg_move_pct": None, "count": 0, "confidence": 0,
            })
            continue
        # Use clean data for tradeable hours; raw for thin/gap so count is visible
        src = df if h_skip else df_clean
        sub = src[(src["dow"] == cur_dow) & (src["hour"] == h)]
        s = slot_stats(sub)
        s["hour"] = h
        s["session"] = session_for(h)
        s["is_now"] = (h == cur_hour)
        s["thin_market"] = h_thin
        s["dead_hour"]   = False
        s["monday_gap"]  = h_mongap
        s["skip_bar"]    = h_skip
        today_forecast.append(s)

    # ── Next high-volatility window (next 8h) ──────────────────────────
    upcoming = []
    for offset in range(1, 9):
        future_time = now + pd.Timedelta(hours=offset)
        h = int(future_time.hour)
        d = int(future_time.dayofweek)
        sub = df[(df["dow"] == d) & (df["hour"] == h)]
        if len(sub) >= 10:
            upcoming.append({
                "hour": h, "dow": d, "day_name": DAYS[d],
                "offset_h": offset,
                "avg_range": round(float(sub["range"].mean()), 4),
                "bull_pct": round(float(sub["bull"].mean()) * 100, 1),
                "session": session_for(h),
                "count": int(len(sub)),
            })

    overall_avg_range = float(df["range"].mean())
    next_high_vol = None
    for u in upcoming:
        if u["avg_range"] > overall_avg_range * 1.15:
            next_high_vol = u
            break

    # ── Recent regime: last N days vs all-time ─────────────────────────
    cutoff = df["local_time"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["local_time"] >= cutoff]

    recent_by_hour = (
        recent.groupby("hour")
        .agg(avg_range=("range", "mean"), bull_pct=("bull", "mean"), count=("range", "count"))
        .reset_index()
    )
    recent_by_hour["bull_pct"] = (recent_by_hour["bull_pct"] * 100).round(1)
    recent_by_hour = recent_by_hour.round(5)

    historical_by_hour = (
        df.groupby("hour")
        .agg(avg_range=("range", "mean"), bull_pct=("bull", "mean"))
        .reset_index().round(5)
    )

    regime_vs_hist = {
        "recent_avg_range": round(float(recent["range"].mean()), 4) if len(recent) else None,
        "historical_avg_range": round(overall_avg_range, 4),
        "delta_pct": round((float(recent["range"].mean()) / overall_avg_range - 1) * 100, 1)
                     if len(recent) else None,
        "recent_bull_pct": round(float(recent["bull"].mean()) * 100, 1) if len(recent) else None,
        "historical_bull_pct": round(float(df["bull"].mean()) * 100, 1),
        "lookback_days": lookback_days,
        "recent_bars": int(len(recent)),
    }

    # ── Today's actual bars so far (for "live tracking") ───────────────
    today_local = now.date()
    today_bars = df[df["date"] == today_local]
    today_summary = None
    if len(today_bars) > 0:
        today_summary = {
            "bars": int(len(today_bars)),
            "high": round(float(today_bars["high"].max()), 4),
            "low": round(float(today_bars["low"].min()), 4),
            "range": round(float(today_bars["high"].max() - today_bars["low"].min()), 4),
            "open": round(float(today_bars.iloc[0]["open"]), 4),
            "current": round(last_price, 4),
            "change_pct": round((last_price / float(today_bars.iloc[0]["open"]) - 1) * 100, 3),
            "bull_bars": int(today_bars["bull"].sum()),
            "bear_bars": int(today_bars["bear"].sum()),
        }

    return {
        "now": {
            "bkk_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": cur_hour,
            "day_name": cur_day_name,
            "session": session_for(cur_hour),
            "minutes_to_next_hour": int(minutes_to_next),
            "last_price": round(last_price, 4),
            "last_bar_time": str(last_bar_time),
            "thin_market": bool(is_thin),
            "dead_hour":   bool(is_dead),
            "monday_gap":  bool(is_mongap),
            "skip_bar":    bool(is_skip),
        },
        "market_flags": {
            "thin_hours":    sorted(THIN_HOURS),
            "dead_hours":    sorted(DEAD_HOURS),
            "mon_gap_hours": sorted(MON_GAP_HOURS),
            "note": (
                "thin_hours (4-6 AM BKK) = NY close / illiquid pre-reset. "
                "dead_hours (7 AM BKK) = Vantage MT5 daily reset — no bar generated. "
                "mon_gap_hours (8 AM BKK Mon) = first bar after weekend, gap risk."
            ),
        },
        "this_slot": this_slot,                # same DOW + same hour history
        "this_hour_overall": this_hour_overall,  # all DOWs for this hour
        "today_forecast": today_forecast,
        "next_high_vol": next_high_vol,
        "upcoming": upcoming,
        "regime": regime_vs_hist,
        "today_summary": today_summary,
        "recent_by_hour": recent_by_hour.to_dict(orient="records"),
        "historical_by_hour": historical_by_hour.to_dict(orient="records"),
        "sessions": SESSIONS,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    df, err = load_data()
    if err:
        return jsonify({"error": err}), 500
    enr = enrich(df)
    stats = compute_stats(enr)
    stats["meta"] = {
        "symbol": "XAUUSD.sc",
        "bars": len(df),
        "from": str(df["time"].dt.tz_convert(BKK).min()),
        "to": str(df["time"].dt.tz_convert(BKK).max()),
        "tz": "BKK (UTC+7)",
    }
    return jsonify(stats)


@app.route("/api/now")
def api_now():
    df, err = load_data()
    if err:
        return jsonify({"error": err}), 500
    enr = enrich(df)
    lookback = int(request.args.get("lookback", 30))
    payload = compute_now(enr, lookback_days=lookback)
    return jsonify(payload)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    import subprocess, sys
    script = os.path.join(os.path.dirname(__file__), "fetch_data.py")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr or result.stdout}), 500
    return jsonify({"ok": True, "output": result.stdout})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
