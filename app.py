import os, json, math
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "xauusd_1h.csv")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_data():
    if not os.path.exists(DATA_PATH):
        return None, "Data file not found. Run fetch_data.py first."
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df, None


BKK = "Asia/Bangkok"

def compute_stats(df):
    df = df.copy()
    local = df["time"].dt.tz_convert(BKK)
    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek                # 0=Mon … 6=Sun
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["move_pct"] = df["body"] / df["open"] * 100
    df["bull"] = (df["close"] > df["open"]).astype(int)
    df["bear"] = (df["close"] < df["open"]).astype(int)

    # ── per-hour aggregate (all days) ──────────────────────────────────────
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
    by_hour = by_hour.round(5)

    # ── per-day aggregate (all hours) ─────────────────────────────────────
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

    # ── heatmap: hour × day ───────────────────────────────────────────────
    pivot_range = (
        df.groupby(["dow", "hour"])["range"]
        .mean()
        .unstack(level="hour")
        .reindex(index=range(7), columns=range(24))
    )
    pivot_move = (
        df.groupby(["dow", "hour"])["move_pct"]
        .mean()
        .unstack(level="hour")
        .reindex(index=range(7), columns=range(24))
    )
    pivot_bull = (
        df.groupby(["dow", "hour"])["bull"]
        .mean()
        .unstack(level="hour")
        .reindex(index=range(7), columns=range(24))
    ) * 100
    pivot_count = (
        df.groupby(["dow", "hour"])["range"]
        .count()
        .unstack(level="hour")
        .reindex(index=range(7), columns=range(24))
    )

    def pivot_to_list(p):
        rows = p.round(4).values.tolist()
        return [[None if isinstance(v, float) and math.isnan(v) else v for v in row] for row in rows]

    # ── top 20 hour/day combos by avg range ───────────────────────────────
    hd = (
        df.groupby(["dow", "hour"])
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    df, err = load_data()
    if err:
        return jsonify({"error": err}), 500
    stats = compute_stats(df)
    # metadata
    stats["meta"] = {
        "symbol": "XAUUSD.sc",
        "bars": len(df),
        "from": str(df["time"].dt.tz_convert(BKK).min()),
        "to": str(df["time"].dt.tz_convert(BKK).max()),
        "tz": "BKK (UTC+7)",
    }
    return jsonify(stats)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Re-runs fetch_data.py to pull fresh bars from MT5."""
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
