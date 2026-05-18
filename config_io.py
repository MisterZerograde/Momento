"""
Persistent user config for Momento (MT5 path, symbol, broker timezone, etc).
Shared between app.py and fetch_data.py.
"""
import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "mt5_path":          r"C:\Program Files\Vantage International MT5\terminal64.exe",
    "symbol":            "XAUUSD.sc",
    "bars":              50000,
    "max_history_from":  "2018-01-01", # earliest date MAX will pull from
    "broker_utc_offset": 3,            # Vantage server runs UTC+3
    "broker_name":       "Vantage",    # display only
    "configured":        False,        # flips True after the user saves once
}

ALLOWED_KEYS = set(DEFAULT_CONFIG.keys())


def load_config():
    """Load config.json. Backfills defaults for missing keys. Never raises."""
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return dict(DEFAULT_CONFIG)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)


def save_config(updates):
    """
    Merge `updates` into existing config and write atomically.
    Only keys in ALLOWED_KEYS are accepted. Returns the merged config.
    """
    cfg = load_config()
    for k, v in (updates or {}).items():
        if k in ALLOWED_KEYS:
            cfg[k] = v
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
    return cfg


def broker_tz_string(offset):
    """Convert hour offset (e.g. 3) to a pandas-compatible tz string.
    Note Etc/GMT signs are inverted: Etc/GMT-3 means UTC+3."""
    offset = int(offset)
    if offset == 0:
        return "UTC"
    sign = "-" if offset > 0 else "+"
    return f"Etc/GMT{sign}{abs(offset)}"


def market_hours_for_tz(broker_utc_offset, local_utc_offset=7):
    """
    Given the broker's UTC offset and the viewer's local UTC offset (default
    BKK=7), return (thin_hours, dead_hours, mon_gap_hours) as sets of BKK hours.

    Reasoning:
      dead_hour    = broker midnight in viewer local time (no bar generated)
      thin_hours   = the two hours preceding it (NY close + pre-rollover)
      mon_gap_hours = the hour right after, on Monday only (weekend gap)
    """
    dead_hour     = (24 - int(broker_utc_offset) + int(local_utc_offset)) % 24
    thin_hours    = {(dead_hour - 2) % 24, (dead_hour - 1) % 24}
    mon_gap_hours = {(dead_hour + 1) % 24}
    return thin_hours, {dead_hour}, mon_gap_hours


def detect_mt5_installations():
    """
    Scan common Windows install locations for `terminal64.exe`.
    Returns a list of {path, name} dicts.
    """
    candidates = []
    roots = []

    # Standard Program Files roots
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env)
        if root and os.path.isdir(root):
            roots.append(root)

    # Per-user AppData (some brokers install here)
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(appdata)
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        roots.append(localappdata)

    seen = set()
    for root in roots:
        try:
            for entry in os.listdir(root):
                folder = os.path.join(root, entry)
                if not os.path.isdir(folder):
                    continue
                exe = os.path.join(folder, "terminal64.exe")
                if os.path.isfile(exe) and exe.lower() not in seen:
                    seen.add(exe.lower())
                    candidates.append({"path": exe, "name": entry})
        except (PermissionError, OSError):
            continue
    return candidates
