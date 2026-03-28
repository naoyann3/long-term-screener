from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")
MARKET_CLOSE_CUTOFF = time(16, 0)
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS = PRICE_COLUMNS + ["Volume"]


def prepare_price_history(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return None

    hist = df.copy()
    if "Adj Close" not in hist.columns:
        hist["Adj Close"] = hist["Close"]
    if "Stock Splits" not in hist.columns:
        hist["Stock Splits"] = 0.0

    hist = hist[REQUIRED_COLUMNS + ["Adj Close", "Stock Splits"]].dropna(subset=REQUIRED_COLUMNS).copy()
    raw_close = hist["Close"].replace(0, pd.NA)
    hist["adjustment_factor"] = (hist["Adj Close"] / raw_close).replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)

    for col in PRICE_COLUMNS:
        hist[f"raw_{col.lower()}"] = hist[col]
        hist[col] = hist[col] * hist["adjustment_factor"]

    hist["Stock Splits"] = hist["Stock Splits"].fillna(0.0)
    return hist


def select_latest_completed_row(df: pd.DataFrame, now: datetime | None = None) -> tuple[pd.Series, datetime.date]:
    if df.empty:
        raise ValueError("price history is empty")

    if now is None:
        now_jst = datetime.now(JST)
    elif now.tzinfo is None:
        now_jst = now.replace(tzinfo=JST)
    else:
        now_jst = now.astimezone(JST)
    latest_pos = len(df) - 1
    latest_ts = pd.Timestamp(df.index[latest_pos])
    latest_date = latest_ts.date()

    if latest_date >= now_jst.date() and now_jst.time() < MARKET_CLOSE_CUTOFF and len(df) >= 2:
        latest_pos -= 1
        latest_ts = pd.Timestamp(df.index[latest_pos])
        latest_date = latest_ts.date()

    return df.iloc[latest_pos], latest_date


def adjusted_entry_price(entry_price: float | None, entry_date: str, hist: pd.DataFrame) -> float | None:
    if entry_price in (None, 0) or not entry_date:
        return entry_price

    try:
        entry_ts = pd.Timestamp(entry_date).normalize()
    except Exception:
        return entry_price

    normalized_index = pd.DatetimeIndex(hist.index)
    if normalized_index.tz is not None:
        normalized_index = normalized_index.tz_localize(None)
    normalized_index = normalized_index.normalize()
    splits = hist.loc[normalized_index > entry_ts, "Stock Splits"]
    split_factor = 1.0
    for ratio in splits:
        try:
            ratio_float = float(ratio)
        except Exception:
            continue
        if ratio_float > 0:
            split_factor *= ratio_float

    if split_factor == 0:
        return entry_price
    return entry_price / split_factor
