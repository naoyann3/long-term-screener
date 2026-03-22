from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from output_format import format_long_term_tracking_output

TRACKED_TICKERS_CSV = "tracked_tickers.csv"
OUTPUT_CSV = "long_term_tracking.csv"
OUTPUT_DIR = "results/long_term_tracking"


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def _tracked_path() -> Path:
    return _base_dir() / TRACKED_TICKERS_CSV


def _latest_output_path() -> Path:
    return _base_dir() / OUTPUT_CSV


def _history_dir() -> Path:
    return _base_dir() / OUTPUT_DIR


def ensure_dirs() -> None:
    _history_dir().mkdir(parents=True, exist_ok=True)


def ensure_template() -> None:
    path = _tracked_path()
    if path.exists():
        return

    template = pd.DataFrame(
        columns=["ticker", "name", "entry_date", "entry_price", "note"]
    )
    template.to_csv(path, index=False, encoding="utf-8-sig")


def load_tracked_tickers() -> pd.DataFrame:
    ensure_template()
    df = pd.read_csv(_tracked_path())
    if df.empty:
        return df

    if "ticker" not in df.columns:
        raise ValueError("tracked_tickers.csv に ticker 列が必要です")

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] != ""].copy()
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    if "entry_date" not in df.columns:
        df["entry_date"] = ""
    if "entry_price" not in df.columns:
        df["entry_price"] = ""
    if "note" not in df.columns:
        df["note"] = ""
    return df.reset_index(drop=True)


def fetch_history(ticker: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period="18mo", interval="1d", auto_adjust=True)
    except Exception as exc:
        print(f"fetch_history error: {ticker} {exc}")
        return None

    if hist is None or hist.empty or len(hist) < 200:
        return None

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in hist.columns for col in need_cols):
        return None

    return hist[need_cols].dropna().copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma25"] = df["Close"].rolling(25).mean()
    df["ma75"] = df["Close"].rolling(75).mean()
    df["ma200"] = df["Close"].rolling(200).mean()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio_20"] = df["Volume"] / df["vol_avg20"]
    df["change_20d_pct"] = (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100
    df["change_60d_pct"] = (df["Close"] - df["Close"].shift(60)) / df["Close"].shift(60) * 100
    df["high_60"] = df["High"].rolling(60).max()
    df["drawdown_from_60d_high_pct"] = (df["Close"] - df["high_60"]) / df["high_60"] * 100
    return df


def pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return (numerator - denominator) / denominator * 100


def upper_shadow_pct(latest: pd.Series) -> float:
    high = float(latest["High"])
    low = float(latest["Low"])
    open_ = float(latest["Open"])
    close = float(latest["Close"])
    day_range = high - low
    if day_range <= 0:
        return 0.0
    body_top = max(open_, close)
    return max(high - body_top, 0.0) / day_range * 100


def judge_status(latest: pd.Series) -> tuple[str, int, list[str]]:
    flags: list[str] = []
    score = 0

    if latest["Close"] < latest["ma25"]:
        score += 1
        flags.append("終値が25日線割れ")
    if latest["Close"] < latest["ma75"]:
        score += 2
        flags.append("終値が75日線割れ")
    if pd.notna(latest["ma200"]) and latest["Close"] < latest["ma200"]:
        score += 2
        flags.append("終値が200日線割れ")
    if latest["ma25"] < latest["ma75"]:
        score += 2
        flags.append("25日線が75日線を下回る")
    if pd.notna(latest["ma200"]) and latest["ma75"] < latest["ma200"]:
        score += 2
        flags.append("75日線が200日線を下回る")
    if latest["drawdown_from_60d_high_pct"] <= -12:
        score += 2
        flags.append("60日高値から大きく下落")
    elif latest["drawdown_from_60d_high_pct"] <= -8:
        score += 1
        flags.append("60日高値から下落")
    if latest["change_20d_pct"] < -8:
        score += 1
        flags.append("20日騰落率が悪化")
    if latest["volume_ratio_20"] >= 2.5 and upper_shadow_pct(latest) >= 45:
        score += 1
        flags.append("出来高急増+長い上ヒゲ")

    if score >= 6:
        status = "撤退候補"
    elif score >= 3:
        status = "警戒"
    elif flags:
        status = "継続(注意)"
    else:
        status = "継続"
    return status, score, flags


def run() -> None:
    ensure_dirs()
    tracked = load_tracked_tickers()
    if tracked.empty:
        print("tracked_tickers.csv を作成しました。監視したい銘柄を入力してください。")
        return

    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: list[dict] = []
    run_date = None

    for _, row in tracked.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        hist = fetch_history(ticker)
        if hist is None:
            continue
        hist = add_indicators(hist)
        latest = hist.iloc[-2]
        latest_date = pd.Timestamp(hist.index[-2]).date()
        run_date = latest_date if run_date is None else max(run_date, latest_date)

        status, status_score, flags = judge_status(latest)

        try:
            entry_price = float(row["entry_price"]) if str(row["entry_price"]).strip() else None
        except Exception:
            entry_price = None

        rows.append(
            {
                "generated_at": generated_at,
                "ticker": ticker,
                "name": name,
                "entry_date": row["entry_date"],
                "entry_price": entry_price,
                "close": round(float(latest["Close"]), 3),
                "close_vs_entry_pct": round(pct(float(latest["Close"]), entry_price), 3) if entry_price else None,
                "close_vs_ma25_pct": round(pct(float(latest["Close"]), float(latest["ma25"])), 3) if pd.notna(latest["ma25"]) else None,
                "close_vs_ma75_pct": round(pct(float(latest["Close"]), float(latest["ma75"])), 3) if pd.notna(latest["ma75"]) else None,
                "close_vs_ma200_pct": round(pct(float(latest["Close"]), float(latest["ma200"])), 3) if pd.notna(latest["ma200"]) else None,
                "ma25_vs_ma75_pct": round(pct(float(latest["ma25"]), float(latest["ma75"])), 3) if pd.notna(latest["ma25"]) and pd.notna(latest["ma75"]) else None,
                "ma75_vs_ma200_pct": round(pct(float(latest["ma75"]), float(latest["ma200"])), 3) if pd.notna(latest["ma75"]) and pd.notna(latest["ma200"]) else None,
                "change_20d_pct": round(float(latest["change_20d_pct"]), 3),
                "change_60d_pct": round(float(latest["change_60d_pct"]), 3),
                "drawdown_from_60d_high_pct": round(float(latest["drawdown_from_60d_high_pct"]), 3),
                "volume_ratio_20": round(float(latest["volume_ratio_20"]), 3),
                "upper_shadow_pct": round(upper_shadow_pct(latest), 3),
                "status": status,
                "status_score": status_score,
                "warning_flags": " / ".join(flags),
                "note": row["note"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No tracking rows")
        return

    df = df.sort_values(["status_score", "drawdown_from_60d_high_pct"], ascending=[False, True]).reset_index(drop=True)
    display_df = format_long_term_tracking_output(df)
    display_df.to_csv(_latest_output_path(), index=False, encoding="utf-8-sig")

    if run_date is None:
        run_date = datetime.now().date()
    history_path = _history_dir() / f"{run_date.isoformat()}_tracking.csv"
    display_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    print(display_df.to_string(index=False))
    print(f"\nTracking CSV saved: {_latest_output_path()}")
    print(f"Tracking history saved: {history_path}")


if __name__ == "__main__":
    run()
