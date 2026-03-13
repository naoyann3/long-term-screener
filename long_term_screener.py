from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

from config import LONG_TERM_SCREEN_VERSION, LONG_TERM_WATCHLISTS_DIR, ensure_results_dirs

TICKERS_CSV = "tickers.csv"
OUTPUT_CSV = "long_term_watchlist.csv"

MAX_TICKERS = 250
SLEEP_SEC = 0.8
TOP_N_OUTPUT = 50

MIN_TURNOVER = 100_000_000
MIN_MARKET_CAP = 30_000_000_000
MIN_REVENUE_GROWTH_PCT = 5.0
MIN_PROFIT_MARGIN_PCT = 5.0
MIN_ROE_PCT = 8.0
MAX_52W_HIGH_GAP_PCT = 20.0


def _ticker_path() -> Path:
    return Path(__file__).resolve().parent / TICKERS_CSV


def _latest_output_path() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_CSV


def load_tickers() -> pd.DataFrame:
    df = pd.read_csv(_ticker_path())
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    return df.head(MAX_TICKERS).reset_index(drop=True)


def fetch_price_history(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="18mo",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception as exc:
        print(f"fetch_price_history error: {ticker} {exc}")
        return None

    if df is None or df.empty or len(df) < 120:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in need_cols):
        return None

    df = df[need_cols].dropna().copy()
    return df


def fetch_fundamentals(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        print(f"fetch_fundamentals error: {ticker} {exc}")
        return None

    if not info:
        return None

    market_cap = info.get("marketCap")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    revenue_growth = info.get("revenueGrowth")
    current_ratio = info.get("currentRatio")
    debt_to_equity = info.get("debtToEquity")
    sector = info.get("sector")
    industry = info.get("industry")

    return {
        "market_cap": float(market_cap) if market_cap is not None else None,
        "roe_pct": float(roe) * 100 if roe is not None else None,
        "profit_margin_pct": float(profit_margin) * 100 if profit_margin is not None else None,
        "revenue_growth_pct": float(revenue_growth) * 100 if revenue_growth is not None else None,
        "current_ratio": float(current_ratio) if current_ratio is not None else None,
        "debt_to_equity": float(debt_to_equity) if debt_to_equity is not None else None,
        "sector": sector,
        "industry": industry,
    }


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma25"] = df["Close"].rolling(25).mean()
    df["ma75"] = df["Close"].rolling(75).mean()
    df["ma200"] = df["Close"].rolling(200).mean()
    df["ma25_slope_pct"] = (df["ma25"] - df["ma25"].shift(5)) / df["ma25"].shift(5) * 100
    df["ma75_slope_pct"] = (df["ma75"] - df["ma75"].shift(5)) / df["ma75"].shift(5) * 100
    df["ma200_slope_pct"] = (df["ma200"] - df["ma200"].shift(5)) / df["ma200"].shift(5) * 100
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["turnover"] = df["Close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000
    df["change_20d_pct"] = (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100
    df["change_60d_pct"] = (df["Close"] - df["Close"].shift(60)) / df["Close"].shift(60) * 100
    df["change_120d_pct"] = (df["Close"] - df["Close"].shift(120)) / df["Close"].shift(120) * 100
    df["recent_high_252"] = df["High"].rolling(252, min_periods=120).max()
    df["gap_to_52w_high_pct"] = (df["recent_high_252"] - df["Close"]) / df["Close"] * 100
    df["volume_ratio_20"] = df["Volume"] / df["vol_avg20"]
    return df


def passes_long_term_filter(latest: pd.Series, fundamentals: dict) -> bool:
    if latest["turnover"] < MIN_TURNOVER:
        return False

    if latest["Close"] < latest["ma75"]:
        return False

    if latest["ma25"] < latest["ma75"]:
        return False

    if pd.notna(latest["ma200"]) and latest["Close"] < latest["ma200"]:
        return False

    if latest["change_60d_pct"] < 0:
        return False

    if latest["gap_to_52w_high_pct"] > MAX_52W_HIGH_GAP_PCT:
        return False

    market_cap = fundamentals.get("market_cap")
    if market_cap is None or market_cap < MIN_MARKET_CAP:
        return False

    revenue_growth = fundamentals.get("revenue_growth_pct")
    if revenue_growth is None or revenue_growth < MIN_REVENUE_GROWTH_PCT:
        return False

    profit_margin = fundamentals.get("profit_margin_pct")
    if profit_margin is None or profit_margin < MIN_PROFIT_MARGIN_PCT:
        return False

    roe = fundamentals.get("roe_pct")
    if roe is None or roe < MIN_ROE_PCT:
        return False

    return True


def score_row(latest: pd.Series, fundamentals: dict) -> tuple[float, float, float, float]:
    trend_score = 0.0
    quality_score = 0.0
    strength_score = 0.0
    risk_penalty = 0.0

    if latest["Close"] >= latest["ma25"]:
        trend_score += 2.0
    if latest["Close"] >= latest["ma75"]:
        trend_score += 2.5
    if pd.notna(latest["ma200"]) and latest["Close"] >= latest["ma200"]:
        trend_score += 1.5
    if latest["ma25_slope_pct"] > 0:
        trend_score += 1.5
    if latest["ma75_slope_pct"] > 0:
        trend_score += 1.8
    if pd.notna(latest["ma200_slope_pct"]) and latest["ma200_slope_pct"] > 0:
        trend_score += 1.0

    revenue_growth = fundamentals.get("revenue_growth_pct") or 0.0
    profit_margin = fundamentals.get("profit_margin_pct") or 0.0
    roe = fundamentals.get("roe_pct") or 0.0
    quality_score += min(revenue_growth, 30.0) * 0.12
    quality_score += min(profit_margin, 25.0) * 0.15
    quality_score += min(roe, 25.0) * 0.12

    strength_score += min(max(latest["change_20d_pct"], 0.0), 30.0) * 0.08
    strength_score += min(max(latest["change_60d_pct"], 0.0), 50.0) * 0.08
    strength_score += min(max(20.0 - latest["gap_to_52w_high_pct"], 0.0), 20.0) * 0.15
    strength_score += min(max(latest["volume_ratio_20"], 0.0), 3.0) * 0.5

    debt_to_equity = fundamentals.get("debt_to_equity")
    if debt_to_equity is not None and debt_to_equity > 150:
        risk_penalty -= 1.5
    if latest["change_20d_pct"] > 35:
        risk_penalty -= 1.0
    if latest["volume_ratio_20"] > 4:
        risk_penalty -= 0.8

    total = trend_score + quality_score + strength_score + risk_penalty
    return round(total, 2), round(trend_score, 2), round(quality_score, 2), round(strength_score + risk_penalty, 2)


def run():
    ensure_results_dirs()
    tickers = load_tickers()
    rows: list[dict] = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    screen_date = None

    for idx, row in tickers.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        print(f"{idx + 1}/{len(tickers)} {ticker}")

        hist = fetch_price_history(ticker)
        if hist is None:
            continue

        fundamentals = fetch_fundamentals(ticker)
        if fundamentals is None:
            continue

        hist = calc_indicators(hist)
        latest = hist.iloc[-2]
        latest_date = pd.Timestamp(hist.index[-2]).date()

        if not passes_long_term_filter(latest, fundamentals):
            time.sleep(SLEEP_SEC)
            continue

        screen_date = latest_date if screen_date is None else max(screen_date, latest_date)
        score, trend_score, quality_score, strength_score = score_row(latest, fundamentals)

        rows.append(
            {
                "run_date": latest_date.isoformat(),
                "screen_version": LONG_TERM_SCREEN_VERSION,
                "generated_at": generated_at,
                "ticker": ticker,
                "name": name,
                "score": score,
                "trend_score": trend_score,
                "quality_score": quality_score,
                "strength_score": strength_score,
                "close": round(float(latest["Close"]), 3),
                "turnover_million": round(float(latest["turnover_million"]), 3),
                "market_cap_billion": round((fundamentals["market_cap"] or 0.0) / 1_000_000_000, 3),
                "revenue_growth_pct": round(fundamentals["revenue_growth_pct"], 3),
                "profit_margin_pct": round(fundamentals["profit_margin_pct"], 3),
                "roe_pct": round(fundamentals["roe_pct"], 3),
                "current_ratio": round(fundamentals["current_ratio"], 3) if fundamentals["current_ratio"] is not None else None,
                "debt_to_equity": round(fundamentals["debt_to_equity"], 3) if fundamentals["debt_to_equity"] is not None else None,
                "change_20d_pct": round(float(latest["change_20d_pct"]), 3),
                "change_60d_pct": round(float(latest["change_60d_pct"]), 3),
                "change_120d_pct": round(float(latest["change_120d_pct"]), 3),
                "gap_to_52w_high_pct": round(float(latest["gap_to_52w_high_pct"]), 3),
                "volume_ratio_20": round(float(latest["volume_ratio_20"]), 3),
                "ma25_slope_pct": round(float(latest["ma25_slope_pct"]), 3),
                "ma75_slope_pct": round(float(latest["ma75_slope_pct"]), 3),
                "ma200_slope_pct": round(float(latest["ma200_slope_pct"]), 3) if pd.notna(latest["ma200_slope_pct"]) else None,
                "sector": fundamentals["sector"],
                "industry": fundamentals["industry"],
            }
        )
        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No long-term candidates")
        return

    df = df.sort_values(["score", "quality_score", "trend_score"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    output_df = df[
        [
            "run_date",
            "screen_version",
            "generated_at",
            "rank",
            "ticker",
            "name",
            "score",
            "trend_score",
            "quality_score",
            "strength_score",
            "close",
            "turnover_million",
            "market_cap_billion",
            "revenue_growth_pct",
            "profit_margin_pct",
            "roe_pct",
            "current_ratio",
            "debt_to_equity",
            "change_20d_pct",
            "change_60d_pct",
            "change_120d_pct",
            "gap_to_52w_high_pct",
            "volume_ratio_20",
            "ma25_slope_pct",
            "ma75_slope_pct",
            "ma200_slope_pct",
            "sector",
            "industry",
        ]
    ].head(TOP_N_OUTPUT)

    if screen_date is None:
        screen_date = pd.Timestamp(output_df["run_date"].max()).date()

    latest_output_path = _latest_output_path()
    dated_output_path = LONG_TERM_WATCHLISTS_DIR / f"{screen_date.isoformat()}_{LONG_TERM_SCREEN_VERSION}.csv"
    output_df.to_csv(latest_output_path, index=False, encoding="utf-8-sig")
    output_df.to_csv(dated_output_path, index=False, encoding="utf-8-sig")

    print("\n==== Long Term Watchlist ====")
    print(output_df.to_string(index=False))
    print(f"\nCSV出力完了: {latest_output_path.name}")
    print(f"履歴保存完了: {dated_output_path}")


if __name__ == "__main__":
    run()
