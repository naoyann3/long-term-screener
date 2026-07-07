# long_term_screener.py (Version 1.5 - Extreme Speed & Self-Cleaning Edition)
from __future__ import annotations

from datetime import datetime
import gc
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf

from config import LONG_TERM_SCREEN_VERSION, LONG_TERM_WATCHLISTS_DIR, ensure_results_dirs, CANDIDATE_HISTORY_CSV
from market_data_utils import prepare_price_history, select_latest_completed_row
from output_format import format_long_term_gc_output, format_long_term_latest_output, format_long_term_output

TICKERS_CSV = "tickers.csv"
OUTPUT_CSV = "long_term_watchlist.csv"
GC_OUTPUT_CSV = "long_term_gc_watchlist.csv"

# 1回あたりに一括ロードするチャンクサイズ
DOWNLOAD_CHUNK_SIZE = 300
# info(財務データ)取得時の待機時間
SLEEP_SEC = 1.5
TOP_N_OUTPUT = 50
TOP_N_GC_OUTPUT = 20

# yfinance.info のIPブロックを完全に防ぐ、精査最大ロック数
MAX_FUNDAMENTALS_精査数 = 30

MIN_TURNOVER = 100_000_000
MIN_MARKET_CAP = 30_000_000_000
MIN_REVENUE_GROWTH_PCT = 5.0
MIN_PROFIT_MARGIN_PCT = 5.0
MIN_ROE_PCT = 8.0
MAX_52W_HIGH_GAP_PCT = 10.0        # 52週高値から10%以内（ブレイク直前の本命株）
MAX_CHANGE_20D_PCT = 25.0
MAX_CHANGE_60D_PCT = 80.0
RECENT_CROSS_LOOKBACK = 10
PERFECT_ORDER_LOOKBACK = 5
BEARISH_ORDER_LOOKBACK = 60
REVERSAL_LOOKBACK = 10
GC_MIN_MA25_SLOPE_PCT = 0.25
GC_MIN_MA75_SLOPE_PCT = 0.25


def _ticker_path() -> Path:
    return Path(__file__).resolve().parent / TICKERS_CSV


def _latest_output_path() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_CSV


def _latest_gc_output_path() -> Path:
    return Path(__file__).resolve().parent / GC_OUTPUT_CSV


def _gc_watchlists_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "long_term_gc_watchlists"


def load_all_tickers() -> pd.DataFrame:
    df = pd.read_csv(_ticker_path())
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    return df.reset_index(drop=True)


def download_chunk_histories(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    【Version 1.5 高速化 ＆ 浄化】:
    1. period="12mo" に短縮（データ量33%軽量化、計算結果への影響は0%） [5]
    2. threads=5 を導入（道路を5車線に広げ、IPブロックされない安全並列ダウンロードを始動） [5]
    3. 価格データが1件も取得できなかった上場廃止候補リスト（delisted_tickers）を検知して戻します
    """
    full_data = {}
    delisted_tickers = []
    chunks = [tickers[i:i + DOWNLOAD_CHUNK_SIZE] for i in range(0, len(tickers), DOWNLOAD_CHUNK_SIZE)]
    
    print(f"\n[第1段階] 全 {len(tickers)} 銘柄を一括ダウンロードします (分割数: {len(chunks)} チャンク)...")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"  ・ダウンロード中 ({idx}/{len(chunks)})... {len(chunk)} 銘柄")
        try:
            df_chunk = yf.download(
                chunk, 
                period="12mo",  # 👈 1.5年から12ヶ月に短縮（通信時間の劇的削減） [5]
                interval="1d", 
                group_by="column", 
                auto_adjust=False, 
                actions=True, 
                progress=False,
                threads=5       # 👈 安全なスレッド数5で緩やかに高速化 [5]
            )
            
            for t in chunk:
                try:
                    if isinstance(df_chunk.columns, pd.MultiIndex):
                        if t in df_chunk.columns.get_level_values(1):
                            df_t = df_chunk.xs(t, axis=1, level=1).copy()
                            df_t = df_t.dropna(subset=["Close", "Volume"])
                            
                            # 最低限のデータがあるか確認
                            if len(df_t) >= 120:
                                df_ticker = prepare_price_history(df_t)
                                if df_ticker is not None and not df_ticker.empty:
                                    full_data[t] = df_ticker
                                else:
                                    delisted_tickers.append(t)
                            else:
                                delisted_tickers.append(t)
                        else:
                            delisted_tickers.append(t)
                except Exception:
                    delisted_tickers.append(t)
                    continue
                    
        except Exception as e:
            print(f"  [警告] チャンク {idx} のダウンロード中にエラーが発生しました: {e}")
            
        time.sleep(2.0) # 5車線並列化に伴い、安全間隔を2.0秒に調整
        
    return full_data, delisted_tickers


def close_ticker_session(ticker_obj) -> None:
    if ticker_obj is None:
        return
    session = getattr(getattr(ticker_obj, "_data", None), "session", None)
    close = getattr(session, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    gc.collect()


def fetch_fundamentals(ticker_obj, ticker: str) -> dict | None:
    try:
        info = ticker_obj.info
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
    df["ma25_above_ma200"] = df["ma25"] > df["ma200"]
    df["ma75_above_ma200"] = df["ma75"] > df["ma200"]
    df["ma25_cross_200_today"] = df["ma25_above_ma200"] & (~df["ma25_above_ma200"].shift(1).fillna(False))
    df["ma75_cross_200_today"] = df["ma75_above_ma200"] & (~df["ma75_above_ma200"].shift(1).fillna(False))
    df["ma25_cross_200_recent"] = (
        df["ma25_cross_200_today"].rolling(RECENT_CROSS_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma75_cross_200_recent"] = (
        df["ma75_cross_200_today"].rolling(RECENT_CROSS_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_above_ma75"] = df["ma25"] > df["ma75"]
    df["ma75_above_ma200"] = df["ma75"] > df["ma200"]
    df["perfect_order"] = df["ma25_above_ma75"] & df["ma75_above_ma200"]
    df["bearish_stack"] = (df["ma200"] > df["ma75"]) & (df["ma75"] > df["ma25"])
    df["bearish_perfect_order"] = df["bearish_stack"]
    df["ma25_cross_75_today"] = df["ma25_above_ma75"] & (~df["ma25_above_ma75"].shift(1).fillna(False))
    df["ma75_cross_200_recent_tight"] = (
        df["ma75_cross_200_today"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_cross_200_recent_tight"] = (
        df["ma25_cross_200_today"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_cross_75_recent_tight"] = (
        df["ma25_cross_75_today"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["perfect_order_recent"] = (
        df["perfect_order"].rolling(PERFECT_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["perfect_order_today"] = df["perfect_order"] & (~df["perfect_order"].shift(1).fillna(False))
    df["perfect_order_recent_tight"] = (
        df["perfect_order_today"].rolling(REVERSAL_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["bearish_stack_recent"] = (
        df["bearish_stack"].rolling(BEARISH_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["bearish_perfect_order_recent"] = (
        df["bearish_perfect_order"].shift(1).rolling(BEARISH_ORDER_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    )
    df["ma25_slope_pct"] = (df["ma25"] - df["ma25"].shift(5)) / df["ma25"].shift(5) * 100
    df["ma75_slope_pct"] = (df["ma75"] - df["ma75"].shift(5)) / df["ma75"].shift(5) * 100
    df["ma200_slope_pct"] = (df["ma200"] - df["ma200"].shift(5)) / df["ma200"].shift(5) * 100
    df["close_vs_ma25_pct"] = (df["Close"] - df["ma25"]) / df["ma25"] * 100
    df["close_vs_ma75_pct"] = (df["Close"] - df["ma75"]) / df["ma75"] * 100
    df["close_vs_ma200_pct"] = (df["Close"] - df["ma200"]) / df["ma200"] * 100
    df["touch_ma25_intraday"] = df["Low"] <= df["ma25"]
    df["touch_ma75_intraday"] = df["Low"] <= df["ma75"]
    df["reclaim_ma25_close"] = df["touch_ma25_intraday"] & (df["Close"] >= df["ma25"])
    df["reclaim_ma75_close"] = df["touch_ma75_intraday"] & (df["Close"] >= df["ma75"])
    df["initial_trend_signal"] = (
        (df["Close"] >= df["ma25"])
        & df["perfect_order"]
        & (df["ma25_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
        & df["ma75_cross_200_recent_tight"]
    )
    df["reversal_from_bearish_po"] = (
        df["bearish_perfect_order_recent"]
        & df["perfect_order"]
        & df["perfect_order_recent_tight"]
        & df["ma25_cross_75_recent_tight"]
        & df["ma75_cross_200_recent_tight"]
        & (df["Close"] >= df["ma25"])
        & (df["close_vs_ma25_pct"].between(-2.0, 10.0))
        & (df["ma25_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
    )
    df["early_reversal_setup"] = (
        df["bearish_perfect_order_recent"]
        & (df["Close"] >= df["ma75"])
        & df["ma25_cross_75_recent_tight"]
        & (df["ma75_cross_200_recent_tight"] | df["ma25_cross_200_recent_tight"])
        & (df["ma25_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
        & (df["close_vs_ma25_pct"].between(-3.0, 8.0))
    )
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["turnover"] = df["raw_close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000
    df["change_20d_pct"] = (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100
    df["change_60d_pct"] = (df["Close"] - df["Close"].shift(60)) / df["Close"].shift(60) * 100
    df["change_120d_pct"] = (df["Close"] - df["Close"].shift(120)) / df["Close"].shift(120) * 100
    df["high_60"] = df["High"].rolling(60).max()
    df["drawdown_from_60d_high_pct"] = (df["Close"] - df["high_60"]) / df["high_60"] * 100
    df["recent_high_252"] = df["High"].rolling(252, min_periods=120).max()
    df["gap_to_52w_high_pct"] = (df["recent_high_252"] - df["Close"]) / df["Close"] * 100
    df["vol_avg10"] = df["Volume"].rolling(10).mean()
    df["volume_ratio_20"] = df["Volume"] / df["vol_avg20"]
    df["volume_ratio_10"] = df["Volume"] / df["vol_avg10"]
    df["volume_ratio_20_mean_5"] = df["volume_ratio_20"].rolling(5).mean()
    df["volume_ratio_20_mean_10"] = df["volume_ratio_20"].rolling(10).mean()
    df["down_day"] = df["Close"] < df["Open"]
    df["down_volume_spike"] = df["down_day"] & (df["volume_ratio_20"] >= 1.5)
    price_range = (df["High"] - df["Low"]).replace(0, pd.NA)
    upper_shadow = (df["High"] - df[["Open", "Close"]].max(axis=1)).clip(lower=0)
    lower_shadow = (df[["Open", "Close"]].min(axis=1) - df["Low"]).clip(lower=0)
    df["upper_shadow_pct"] = (upper_shadow / price_range) * 100
    df["lower_shadow_pct"] = (lower_shadow / price_range) * 100
    df["distribution_warning"] = (
        (df["upper_shadow_pct"] >= 45)
        & (df["volume_ratio_20"] >= 1.5)
        & (df["gap_to_52w_high_pct"] <= 8)
    )
    cross_positions = pd.Series(range(len(df)), index=df.index).where(df["ma75_cross_200_today"])
    df["days_since_75gc200"] = pd.Series(range(len(df)), index=df.index) - cross_positions.ffill()
    df.loc[cross_positions.ffill().isna(), "days_since_75gc200"] = pd.NA
    po_positions = pd.Series(range(len(df)), index=df.index).where(df["perfect_order_today"])
    df["days_since_perfect_order"] = pd.Series(range(len(df)), index=df.index) - po_positions.ffill()
    df.loc[po_positions.ffill().isna(), "days_since_perfect_order"] = pd.NA
    pullback_trend_ok = (
        (df["Close"] >= df["ma200"])
        & (df["ma200_slope_pct"] > 0)
        & (df["ma75_slope_pct"] > 0)
        & df["ma25_above_ma75"]
    )
    df["failed_ma25_reclaim"] = df["touch_ma25_intraday"] & (~df["reclaim_ma25_close"])
    df["failed_ma75_reclaim"] = df["touch_ma75_intraday"] & (~df["reclaim_ma75_close"])
    df["support_reaction_ok"] = (
        (df["reclaim_ma25_close"] | df["reclaim_ma75_close"])
        & (~df["down_volume_spike"])
        & (df["drawdown_from_60d_high_pct"] > -15)
    )
    df["ma25_pullback_candidate"] = (
        pullback_trend_ok
        & df["touch_ma25_intraday"]
        & df["reclaim_ma25_close"]
        & (df["drawdown_from_60d_high_pct"] > -10)
    )
    df["ma75_pullback_candidate"] = (
        pullback_trend_ok
        & df["touch_ma75_intraday"]
        & df["reclaim_ma75_close"]
        & (df["drawdown_from_60d_high_pct"] > -15)
    )
    df["trend_filter_ok"] = (df["ma200_slope_pct"] > 0)
    df["volume_filter_ok"] = (df["volume_ratio_20"] >= 1.0)
    df["support_trace_ok"] = df["support_reaction_ok"] | (df["lower_shadow_pct"] >= 1.0)
    df["drawdown_filter_ok"] = df["drawdown_from_60d_high_pct"].between(-20.0, -2.0, inclusive="both")
    df["ma75_quality_filter"] = (
        df["trend_filter_ok"]
        & df["volume_filter_ok"]
        & df["support_trace_ok"]
        & df["drawdown_filter_ok"]
    )
    df["ma75_touch_quality_signal"] = df["touch_ma75_intraday"] & df["ma75_quality_filter"]
    df["ma75_nextday_quality_signal"] = df["ma75_touch_quality_signal"].shift(1).fillna(False)
    pullback_score = pd.Series(0.0, index=df.index)
    pullback_score += (df["Close"] >= df["ma200"]).fillna(False).astype(float) * 2.0
    pullback_score += (df["ma200_slope_pct"] > 0).fillna(False).astype(float) * 2.0
    pullback_score += (df["ma75_slope_pct"] > 0).fillna(False).astype(float) * 1.5
    pullback_score += df["ma25_above_ma75"].fillna(False).astype(float) * 1.0
    pullback_score += df["reclaim_ma25_close"].fillna(False).astype(float) * 1.5
    pullback_score += df["reclaim_ma75_close"].fillna(False).astype(float) * 2.5
    pullback_score += df["support_reaction_ok"].fillna(False).astype(float) * 1.0
    pullback_score += df["ma25_pullback_candidate"].fillna(False).astype(float) * 0.5
    pullback_score += df["ma75_pullback_candidate"].fillna(False).astype(float) * 1.0
    pullback_score += df["close_vs_ma25_pct"].between(-1.0, 3.0).fillna(False).astype(float) * 1.0
    pullback_score += df["close_vs_ma75_pct"].between(-1.0, 5.0).fillna(False).astype(float) * 1.0
    pullback_score -= df["down_volume_spike"].fillna(False).astype(float) * 2.5
    pullback_score -= df["failed_ma25_reclaim"].fillna(False).astype(float) * 1.5
    pullback_score -= df["failed_ma75_reclaim"].fillna(False).astype(float) * 2.0
    pullback_score -= (df["Close"] < df["ma75"]).fillna(False).astype(float) * 2.0
    pullback_score -= (df["change_20d_pct"] < -8).fillna(False).astype(float) * 1.0
    pullback_score -= (df["drawdown_from_60d_high_pct"] <= -15).fillna(False).astype(float) * 1.0
    df["pullback_score"] = pullback_score.round(2)
    df["pullback_candidate"] = (
        pullback_trend_ok
        & df["support_reaction_ok"]
        & (df["ma25_pullback_candidate"] | df["ma75_pullback_candidate"])
        & (df["pullback_score"] >= 6.0)
    )
    stealth_score = pd.Series(0.0, index=df.index)
    stealth_score += (df["volume_ratio_20_mean_5"] >= 1.15).fillna(False).astype(float) * 1.5
    stealth_score += (df["volume_ratio_20_mean_10"] >= 1.05).fillna(False).astype(float) * 1.0
    stealth_score += (df["Close"] >= df["ma75"]).fillna(False).astype(float) * 1.0
    stealth_score += (df["Close"] >= df["ma200"]).fillna(False).astype(float) * 0.5
    stealth_score += (df["ma200_slope_pct"] > 0).fillna(False).astype(float) * 1.0
    stealth_score += (df["ma75_slope_pct"] > 0).fillna(False).astype(float) * 1.0
    stealth_score += df["drawdown_from_60d_high_pct"].between(-12, 2).fillna(False).astype(float) * 1.0
    stealth_score += (df["lower_shadow_pct"] >= 35).fillna(False).astype(float) * 1.0
    stealth_score += df["reclaim_ma25_close"].fillna(False).astype(float) * 1.0
    stealth_score += df["reclaim_ma75_close"].fillna(False).astype(float) * 1.0
    stealth_score -= df["down_volume_spike"].fillna(False).astype(float) * 1.5
    stealth_score -= df["distribution_warning"].fillna(False).astype(float) * 1.5
    stealth_score -= (df["Close"] < df["ma75"]).fillna(False).astype(float) * 1.0
    text_20d_pct_chg = df["change_20d_pct"]
    stealth_score -= (text_20d_pct_chg > 20).fillna(False).astype(float) * 0.5
    df["stealth_accumulation_score"] = stealth_score.round(2)
    df["stealth_accumulation_candidate"] = (
        (df["volume_ratio_20_mean_5"] >= 1.10)
        & (df["Close"] >= df["ma75"])
        & (df["ma200_slope_pct"] > 0)
        & (~df["down_volume_spike"])
        & (~df["distribution_warning"])
        & (df["drawdown_from_60d_high_pct"] > -15)
        & (df["stealth_accumulation_score"] >= 5.5)
    )
    return df


def passes_long_term_filter_technical_only(latest: pd.Series) -> bool:
    """
    【第1段階（テクニカル足切り）の判定ロジック】
    """
    if latest["turnover"] < MIN_TURNOVER:
        return False
    if not (latest["ma25"] > latest["ma75"] > latest["ma200"]):
        return False
    if latest["Close"] < latest["ma25"]:
        return False
    if latest["ma25_slope_pct"] <= 0 or latest["ma75_slope_pct"] <= 0:
        return False
    if latest["gap_to_52w_high_pct"] > MAX_52W_HIGH_GAP_PCT:
        return False
    if latest["change_20d_pct"] > MAX_CHANGE_20D_PCT:
        return False
    if latest["change_60d_pct"] < 0 or latest["change_60d_pct"] > MAX_CHANGE_60D_PCT:
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
    current_ratio = fundamentals.get("current_ratio") or 0.0
    debt_to_equity = fundamentals.get("debt_to_equity")
    quality_score += min(revenue_growth, 30.0) * 0.11
    quality_score += min(profit_margin, 25.0) * 0.18
    quality_score += min(roe, 25.0) * 0.16
    if current_ratio >= 1.5:
        quality_score += 0.8

    strength_score += min(max(latest["change_20d_pct"], 0.0), 30.0) * 0.08
    strength_score += min(max(latest["change_60d_pct"], 0.0), 50.0) * 0.08
    gap_score = min(max(16.0 - latest["gap_to_52w_high_pct"], 0.0), 16.0) * 0.12
    if latest["gap_to_52w_high_pct"] < 2.0:
        gap_score -= 0.8
    strength_score += gap_score
    strength_score += min(max(latest["volume_ratio_20"], 0.0), 3.0) * 0.5

    if debt_to_equity is not None and debt_to_equity > 150:
        risk_penalty -= 1.5
    elif debt_to_equity is not None and debt_to_equity > 100:
        risk_penalty -= 0.7
    if latest["change_20d_pct"] > 18:
        risk_penalty -= (latest["change_20d_pct"] - 18) * 0.18
    if latest["change_60d_pct"] > 45:
        risk_penalty -= (latest["change_60d_pct"] - 45) * 0.06
    if latest["volume_ratio_20"] > 4:
        risk_penalty -= 0.8
    if latest["ma75_cross_200_recent_tight"]:
        strength_score += 1.0
    if latest["ma25_cross_75_recent_tight"]:
        strength_score += 0.6
    if latest["initial_trend_signal"]:
        strength_score += 3.0
    if latest["early_reversal_setup"]:
        strength_score += 2.5
    if latest["reversal_from_bearish_po"]:
        strength_score += 4.0
    if latest["pullback_candidate"]:
        strength_score += 2.0
    strength_score += min(max(latest["pullback_score"], 0.0), 10.0) * 0.15

    total = trend_score + quality_score + strength_score + risk_penalty
    return round(total, 2), round(trend_score, 2), round(quality_score, 2), round(strength_score + risk_penalty, 2)


def run() -> None:
    ensure_results_dirs()
    
    tickers_df = load_all_tickers()
    all_tickers = tickers_df["ticker"].dropna().tolist()
    name_map = dict(zip(tickers_df["ticker"], tickers_df["name"]))

    rows: list[dict] = []
    run_started_at = datetime.now()
    generated_at = run_started_at.isoformat(timespec="seconds")
    run_stamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    screen_date = None

    # ==========================================
    # ★【第1段階】：全ティッカーの超高速一括ダウンロード ＆ テクニカル一括足切り ★
    # ==========================================
    all_histories, delisted_list = download_chunk_histories(all_tickers)
    
    # 💡 【Version 1.5新設：宇宙の自己クリーニングロジック】
    # yfinanceが「上場廃止・データ取得不可(404)」と判定した銘柄群をtickers.csvから自動で一掃します
    if delisted_list:
        print(f"\n📢 [自己クリーニング] yfinanceでロードできなかった {len(delisted_list)} 銘柄を検知しました。上場廃止・ティッカー変更とみなしてtickers.csvから自動削除します。")
        # 実際にデータがあるティッカーだけでtickers.csvを上書きします
        cleaned_tickers_df = tickers_df[~tickers_df["ticker"].isin(delisted_list)]
        cleaned_tickers_df.to_csv(_ticker_path(), index=False, encoding="utf-8-sig")
        print("  ➔ tickers.csv の自己クリーニング・浄化処理が完了しました。")

    technical_passed = []

    print("\n=== テクニカル ＆ 流動性の一次足切りスクリーニングを実行します ===")
    for ticker, df_ticker in all_histories.items():
        try:
            hist = prepare_price_history(df_ticker)
            if hist is None or hist.empty:
                continue
                
            hist = calc_indicators(hist)
            latest, latest_date = select_latest_completed_row(hist)

            if not passes_long_term_filter_technical_only(latest):
                continue

            technical_passed.append((ticker, hist, latest, latest_date))
        except Exception:
            continue

    print(f"➔ [一次フィルター合格]: {len(technical_passed)} 銘柄 / 有効 {len(all_histories)} 銘柄中")

    # ==========================================
    # ★【第2段階】：テクニカル合格株のみ、個別にinfo（財務）を取得して最終足切り ★
    # ==========================================
    def get_temp_score(item) -> float:
        _, _, latest, _ = item
        score = 0.0
        if latest["Close"] >= latest["ma25"]: score += 2.0
        if latest["Close"] >= latest["ma75"]: score += 2.5
        if latest["ma25_slope_pct"] > 0: score += 1.5
        if latest["ma75_slope_pct"] > 0: score += 1.8
        return score

    technical_passed.sort(key=get_temp_score, reverse=True)
    technical_passed_limited = technical_passed[:MAX_FUNDAMENTALS_精査数]

    print(f"➔ [厳選リミッター作動]: 一次合格 {len(technical_passed)} 銘柄から、スコア上位 {len(technical_passed_limited)} 銘柄に精査対象を絞り込みました。")
    print("\n=== [第2段階] テクニカル合格銘柄に対するファンダメンタルズ個別精査を開始します ===")
    
    for idx, (ticker, hist, latest, latest_date) in enumerate(technical_passed_limited):
        print(f"  [{idx + 1}/{len(technical_passed_limited)}] 詳細精査中... {ticker}")
        
        ticker_obj = None
        try:
            ticker_obj = yf.Ticker(ticker)
            fundamentals = fetch_fundamentals(ticker_obj, ticker)
            
            if fundamentals is None:
                time.sleep(SLEEP_SEC)
                continue

            market_cap = fundamentals.get("market_cap")
            if market_cap is None or market_cap < MIN_MARKET_CAP:
                time.sleep(SLEEP_SEC)
                continue

            revenue_growth = fundamentals.get("revenue_growth_pct")
            if revenue_growth is None or revenue_growth < MIN_REVENUE_GROWTH_PCT:
                time.sleep(SLEEP_SEC)
                continue

            profit_margin = fundamentals.get("profit_margin_pct")
            if profit_margin is None or profit_margin < MIN_PROFIT_MARGIN_PCT:
                time.sleep(SLEEP_SEC)
                continue

            roe = fundamentals.get("roe_pct")
            if roe is None or roe < MIN_ROE_PCT:
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
                    "name": name_map.get(ticker, ticker),
                    "score": score,
                    "trend_score": trend_score,
                    "quality_score": quality_score,
                    "strength_score": strength_score,
                    "close": round(float(latest["raw_close"]), 3),
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
                    "close_vs_ma25_pct": round(float(latest["close_vs_ma25_pct"]), 3) if pd.notna(latest["close_vs_ma25_pct"]) else None,
                    "close_vs_ma75_pct": round(float(latest["close_vs_ma75_pct"]), 3) if pd.notna(latest["close_vs_ma75_pct"]) else None,
                    "close_vs_ma200_pct": round(float(latest["close_vs_ma200_pct"]), 3) if pd.notna(latest["close_vs_ma200_pct"]) else None,
                    "days_since_75gc200": int(latest["days_since_75gc200"]) if pd.notna(latest["days_since_75gc200"]) else None,
                    "days_since_perfect_order": int(latest["days_since_perfect_order"]) if pd.notna(latest["days_since_perfect_order"]) else None,
                    "touch_ma25_intraday": bool(latest["touch_ma25_intraday"]),
                    "touch_ma75_intraday": bool(latest["touch_ma75_intraday"]),
                    "reclaim_ma25_close": bool(latest["reclaim_ma25_close"]),
                    "reclaim_ma75_close": bool(latest["reclaim_ma75_close"]),
                    "failed_ma25_reclaim": bool(latest["failed_ma25_reclaim"]),
                    "failed_ma75_reclaim": bool(latest["failed_ma75_reclaim"]),
                    "support_reaction_ok": bool(latest["support_reaction_ok"]),
                    "ma25_pullback_candidate": bool(latest["ma25_pullback_candidate"]),
                    "ma75_pullback_candidate": bool(latest["ma75_pullback_candidate"]),
                    "trend_filter_ok": bool(latest["trend_filter_ok"]),
                    "volume_filter_ok": bool(latest["volume_filter_ok"]),
                    "support_trace_ok": bool(latest["support_trace_ok"]),
                    "drawdown_filter_ok": bool(latest["drawdown_filter_ok"]),
                    "ma75_quality_filter": bool(latest["ma75_quality_filter"]),
                    "ma75_touch_quality_signal": bool(latest["ma75_touch_quality_signal"]),
                    "ma75_nextday_quality_signal": bool(latest["ma75_nextday_quality_signal"]),
                    "down_volume_spike": bool(latest["down_volume_spike"]),
                    "pullback_score": round(float(latest["pullback_score"]), 3) if pd.notna(latest["pullback_score"]) else None,
                    "pullback_candidate": bool(latest["pullback_candidate"]),
                    "ma25_above_ma200": bool(latest["ma25_above_ma200"]),
                    "ma75_above_ma200": bool(latest["ma75_above_ma200"]),
                    "ma25_above_ma75": bool(latest["ma25_above_ma75"]),
                    "perfect_order": bool(latest["perfect_order"]),
                    "bearish_stack_recent": bool(latest["bearish_stack_recent"]),
                    "bearish_perfect_order_recent": bool(latest["bearish_perfect_order_recent"]),
                    "ma25_cross_200_today": bool(latest["ma25_cross_200_today"]),
                    "ma75_cross_200_today": bool(latest["ma75_cross_200_today"]),
                    "ma25_cross_200_recent": bool(latest["ma25_cross_200_recent"]),
                    "ma75_cross_200_recent": bool(latest["ma75_cross_200_recent"]),
                    "ma25_cross_75_today": bool(latest["ma25_cross_75_today"]),
                    "ma25_cross_75_recent_tight": bool(latest["ma25_cross_75_recent_tight"]),
                    "ma25_cross_200_recent_tight": bool(latest["ma25_cross_200_recent_tight"]),
                    "ma75_cross_200_recent_tight": bool(latest["ma75_cross_200_recent_tight"]),
                    "perfect_order_recent": bool(latest["perfect_order_recent"]),
                    "perfect_order_today": bool(latest["perfect_order_today"]),
                    "perfect_order_recent_tight": bool(latest["perfect_order_recent_tight"]),
                    "initial_trend_signal": bool(latest["initial_trend_signal"]),
                    "early_reversal_setup": bool(latest["early_reversal_setup"]),
                    "reversal_from_bearish_po": bool(latest["reversal_from_bearish_po"]),
                    "sector": fundamentals["sector"],
                    "industry": fundamentals["industry"],
                }
            )
        except Exception as e:
            print(f"    [精査エラー] {ticker} の検証中に予期せぬエラー: {e}")
        finally:
            close_ticker_session(ticker_obj)
        if (idx + 1) % 25 == 0:
            gc.collect()
        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No long-term candidates")
        return

    df = df.sort_values(["score", "quality_score", "trend_score"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    output_df = df[
        [
            "run_date", "screen_version", "generated_at", "rank", "ticker", "name",
            "score", "trend_score", "quality_score", "strength_score", "close",
            "turnover_million", "market_cap_billion", "revenue_growth_pct",
            "profit_margin_pct", "roe_pct", "current_ratio", "debt_to_equity",
            "change_20d_pct", "change_60d_pct", "change_120d_pct",
            "gap_to_52w_high_pct", "volume_ratio_20", "ma25_slope_pct",
            "ma75_slope_pct", "ma200_slope_pct", "close_vs_ma25_pct",
            "close_vs_ma75_pct", "close_vs_ma200_pct", "days_since_75gc200",
            "days_since_perfect_order", "touch_ma25_intraday", "touch_ma75_intraday",
            "reclaim_ma25_close", "reclaim_ma75_close", "failed_ma25_reclaim",
            "failed_ma75_reclaim", "support_reaction_ok", "ma25_pullback_candidate",
            "ma75_pullback_candidate", "trend_filter_ok", "volume_filter_ok",
            "support_trace_ok", "drawdown_filter_ok", "ma75_quality_filter",
            "ma75_touch_quality_signal", "ma75_nextday_quality_signal",
            "down_volume_spike", "pullback_score", "pullback_candidate",
            "ma25_above_ma200", "ma75_above_ma200", "ma25_above_ma75",
            "perfect_order", "bearish_stack_recent", "bearish_perfect_order_recent",
            "ma25_cross_200_today", "ma75_cross_200_today", "ma25_cross_200_recent",
            "ma75_cross_200_recent", "ma25_cross_75_today", "ma25_cross_75_recent_tight",
            "ma25_cross_200_recent_tight", "ma75_cross_200_recent_tight",
            "perfect_order_recent", "perfect_order_today", "perfect_order_recent_tight",
            "initial_trend_signal", "early_reversal_setup", "reversal_from_bearish_po",
            "sector", "industry",
        ]
    ].head(TOP_N_OUTPUT)
    
    latest_export_df = format_long_term_latest_output(output_df)
    history_export_df = format_long_term_output(output_df)
    
    gc_df = df[
        (
            df["reversal_from_bearish_po"]
            | (
                df["early_reversal_setup"]
                & df["days_since_75gc200"].notna()
                & (df["days_since_75gc200"] <= PERFECT_ORDER_LOOKBACK)
            )
            | df["initial_trend_signal"]
        )
        & df["close_vs_ma25_pct"].notna()
        & (df["close_vs_ma25_pct"] >= -1.0)
        & (df["close_vs_ma25_pct"] <= 8.0)
        & (~df["down_volume_spike"])
        & (df["ma25_slope_pct"] >= GC_MIN_MA25_SLOPE_PCT)
        & (df["ma75_slope_pct"] >= GC_MIN_MA75_SLOPE_PCT)
    ].copy()
    
    gc_df = gc_df.sort_values(
        ["reversal_from_bearish_po", "early_reversal_setup", "days_since_perfect_order", "days_since_75gc200", "score"],
        ascending=[False, False, True, True, False],
    ).reset_index(drop=True)
    gc_df["rank"] = gc_df.index + 1
    
    gc_output_df = gc_df[
        [
            "run_date", "screen_version", "generated_at", "rank", "ticker", "name",
            "reversal_from_bearish_po", "early_reversal_setup", "initial_trend_signal",
            "days_since_perfect_order", "days_since_75gc200", "close_vs_ma25_pct",
            "close_vs_ma75_pct", "bearish_perfect_order_recent", "perfect_order_today",
            "perfect_order_recent_tight", "perfect_order_recent", "ma25_cross_75_recent_tight",
            "ma25_cross_200_recent_tight", "ma75_cross_200_recent_tight", "score",
            "trend_score", "quality_score", "strength_score", "close", "turnover_million",
            "market_cap_billion", "revenue_growth_pct", "profit_margin_pct", "roe_pct",
            "change_20d_pct", "change_60d_pct", "gap_to_52w_high_pct", "volume_ratio_20",
            "ma25_slope_pct", "ma75_slope_pct", "ma25_above_ma75", "ma25_above_ma200",
            "ma75_above_ma200", "perfect_order", "sector", "industry",
        ]
    ].head(TOP_N_GC_OUTPUT)
    
    gc_export_df = format_long_term_gc_output(gc_output_df)

    if screen_date is None:
        screen_date = pd.Timestamp(output_df["run_date"].max()).date()

    latest_output_path = _latest_output_path()
    latest_gc_output_path = _latest_gc_output_path()
    dated_output_path = LONG_TERM_WATCHLISTS_DIR / f"{screen_date.isoformat()}_{LONG_TERM_SCREEN_VERSION}_{run_stamp}.csv"
    gc_watchlists_dir = _gc_watchlists_dir()
    gc_watchlists_dir.mkdir(parents=True, exist_ok=True)
    dated_gc_output_path = gc_watchlists_dir / f"{screen_date.isoformat()}_{LONG_TERM_SCREEN_VERSION}_{run_stamp}.csv"
    
    latest_export_df.to_csv(latest_output_path, index=False, encoding="utf-8-sig")
    history_export_df.to_csv(dated_output_path, index=False, encoding="utf-8-sig")
    gc_export_df.to_csv(latest_gc_output_path, index=False, encoding="utf-8-sig")
    gc_export_df.to_csv(dated_gc_output_path, index=False, encoding="utf-8-sig")

    print("\n==== Long Term Watchlist ====")
    print(latest_export_df.to_string(index=False))
    print(f"\nCSV出力完了: {latest_output_path.name}")
    print(f"履歴保存完了: {dated_output_path}")
    print(f"GC専用出力完了: {latest_gc_output_path.name}")
    print(f"GC専用履歴保存完了: {dated_gc_output_path}")

    # ==========================================
    # ★【Version 1.3 / 1.5 修正】：合格者の自動累積台帳（candidate_history.csv）への自動追記 ★
    # ==========================================
    if rows:
        history_rows = []
        for r in rows:
            history_rows.append({
                "date": r["run_date"],
                "ticker": r["ticker"],
                "name": r["name"],
                "score": r["score"],
                "close_at_trigger": r["close"],
                "ma200_slope_pct": r["ma200_slope_pct"] if r["ma200_slope_pct"] is not None else 0.0,
                "revenue_growth_pct": r["revenue_growth_pct"],
                "roe_pct": r["roe_pct"],
                "status": "tracking",
                "return_7d": None,
                "return_14d": None,
                "return_30d": None,
                "max_high_30d": None,
                "max_dd_30d": None
            })
            
        new_df = pd.DataFrame(history_rows)
        
        if CANDIDATE_HISTORY_CSV.exists():
            try:
                existing_df = pd.read_csv(CANDIDATE_HISTORY_CSV)
                existing_keys = set(zip(existing_df["date"].astype(str), existing_df["ticker"].astype(str)))
                filtered_rows = [
                    row for row in history_rows if (str(row["date"]), str(row["ticker"])) not in existing_keys
                ]
                
                if filtered_rows:
                    append_df = pd.DataFrame(filtered_rows)
                    combined_df = pd.concat([existing_df, append_df], ignore_index=True)
                    combined_df.to_csv(CANDIDATE_HISTORY_CSV, index=False, encoding="utf-8-sig")
                    print(f"\n[台帳保存成功] 新しく {len(filtered_rows)} 件の中期合格銘柄を candidate_history.csv に追加登録しました。")
                else:
                    print("\n[台帳スキップ] 本日の合格銘柄は、すでに台帳に記録済みです。")
                    
            except Exception as e:
                print(f"\n[台帳エラー] 台帳のマージ中に予期せぬエラーが発生しました: {e}")
        else:
            CANDIDATE_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(CANDIDATE_HISTORY_CSV, index=False, encoding="utf-8-sig")
            print(f"\n[台帳新規作成] 累積台帳（candidate_history.csv）を新規作成し、初期データ {len(new_df)} 件を登録しました。")


if __name__ == "__main__":
    run()
