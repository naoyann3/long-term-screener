# long_term_screener.py (Version 2.5 - Forgotten & Waiting Watchlist Complete)
from __future__ import annotations

from datetime import datetime
import gc
import json
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

DOWNLOAD_CHUNK_SIZE = 300
SLEEP_SEC = 1.5
TOP_N_OUTPUT = 50
TOP_N_GC_OUTPUT = 20

# yfinance.info のIPブロックを防ぐ、精査最大ロック数
MAX_FUNDAMENTALS_精査数 = 30

MIN_TURNOVER = 100_000_000
MIN_MARKET_CAP = 100_000_000_000  # 時価総額1,000億円（エリート厳選）
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

FUND_DIR = Path(__file__).resolve().parent / "results" / "data_cache" / "fundamentals"
if not FUND_DIR.exists():
    FUND_DIR = Path(__file__).resolve().parent / "data_cache" / "fundamentals"
    if not FUND_DIR.exists():
        FUND_DIR = Path(__file__).resolve().parent.parent / "big_winner_research_results" / "data_cache" / "fundamentals"


def _ticker_path() -> Path:
    return Path(__file__).resolve().parent / TICKERS_CSV


def _latest_output_path() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_CSV


def _latest_gc_output_path() -> Path:
    return Path(__file__).resolve().parent / _latest_output_path().name.replace("watchlist", "gc_watchlist")


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
    full_data = {}
    delisted_list = []
    chunks = [tickers[i:i + DOWNLOAD_CHUNK_SIZE] for i in range(0, len(tickers), DOWNLOAD_CHUNK_SIZE)]
    
    print(f"\n[第1段階] 全 {len(tickers)} 銘柄を一括ダウンロードします (分割数: {len(chunks)} チャンク)...")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"  ・ダウンロード中 ({idx}/{len(chunks)})... {len(chunk)} 銘柄")
        try:
            df_chunk = yf.download(
                chunk, 
                period="12mo",  
                interval="1d", 
                group_by="column", 
                auto_adjust=False, 
                actions=True, 
                progress=False,
                threads=5       
            )
            
            for t in chunk:
                try:
                    if isinstance(df_chunk.columns, pd.MultiIndex):
                        if t in df_chunk.columns.get_level_values(1):
                            df_t = df_chunk.xs(t, axis=1, level=1).copy()
                            df_t = df_t.dropna(subset=["Close", "Volume"])
                            if len(df_t) >= 120:
                                df_ticker = prepare_price_history(df_t)
                                if df_ticker is not None and not df_ticker.empty:
                                    full_data[t] = df_ticker
                                else:
                                    delisted_list.append(t)
                            else:
                                delisted_list.append(t)
                        else:
                            delisted_list.append(t)
                except Exception:
                    delisted_list.append(t)
                    continue
                    
        except Exception as e:
            print(f"  [警告] チャンク {idx} のダウンロード中にエラーが発生しました: {e}")
            
        time.sleep(2.0)
        
    return full_data, delisted_list


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

    # ─── 💡 【Version 2.5新設：Grok/ChatGPT要求の『Forgotten Score（忘却スコア）』＆『半値八掛け二割引』を完全実装】 ───
    # A. 出来高の長期的トレンドの測定
    df["vol_avg60"] = df["Volume"].rolling(60).mean()
    df["vol_avg20_vs_60_ratio"] = df["vol_avg20"] / df["vol_avg60"]
    
    # B. 15営業日のヨコヨコレンジ
    df["high_15"] = df["High"].rolling(15).max()
    df["low_15"] = df["Low"].rolling(15).min()
    df["range_15_pct"] = (df["high_15"] - df["low_15"]) / df["Close"] * 100
    
    # C. 【半値八掛け二割引（格言：高値から64〜68%以上の大暴落）】の定量化 (③)
    df["drawdown_from_52w_high_pct"] = (df["Close"] - df["recent_high_252"]) / df["recent_high_252"] * 100
    df["deep_value_setup"] = df["drawdown_from_52w_high_pct"] <= -64.0

    # D. 【Forgotten Score (売り枯れ・無関心度：100点満点)】の動的算出 (② & ④)
    forgotten_scores = []
    for idx, row in df.iterrows():
        f_score = 0
        
        # 1. 出来高枯渇度（最大40点）
        # 20日出来高比率が 0.5倍以下（完全な売り枯れ：+20点）
        if row["volume_ratio_20"] <= 0.50:
            f_score += 20
        # 中長期的な関心の枯渇（20日平均が60日平均の 0.6倍以下：+20点）
        if row["vol_avg20_vs_60_ratio"] <= 0.60:
            f_score += 20
            
        # 2. ボラティリティの超緊縮（最大30点）
        # BB幅 5.0%以下の極限スクイーズ（+30点） ｜ 8.0%以下のスクイーズ（+15点）
        if row["bb_width"] <= 5.0:
            f_score += 30
        elif row["bb_width"] <= 8.0:
            f_score += 15
            
        # 3. 価格のヨコヨコ安定性（最大20点）
        # 15営業日の高低変動幅が 5%以内の完全膠着（+20点）
        if row["range_15_pct"] <= 5.0:
            f_score += 20
            
        # 4. RSIの完全中立適正圏（最大10点）
        # RSI14が 40%〜55% の中立ゾーン（+10点）
        if 40.0 <= row["rsi14"] <= 55.0:
            f_score += 10
            
        forgotten_scores.append(f_score)
        
    df["forgotten_score"] = forgotten_scores
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
    
    # 銘柄ごとのセクターマッピングを事前に作成（メモリ上にロード）
    sector_map = {}
    for t in all_tickers:
        fund_path = FUND_DIR / f"{t}.json"
        if fund_path.exists():
            try:
                with open(fund_path, "r", encoding="utf-8") as f:
                    fund_data = json.load(f)
                    sector_map[t] = fund_data.get("sector", "不明")
            except Exception:
                sector_map[t] = "不明"
        else:
            sector_map[t] = "不明"
            
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
    
    if delisted_list:
        print(f"\n📢 [自己クリーニング] yfinanceでロードできなかった {len(delisted_list)} 銘柄を検知しました。上場廃止・ティッカー変更とみなしてtickers.csvから自動削除します。")
        cleaned_tickers_df = tickers_df[~tickers_df["ticker"].isin(delisted_list)]
        cleaned_tickers_df.to_csv(_ticker_path(), index=False, encoding="utf-8-sig")
        print("  ➔ tickers.csv の自己クリーニング・浄化処理が完了しました。")

    # 全合格キャッシュを用いて「各セクターの過去40日のモメンタム」を自動集計・5段階星評価
    sector_performances = {}
    for t, hist in all_histories.items():
        s = sector_map.get(t, "不明")
        if s == "不明" or len(hist) < 40:
            continue
        try:
            pct_40d = (hist["Close"].iloc[-1] - hist["Close"].iloc[-40]) / hist["Close"].iloc[-40] * 100
            if s not in sector_performances:
                sector_performances[s] = []
            sector_performances[s].append(pct_40d)
        except Exception:
            continue

    sector_avg_mom = {}
    for s, p_list in sector_performances.items():
        sector_avg_mom[s] = np.mean(p_list) if p_list else 0.0

    sector_meta = {}
    print("\n=== [Version 2.5] 本日のセクターモメンタムを全自動集計しました ===")
    for s, val in sorted(sector_avg_mom.items(), key=lambda x: x[1], reverse=True):
        if val >= 10.0:
            stars = "★★★★★"
            coeff = 1.20
        elif val >= 5.0:
            stars = "★★★★☆"
            coeff = 1.10
        elif val >= -2.0:
            stars = "★★★☆☆"
            coeff = 1.00
        elif val >= -8.0:
            stars = "★★☆☆☆"
            coeff = 0.90
        else:
            stars = "★☆☆☆☆"
            coeff = 0.80
        sector_meta[s] = {"stars": stars, "coeff": coeff, "avg_return": val}
        print(f"  ・📁 {s:25s} ➔ 40日平均: {val:+6.2f}% ｜ 評価: {stars} (セクター係数: {coeff:.2f})")

    technical_passed = []
    waiting_passed = []

    print("\n=== テクニカル 一次足切りスクリーニングを実行します ===")
    for ticker, df_ticker in all_histories.items():
        try:
            hist = prepare_price_history(df_ticker)
            if hist is None or hist.empty:
                continue
                
            hist = calc_indicators(hist)
            latest, latest_date = select_latest_completed_row(hist)

            # 💡 【Version 2.5：二重構造スキャン（①：待機銘柄・ウォッチリスト）の自動分離】
            # ルートA: 通常の買い候補シグナル
            if passes_long_term_filter_technical_only(latest):
                technical_passed.append((ticker, hist, latest, latest_date, "buy_signal"))
                continue
                
            # ルートB: 待機銘柄（テクニカルPO等は未完成だが、出来高が極限に枯渇し、Forgotten Scoreが70点以上の仕込み前夜）
            if latest["turnover"] >= MIN_TURNOVER and latest["forgotten_score"] >= 70:
                waiting_passed.append((ticker, hist, latest, latest_date, "waiting"))

        except Exception:
            continue

    print(f"➔ [一次合格：買い候補シグナル] : {len(technical_passed)} 銘柄")
    print(f"➔ [一次合格：注目待機リスト]   : {len(waiting_passed)} 銘柄")

    # ２つのルートの合格者を一挙にマージ
    all_passed = technical_passed + waiting_passed

    # 期待値と忘却スコアに基づいてソート
    def get_temp_score(item) -> float:
        _, _, latest, _, sig_type = item
        score = 0.0
        # 買い候補には優先的に +5点 のブーストを加算
        if sig_type == "buy_signal":
            score += 5.0
        score += float(latest["forgotten_score"]) * 0.25
        return score

    all_passed.sort(key=get_temp_score, reverse=True)
    all_passed_limited = all_passed[:MAX_FUNDAMENTALS_精査数]

    print(f"➔ [厳選リミッター作動]: 統合合格 {len(all_passed)} 銘柄から、スコア上位 {len(all_passed_limited)} 銘柄に精査対象を絞り込みました。")
    print("\n=== [第2段階] 統合合格銘柄に対するファンダメンタルズ個別精査を開始します ===")
    
    for idx, (ticker, hist, latest, latest_date, sig_type) in enumerate(all_passed_limited):
        print(f"  [{idx + 1}/{len(all_passed_limited)}] 詳細精査中... {ticker} (種別: {sig_type})")
        
        ticker_obj = None
        try:
            ticker_obj = yf.Ticker(ticker)
            fundamentals = fetch_fundamentals(ticker_obj, ticker)
            
            if fundamentals is None:
                time.sleep(SLEEP_SEC)
                continue

            current_sector = fundamentals.get("sector", "不明")
            # 【地雷セクター（不動産、ヘルスケア）の完全除外ハードブロック】
            if current_sector in ["Real Estate", "Healthcare"]:
                print(f"    ➔ ❌ [地雷セクター完全除外] {ticker} は不人気セクター（{current_sector}）のため、自動足切りしました。")
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

            # セクター評価星と係数のマッピング
            s_info = sector_meta.get(current_sector, {"stars": "★★★☆☆", "coeff": 1.0, "avg_return": 0.0})
            sector_stars = s_info["stars"]
            sector_coeff = s_info["coeff"]
            
            # 個別銘柄の40日騰落率の計算
            close_orig_40 = float(hist["Close"].iloc[-40]) if len(hist) >= 40 else float(hist["Close"].iloc[0])
            stock_return_40d = (latest["Close"] - close_orig_40) / close_orig_40 * 100
            
            # 相対強度（Relative Strength）の算出 (③)
            relative_strength = round(stock_return_40d - s_info["avg_return"], 2)
            
            # 最終総合スコアにセクター補正（0.8〜1.2）を乗算
            final_score = round(score * sector_coeff, 2)

            rows.append(
                {
                    "run_date": latest_date.isoformat(),
                    "screen_version": LONG_TERM_SCREEN_VERSION,
                    "generated_at": generated_at,
                    "ticker": ticker,
                    "name": name_map.get(ticker, ticker),
                    "score": final_score,         
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
                    "sector": current_sector,
                    "industry": fundamentals["industry"],
                    "sector_stars": sector_stars,
                    "relative_strength": relative_strength,
                    
                    # ★【Version 2.5新設】：忘却・待機メタデータ
                    "forgotten_score": int(latest["forgotten_score"]),
                    "deep_value_setup": bool(latest["deep_value_setup"]),
                    "position_status": sig_type   # 「buy_signal」または「waiting」を格納
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
    
    # 待機・忘却カラムを最後尾に綺麗にマッピング
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
            "sector", "industry", "sector_stars", "relative_strength",
            "forgotten_score", "deep_value_setup", "position_status" # 追加
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
