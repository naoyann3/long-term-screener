from __future__ import annotations

import pandas as pd


COLUMN_LABELS = {
    "run_date": "判定日",
    "screen_version": "スクリーナー版",
    "generated_at": "生成日時",
    "rank": "監視順位",
    "ticker": "ティッカー",
    "name": "銘柄名",
    "score": "総合スコア",
    "trend_score": "トレンドスコア",
    "quality_score": "業績スコア",
    "strength_score": "強さスコア",
    "close": "終値",
    "turnover_million": "売買代金(百万円)",
    "market_cap_billion": "時価総額(十億円)",
    "revenue_growth_pct": "売上成長率(%)",
    "profit_margin_pct": "利益率(%)",
    "roe_pct": "ROE(%)",
    "current_ratio": "流動比率",
    "debt_to_equity": "D/Eレシオ",
    "change_20d_pct": "20日騰落率(%)",
    "change_60d_pct": "60日騰落率(%)",
    "change_120d_pct": "120日騰落率(%)",
    "gap_to_52w_high_pct": "52週高値差(%)",
    "volume_ratio_20": "出来高倍率(20日)",
    "ma25_slope_pct": "25日線傾き(%)",
    "ma75_slope_pct": "75日線傾き(%)",
    "ma200_slope_pct": "200日線傾き(%)",
    "ma25_above_ma200": "25日線>200日線",
    "ma75_above_ma200": "75日線>200日線",
    "ma25_cross_200_today": "25日GC200(当日)",
    "ma75_cross_200_today": "75日GC200(当日)",
    "ma25_cross_200_recent": "25日GC200(直近)",
    "ma75_cross_200_recent": "75日GC200(直近)",
    "initial_trend_signal": "初動GC200シグナル",
    "sector": "セクター",
    "industry": "業種",
}

WATCHLIST_LATEST_COLUMN_ORDER = [
    "判定日",
    "監視順位",
    "ティッカー",
    "銘柄名",
    "総合スコア",
    "トレンドスコア",
    "業績スコア",
    "強さスコア",
    "終値",
    "売買代金(百万円)",
    "時価総額(十億円)",
    "売上成長率(%)",
    "利益率(%)",
    "ROE(%)",
    "20日騰落率(%)",
    "60日騰落率(%)",
    "52週高値差(%)",
    "出来高倍率(20日)",
    "初動GC200シグナル",
    "25日GC200(直近)",
    "75日GC200(直近)",
    "25日線傾き(%)",
    "75日線傾き(%)",
    "セクター",
    "業種",
]

WATCHLIST_HISTORY_COLUMN_ORDER = [
    "判定日",
    "監視順位",
    "ティッカー",
    "銘柄名",
    "総合スコア",
    "トレンドスコア",
    "業績スコア",
    "強さスコア",
    "終値",
    "売買代金(百万円)",
    "時価総額(十億円)",
    "売上成長率(%)",
    "利益率(%)",
    "ROE(%)",
    "流動比率",
    "D/Eレシオ",
    "20日騰落率(%)",
    "60日騰落率(%)",
    "120日騰落率(%)",
    "52週高値差(%)",
    "出来高倍率(20日)",
    "25日線傾き(%)",
    "75日線傾き(%)",
    "200日線傾き(%)",
    "25日線>200日線",
    "75日線>200日線",
    "25日GC200(当日)",
    "75日GC200(当日)",
    "25日GC200(直近)",
    "75日GC200(直近)",
    "初動GC200シグナル",
    "セクター",
    "業種",
    "スクリーナー版",
    "生成日時",
]

TRACKING_COLUMN_LABELS = {
    "generated_at": "生成日時",
    "ticker": "ティッカー",
    "name": "銘柄名",
    "entry_date": "取得日",
    "entry_price": "取得単価",
    "close": "終値",
    "close_vs_entry_pct": "取得単価比(%)",
    "close_vs_ma25_pct": "25日線乖離(%)",
    "close_vs_ma75_pct": "75日線乖離(%)",
    "close_vs_ma200_pct": "200日線乖離(%)",
    "ma25_vs_ma75_pct": "25日線-75日線差(%)",
    "ma75_vs_ma200_pct": "75日線-200日線差(%)",
    "change_20d_pct": "20日騰落率(%)",
    "change_60d_pct": "60日騰落率(%)",
    "drawdown_from_60d_high_pct": "60日高値からの下落率(%)",
    "volume_ratio_20": "出来高倍率(20日)",
    "upper_shadow_pct": "上ヒゲ率(%)",
    "status": "判定",
    "status_score": "警戒スコア",
    "warning_flags": "警戒サイン",
    "note": "メモ",
}

TRACKING_COLUMN_ORDER = [
    "判定",
    "警戒スコア",
    "ティッカー",
    "銘柄名",
    "取得日",
    "取得単価",
    "終値",
    "取得単価比(%)",
    "25日線乖離(%)",
    "75日線乖離(%)",
    "200日線乖離(%)",
    "25日線-75日線差(%)",
    "75日線-200日線差(%)",
    "20日騰落率(%)",
    "60日騰落率(%)",
    "60日高値からの下落率(%)",
    "出来高倍率(20日)",
    "上ヒゲ率(%)",
    "警戒サイン",
    "メモ",
    "生成日時",
]


def _apply_order(df: pd.DataFrame, ordered_labels: list[str]) -> pd.DataFrame:
    ordered_existing = []
    seen = set()
    for col in ordered_labels:
        if col in df.columns and col not in seen:
            ordered_existing.append(col)
            seen.add(col)
    remaining = [col for col in df.columns if col not in ordered_existing]
    return df[ordered_existing + remaining]


def format_long_term_output(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.rename(columns=COLUMN_LABELS)
    return _apply_order(display_df, WATCHLIST_HISTORY_COLUMN_ORDER)


def format_long_term_latest_output(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.rename(columns=COLUMN_LABELS)
    return display_df[[col for col in WATCHLIST_LATEST_COLUMN_ORDER if col in display_df.columns]]


def format_long_term_tracking_output(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.rename(columns=TRACKING_COLUMN_LABELS)
    return _apply_order(display_df, TRACKING_COLUMN_ORDER)
