# track_long_term_positions.py (Version 1.2 - Advanced Risk-Reward tracking)
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from market_data_utils import adjusted_entry_price, detect_price_data_issue, prepare_price_history, select_latest_completed_row
from output_format import format_long_term_tracking_output

TRACKED_TICKERS_CSV = "tracked_tickers.csv"
OUTPUT_CSV = "long_term_tracking.csv"
OUTPUT_DIR = "results/long_term_tracking"

WEBHOOK_URL = os.environ.get("SPREADSHEET_WEBHOOK_URL")


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
        columns=["ticker", "name", "entry_date", "entry_price", "position_type", "note"]
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
    if "position_type" not in df.columns:
        df["position_type"] = ""
    if "note" not in df.columns:
        df["note"] = ""
    df["position_type"] = df.apply(
        lambda row: normalize_position_type(row.get("position_type", ""), row.get("note", "")),
        axis=1,
    )
    return df.reset_index(drop=True)


def normalize_position_type(value: str, note: str) -> str:
    raw = str(value).strip().lower()
    if raw in {"scout", "core", "review"}:
        return raw

    note_text = str(note)
    if "検証" in note_text or "過去売却" in note_text:
        return "review"
    if "既存保有" in note_text:
        return "core"
    return "scout"


def fetch_history(ticker: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period="18mo", interval="1d", auto_adjust=False, actions=True)
    except Exception as exc:
        print(f"fetch_history error: {ticker} {exc}")
        return None

    if hist is None or hist.empty or len(hist) < 200:
        return None

    return prepare_price_history(hist)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma25"] = df["Close"].rolling(25).mean()
    df["ma75"] = df["Close"].rolling(75).mean()
    df["ma200"] = df["Close"].rolling(200).mean()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio_20"] = df["Volume"] / df["vol_avg20"]
    df["volume_ratio_20_prev"] = df["volume_ratio_20"].shift(1)
    df["change_20d_pct"] = (df["Close"] - df["Close"].shift(20)) / df["Close"].shift(20) * 100
    df["change_60d_pct"] = (df["Close"] - df["Close"].shift(60)) / df["Close"].shift(60) * 100
    df["high_60"] = df["High"].rolling(60).max()
    df["drawdown_from_60d_high_pct"] = (df["Close"] - df["high_60"]) / df["high_60"] * 100
    df["close_below_ma75"] = df["Close"] < df["ma75"]
    df["close_below_ma75_2d"] = df["close_below_ma75"] & df["close_below_ma75"].shift(1).fillna(False)
    df["ma25_below_ma75"] = df["ma25"] < df["ma75"]
    df["ma25_cross_below_75_today"] = df["ma25_below_ma75"] & (~df["ma25_below_ma75"].shift(1).fillna(False))
    df["recent_high_20"] = df["High"].rolling(20).max()
    df["days_from_20d_high"] = (pd.Series(range(len(df)), index=df.index) - pd.Series(range(len(df)), index=df.index).where(df["High"] >= df["recent_high_20"])).ffill()
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


def judge_status_advanced(latest: pd.Series, entry_price: float | None) -> tuple[str, int, list[str]]:
    """
    ③ ＆ ④：【Version 2.3 実戦リスクリワード設計マージ】
    1. 75日線終値割れ(2日連続) ＆ 取得単価から-12%（MAE極限割れ）で「ロスカット強制撤退」をジャッジ
    2. 含み益 +15% 到達後の高値から-10%下落、または高値更新に失敗して-10%下落で「トレーリング利確」をジャッジ
    """
    flags: list[str] = []
    score = 0
    close_price = float(latest["Close"])

    # 1. ロスカット防衛ロジック（75日線2日連続割れ、または-12%強制ロスカット）
    close_below_75ma_2d = bool(latest["close_below_ma75_2d"])
    forced_loss_cut = False
    
    if entry_price is not None and entry_price > 0:
        loss_pct = (close_price - entry_price) / entry_price * 100
        # 取得単価から -12%（MAEの統計的限界下落）を下回ったら強制ロスカット
        if loss_pct <= -12.0:
            forced_loss_cut = True
            flags.append(f"【危険】取得価格から -12% 限界突破 (本日: {loss_pct:+.1f}%)")

    if close_below_75ma_2d:
        score += 4
        flags.append("終値が75日線を2日連続で割れ")
        
    if forced_loss_cut:
        score += 8
        flags.append("MAE制限：強制ロスカット基準に到達")

    # 2. トレーリング・ストップ利確ロジック
    # 60日高値（最高値）からの下落率を判定
    drawdown_from_high = float(latest["drawdown_from_60d_high_pct"])
    is_trailing_stop = False
    
    if entry_price is not None and entry_price > 0:
        profit_pct = (close_price - entry_price) / entry_price * 100
        # 途中で15%以上の含み益（MFE到達）があり、かつ高値から-10%以上下落（利益確保）
        # もしくは、単純に最高値から-10%下落して利益を吐き出しそうになっている場合
        if profit_pct >= 15.0 and drawdown_from_high <= -10.0:
            is_trailing_stop = True
            flags.append(f"【利確】高値から -10% 下落してトレーリング作動")

    if is_trailing_stop:
        score += 5
        flags.append("MFE基準：半分利確・トレーリング推奨")

    # その他の補助警戒シグナル
    if latest["ma25_cross_below_75_today"]:
        score += 5
        flags.append("25日線が75日線を再DC")
    if latest["Close"] < latest["ma25"]:
        score += 1
        flags.append("終値が25日線割れ")
    if drawdown_from_high <= -12:
        score += 2
        flags.append("60日高値から大きく下落")
    if latest["change_20d_pct"] < -8:
        score += 1
        flags.append("20日騰落率が悪化")

    # 最終ステータス判定
    if forced_loss_cut or close_below_75ma_2d or latest["ma25_cross_below_75_today"]:
        status = "撤退"
    elif is_trailing_stop:
        status = "利確"  # 👈 新設：利確シグナル
    elif score >= 3:
        status = "警戒"
    else:
        status = "継続"
        
    return status, score, flags


def suggested_action_advanced(position_type: str, status: str) -> str:
    """
    推奨アクションに、新設の「利確」を追加
    """
    actions = {
        "scout": {
            "継続": "少量で継続観察",
            "継続(注意)": "まだ様子見",
            "警戒": "撤退検討",
            "利確": "利益確定（全決済）",  # 👈 スプレッドシートに「利確」を指示
            "撤退": "ロスカット強制撤退",
        },
        "core": {
            "継続": "保有継続",
            "継続(注意)": "買い増し停止",
            "警戒": "縮小・防衛ライン確認",
            "利確": "利益確保（半分利確・残り追跡）",  # 👈 恩株化・25MA割れ追跡指示
            "撤退": "ロスカット強制撤退",
        },
        "review": {
            "継続": "検証継続",
            "継続(注意)": "検証継続",
            "警戒": "要検証",
            "利確": "検証成功（利益）",
            "撤退": "検証完了（損切り）",
        },
    }
    return actions.get(position_type, actions["scout"]).get(status, "様子見")


def log_tracking_to_spreadsheet(rows: list[dict]) -> None:
    if not WEBHOOK_URL:
        print("\n📢 [Spreadsheet警告] SPREADSHEET_WEBHOOK_URL が未設定のため、ポートフォリオの自動同期をスキップします。")
        return
    if not rows:
        return

    print(f"\n📢 [Spreadsheet] ポートフォリオの健康診断データをGoogleシートに全自動同期します... (対象: {len(rows)} 件)")

    sanitized_rows = []
    for r in rows:
        sanitized = {}
        for k, v in r.items():
            if pd.isna(v) or v is None:
                sanitized[k] = ""
            else:
                sanitized[k] = v
        sanitized_rows.append(sanitized)

    try:
        headers = {"Content-Type": "application/json"}
        payload = {"tracking_positions": sanitized_rows}
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=20)
        
        if response.status_code == 200:
            print(f"  ➔ [同期成功] ポートフォリオが1秒で上書き更新されました。")
        else:
            print(f"  ➔ [同期失敗] ステータスコード: {response.status_code} / 応答: {response.text}")
    except Exception as e:
        print(f"  ➔ [同期エラー] Webhook送信中に例外が発生しました: {e}")


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
        latest, latest_date = select_latest_completed_row(hist)
        run_date = latest_date if run_date is None else max(run_date, latest_date)

        try:
            entry_price = float(row["entry_price"]) if str(row["entry_price"]).strip() else None
        except Exception:
            entry_price = None

        # 3. ＆ 4. の有意差ルールをマージした「高度な健康診断判定」の実行
        status, status_score, flags = judge_status_advanced(latest, entry_price)
        
        data_issue = detect_price_data_issue(latest, hist)
        if data_issue:
            flags.append(data_issue)

        adjusted_entry = adjusted_entry_price(entry_price, str(row["entry_date"]).strip(), hist, latest)

        rows.append(
            {
                "generated_at": generated_at,
                "ticker": ticker,
                "name": name,
                "entry_date": row["entry_date"],
                "entry_price": entry_price,
                "adjusted_entry_price": round(float(adjusted_entry), 3) if adjusted_entry else None,
                "position_type": row["position_type"],
                "close": round(float(latest["raw_close"]), 3),
                "close_vs_entry_pct": round(pct(float(latest["raw_close"]), adjusted_entry), 3) if adjusted_entry else None,
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
                "suggested_action": suggested_action_advanced(row["position_type"], status),
                "data_issue": data_issue,
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

    # スプレッドシートへ自動送信
    log_tracking_to_spreadsheet(rows)


if __name__ == "__main__":
    run()
