# track_long_term_positions.py (Version 1.3 - MFE & Trailing Complete)
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

# 利益保護・トレーリング用の定義パラメータ（定数：Version 2.3）
PROTECTION_TRIGGER_PCT = 10.0      # 建値プロテクションを有効にするトリガー最高利益率（％）
PROTECTION_STOP_LIMIT_PCT = 1.0    # プロテクション発動時の、取得価格比のストップロスライン（％）
TRAILING_TRIGGER_PCT = 15.0        # トレーリング利確を有効にするトリガー最高利益率（％）
TRAILING_GIVEBACK_PCT = 5.0        # トレーリング発動時の、最高値からの最大押し戻し許容率（％）

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


def judge_status_advanced(latest: pd.Series, entry_price: float | None, mfe_pct: float) -> tuple[str, int, list[str]]:
    """
    ④：【重要：5大防衛・利確の優先順位ロジックを完全実装】
    1. 【最優先】MAE損切り（取得価格比 -12%以下）➔ 撤退（ロスカット）
    2. 【第2優先】MFEトレーリング利確（最高益15%超からの5%押し戻し）➔ 利確（トレーリング）
    3. 【第3優先】建値プロテクション（最高益10%超からの買値+1%以下反落）➔ 撤退（同値）
    4. 【第4優先】既存の75日線2日連続割れ / 25日・75日再デッドクロス ➔ 撤退（ロスカット）
    5. 【通常】上記以外 ➔ 継続・警戒
    """
    flags: list[str] = []
    score = 0
    close_price = float(latest["Close"])
    drawdown_from_high = float(latest["drawdown_from_60d_high_pct"])

    # 1. 【最優先】MAE損切り判定（取得単価から -12.0% 以下）
    forced_loss_cut = False
    if entry_price is not None and entry_price > 0:
        loss_pct = (close_price - entry_price) / entry_price * 100
        if loss_pct <= -12.0:
            forced_loss_cut = True

    # 2. 【第2優先】MFEトレーリング利確判定（最高益が15%以上に達したことがあり、最高値から5%以上押し戻されている場合）
    is_trailing_stop = False
    if entry_price is not None and entry_price > 0 and mfe_pct >= TRAILING_TRIGGER_PCT:
        # 最高値（MFE換算）の算出
        max_close_val = entry_price * (1 + mfe_pct / 100.0)
        # 現在値が、その最高値から-5%（TRAILING_GIVEBACK_PCT）以上押し戻されたかを逆算
        giveback = (close_price - max_close_val) / max_close_val * 100
        if giveback <= -TRAILING_GIVEBACK_PCT:
            is_trailing_stop = True

    # 3. 【第3優先】建値プロテクション判定（最高益が10%以上に達したことがあり、現在の含み損益が+1.0%以下まで反落した場合）
    is_protection_stop = False
    if entry_price is not None and entry_price > 0 and mfe_pct >= PROTECTION_TRIGGER_PCT:
        current_profit_pct = (close_price - entry_price) / entry_price * 100
        if current_profit_pct <= PROTECTION_STOP_LIMIT_PCT:
            is_protection_stop = True

    # 4. 【第4優先】既存の移動平均線崩壊シグナル
    close_below_75ma_2d = bool(latest["close_below_ma75_2d"])
    ma25_cross_below_75 = bool(latest["ma25_cross_below_75_today"])

    # --- 規律に基づいた優先順位別の条件分岐ジャッジ ---
    if forced_loss_cut:
        status = "撤退"
        score += 8
        flags.append(f"【危険】取得価格から -12% 限界突破 (本日: {(close_price - entry_price) / entry_price * 100:+.1f}%)")
        flags.append("MAE制限：強制ロスカット基準に到達")
        
    elif is_trailing_stop:
        status = "利確"  # 👈 新設された「利確」ステータス
        score += 5
        flags.append(f"【利益確定】MFE(最高益)+{mfe_pct:.1f}%から -{TRAILING_GIVEBACK_PCT:.1f}%押戻し（トレーリング利確発動）")
        
    elif is_protection_stop:
        status = "撤退"
        score += 3
        flags.append(f"【利益保護】最高益+{mfe_pct:.1f}%到達後の買値付近反落（同値撤退発動）")
        
    elif close_below_75ma_2d or ma25_cross_below_75:
        status = "撤退"
        if close_below_75ma_2d:
            score += 4
            flags.append("終値が75日線を2日連続で割れ")
        if ma25_cross_below_75:
            score += 5
            flags.append("25日線が75日線を再DC")
            
    else:
        # 5. 【通常】上記に該当しない場合は、スコアリング警戒システムにバトンを渡します
        if latest["Close"] < latest["ma25"]:
            score += 1
            flags.append("終値が25日線割れ")
        if drawdown_from_high <= -12:
            score += 2
            flags.append("60日高値から大きく下落")
        elif drawdown_from_high <= -8:
            score += 1
            flags.append("60日高値から下落")
        if latest["change_20d_pct"] < -8:
            score += 1
            flags.append("20日騰落率が悪化")

        if score >= 3:
            status = "警戒"
        else:
            status = "継続"

    return status, score, flags


def suggested_action_advanced(position_type: str, status: str, flags: list[str]) -> str:
    """
    利確・建値撤退に適合した、より具体的な推奨アクションテキストを動的出力します
    """
    flag_str = "".join(flags)
    
    # 警告サインのテキスト内容から、アクションを直接動的に仕分け
    if "利益確定" in flag_str:
        if position_type == "core":
            return "利益確保（半分利確・残り追跡）"
        return "トレーリング利確（利益確定）"
        
    if "利益保護" in flag_str:
        return "建値撤退（利益保護）"

    actions = {
        "scout": {
            "継続": "少量で継続観察",
            "継続(注意)": "まだ様子見",
            "警戒": "撤退検討",
            "利確": "トレーリング利確（利益確定）",
            "撤退": "ロスカット強制撤退",
        },
        "core": {
            "継続": "保有継続",
            "継続(注意)": "買い増し停止",
            "警戒": "縮小・防衛ライン確認",
            "利確": "利益確保（半分利確・残り追跡）",
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

        adjusted_entry = adjusted_entry_price(entry_price, str(row["entry_date"]).strip(), hist, latest)

        # === 1. 【MFE（最高含み益率）】の自動逆算追跡ロジックの埋め込み ===
        mfe_pct = 0.0
        if adjusted_entry and str(row["entry_date"]).strip():
            entry_day_str = str(row["entry_date"]).strip()
            # 日付比較のためにインデックスをクレンジング
            hist_normalized = hist.copy()
            hist_normalized.index = hist_normalized.index.strftime("%Y-%m-%d")
            
            # 取得日以降の未来データのみをスキャン
            future_hist = hist_normalized.loc[hist_normalized.index >= entry_day_str]
            if not future_hist.empty:
                max_close = future_hist["Close"].max()
                # 取得単価からの最高利益率（MFE%）を算出
                mfe_pct = (max_close - adjusted_entry) / adjusted_entry * 100

        # === 2. 高度な5段階優先順位ジャッジ（Version 2.3）の実行 ===
        status, status_score, flags = judge_status_advanced(latest, entry_price, mfe_pct)
        
        data_issue = detect_price_data_issue(latest, hist)
        if data_issue:
            flags.append(data_issue)

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
                "suggested_action": suggested_action_advanced(row["position_type"], status, flags),
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

    # スプレッドシートへ自動送信（※GAS側は修正不要のまま、判定「利確」「建値撤退」をそのまま上書きセル転記します）
    log_tracking_to_spreadsheet(rows)


if __name__ == "__main__":
    run()
