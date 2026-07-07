# track_candidates_performance.py (Version 1.0)
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf

# configから累積台帳のパスを読み込み
from config import CANDIDATE_HISTORY_CSV

SLEEP_SEC = 0.8  # yfinanceのIPブロック制限回避用


def track_candidates_performance() -> None:
    if not CANDIDATE_HISTORY_CSV.exists():
        print(f"台帳ファイル {CANDIDATE_HISTORY_CSV} が見つかりません。")
        return

    try:
        df = pd.read_csv(CANDIDATE_HISTORY_CSV)
    except Exception as e:
        print(f"台帳読み込みエラー: {e}")
        return

    if df.empty:
        print("台帳にデータが存在しません。追跡をスキップします。")
        return

    # statusが 'tracking' (追跡中) の行のみを自動抽出
    tracking_mask = df["status"] == "tracking"
    tracking_rows = df[tracking_mask]

    if tracking_rows.empty:
        print("現在、追跡中(tracking)の中期候補は存在しません（全件確定済み）。")
        return

    print(f"\n=== 中期合格台帳の自動追跡を開始します (追跡対象: {len(tracking_rows)} 件) ===")

    for idx, r in tracking_rows.iterrows():
        ticker = r["ticker"]
        trigger_date_str = r["date"]
        close_at_trigger = float(r["close_at_trigger"])

        print(f"  ・追跡中... {ticker} (抽出日: {trigger_date_str} / 抽出価格: {close_at_trigger:.1f}円)")

        try:
            # yfinanceから過去18ヶ月分の日足データを取得
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="18mo", interval="1d", auto_adjust=False, actions=False)

            if hist is None or hist.empty or len(hist) < 20:
                time.sleep(SLEEP_SEC)
                continue

            # 日付の表記を文字列 (YYYY-MM-DD) に揃える
            hist.index = hist.index.strftime("%Y-%m-%d")

            # 抽出日以降（trigger_date含む）のデータのみに切り出し
            future_hist = hist.loc[hist.index >= trigger_date_str]

            if future_hist.empty:
                time.sleep(SLEEP_SEC)
                continue

            # 抽出日以降に蓄積された実営業日数をカウント
            elapsed_days = len(future_hist)
            closes = future_hist["Close"].values
            highs = future_hist["High"].values
            lows = future_hist["Low"].values

            # --- 営業日ベースでの騰落率 (return) 算出 ---
            # 1. 7日後 (5営業日後 / インデックス 5)
            if elapsed_days >= 6:  # インデックス0が当日、1,2,3,4,5(5営業日後)
                r_7 = (closes[5] - close_at_trigger) / close_at_trigger * 100
                df.at[idx, "return_7d"] = round(r_7, 2)

            # 2. 14日後 (10営業日後 / インデックス 10)
            if elapsed_days >= 11:
                r_14 = (closes[10] - close_at_trigger) / close_at_trigger * 100
                df.at[idx, "return_14d"] = round(r_14, 2)

            # 3. 30日後 (20営業日後 / インデックス 20)
            if elapsed_days >= 21:
                r_30 = (closes[20] - close_at_trigger) / close_at_trigger * 100
                df.at[idx, "return_30d"] = round(r_30, 2)
                # 20営業日が経過した時点で、データの追跡を完了（完了確定ステータス）にする
                df.at[idx, "status"] = "completed"
                print(f"    ➔ 🎉 {ticker} の約1ヶ月(20営業日)の追跡が満期終了しました。数値を確定し completed に変更します。")

            # --- 最大上昇率 ＆ 最大ドローダウンの動的更新（経過営業日内での集計） ---
            lookback_len = min(elapsed_days, 21)  # 最大20営業日後までをスキャン
            period_highs = highs[:lookback_len]
            period_lows = lows[:lookback_len]

            max_high = (np.max(period_highs) - close_at_trigger) / close_at_trigger * 100
            max_dd = (np.min(period_lows) - close_at_trigger) / close_at_trigger * 100

            df.at[idx, "max_high_30d"] = round(max_high, 2)
            df.at[idx, "max_dd_30d"] = round(max_dd, 2)

        except Exception as e:
            print(f"    [追跡エラー] {ticker} の yfinance 解析中にエラーが発生しました: {e}")

        time.sleep(SLEEP_SEC)

    # アップデートされたデータベースを上書き保存
    try:
        df.to_csv(CANDIDATE_HISTORY_CSV, index=False, encoding="utf-8-sig")
        print("\n[追跡完了] アップデートされた成績データを candidate_history.csv に正常に上書き保存しました。")
    except Exception as e:
        print(f"[台帳保存エラー] 台帳の保存に失敗しました: {e}")


if __name__ == "__main__":
    track_candidates_performance()