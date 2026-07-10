# verify_historical_performance.py (Version 1.0 - Forward Backtest Engine)
from datetime import datetime
import glob
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf

# 保存先・入力元ディレクトリの定義
BASE_DIR = Path(__file__).resolve().parent
WATCHLIST_DIR = BASE_DIR / "results" / "long_term_watchlists"
REPORT_OUTPUT = BASE_DIR / "results" / "historical_performance_report.txt"

SLEEP_SEC = 0.5  # yfinanceブロック回避


def run_verification() -> None:
    if not WATCHLIST_DIR.exists():
        print(f"エラー: 過去の監視データフォルダ {WATCHLIST_DIR} が存在しません。")
        return

    # 1. 過去の全CSVファイル（3月以降）を自動で巡回スキャン・ロード
    csv_files = glob.glob(str(WATCHLIST_DIR / "*.csv"))
    if not csv_files:
        print("過去の合格者CSVファイルが見つかりません。")
        return

    print(f"=== 過去の合格履歴データのロードを開始します (検出ファイル数: {len(csv_files)} 件) ===")

    history_rows = []
    for filepath in csv_files:
        try:
            df_file = pd.read_csv(filepath)
            # 必要なカラムが日本語ラベルになっているためマッピング
            # 判定日, ティッカー, 終値, 200日線傾き(%), ROE(%), 売上成長率(%)
            for idx, r in df_file.iterrows():
                history_rows.append(
                    {
                        "date": r.get("判定日", r.get("run_date")),
                        "ticker": r.get("ティッカー", r.get("ticker")),
                        "name": r.get("銘柄名", r.get("name")),
                        "close_at_trigger": float(r.get("終値", r.get("close"))),
                        "ma200_slope_pct": (
                            float(r["200日線傾き(%)"])
                            if "200日線傾き(%)" in r and pd.notna(r["200日線傾き(%)"])
                            else 0.0
                        ),
                    }
                )
        except Exception as e:
            print(f"  [読み込み警告] ファイル {Path(filepath).name} のパース中にエラー: {e}")

    raw_df = pd.DataFrame(history_rows)
    if raw_df.empty:
        print("有効な過去データが1件もありませんでした。")
        return

    # 重複防止：同じ日に同じ銘柄が複数回読み込まれるのを防ぐ
    raw_df = raw_df.drop_duplicates(subset=["date", "ticker"]).reset_index(drop=True)
    unique_tickers = raw_df["ticker"].unique().tolist()

    print(
        f"➔ ロード完了: 累積シグナル点灯総数: {len(raw_df)} 件 (ユニーク銘柄数: {len(unique_tickers)} 社)"
    )
    print("\n=== 各銘柄のその後の価格ヒストリーを取得し、統計検証を実行します ===")

    # yfinanceから、ユニーク銘柄全体の最新の価格データを一括ロード（高速化）
    # 3月から現在までの期間を完全にカバーするために1年分（12mo）取得
    chunk_size = 300
    chunks = [unique_tickers[i : i + chunk_size] for i in range(0, len(unique_tickers), chunk_size)]
    all_histories = {}

    for idx, chunk in enumerate(chunks, 1):
        print(f"  ・最新株価のロード中 ({idx}/{len(chunks)}チャンク)...")
        try:
            df_chunk = yf.download(
                chunk, period="12mo", interval="1d", group_by="column", auto_adjust=False, progress=False
            )
            for t in chunk:
                try:
                    if isinstance(df_chunk.columns, pd.MultiIndex):
                        if t in df_chunk.columns.get_level_values(1):
                            df_t = df_chunk.xs(t, axis=1, level=1).copy()
                            df_t = df_t.dropna(subset=["Close"])
                            if not df_t.empty:
                                df_t.index = df_t.index.strftime("%Y-%m-%d")
                                all_histories[t] = df_t
                except Exception:
                    continue
        except Exception as e:
            print(f"    チャンク {idx} のダウンロード中にエラー: {e}")
        time.sleep(1.0)

    # 各点灯イベントに対する中長期リターンの算出
    results = []
    for idx, r in raw_df.iterrows():
        ticker = r["ticker"]
        name = r["name"]
        trigger_date = r["date"]
        close_orig = r["close_at_trigger"]

        if ticker not in all_histories:
            continue

        hist = all_histories[ticker]
        future_hist = hist.loc[hist.index >= trigger_date]

        if future_hist.empty:
            continue

        closes = future_hist["Close"].values
        highs = future_hist["High"].values
        lows = future_hist["Low"].values
        elapsed = len(future_hist)

        # 経過営業日（5日=7日後、10日=14日後、20日=30日後、40日=60日後、60日=90日後）
        ret_7d = (closes[5] - close_orig) / close_orig * 100 if elapsed >= 6 else None
        ret_14d = (closes[10] - close_orig) / close_orig * 100 if elapsed >= 11 else None
        ret_30d = (closes[20] - close_orig) / close_orig * 100 if elapsed >= 21 else None
        ret_60d = (closes[40] - close_orig) / close_orig * 100 if elapsed >= 41 else None
        ret_90d = (closes[60] - close_orig) / close_orig * 100 if elapsed >= 61 else None

        # 今日現在（最新行）までの累積リターン
        ret_now = (closes[-1] - close_orig) / close_orig * 100

        # 最大上昇率と最大DD（30営業日内）
        lookback_30d = min(elapsed, 21)
        max_high = (np.max(highs[:lookback_30d]) - close_orig) / close_orig * 100
        max_dd = (np.min(lows[:lookback_30d]) - close_orig) / close_orig * 100

        results.append(
            {
                "date": trigger_date,
                "ticker": ticker,
                "name": name,
                "close_orig": close_orig,
                "ma200_slope": r["ma200_slope_pct"],
                "ret_7d": ret_7d,
                "ret_14d": ret_14d,
                "ret_30d": ret_30d,
                "ret_60d": ret_60d,
                "ret_90d": ret_90d,
                "ret_now": ret_now,
                "max_high_30d": max_high,
                "max_dd_30d": max_dd,
            }
        )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("追跡計算できるデータがありませんでした。")
        return

    # === 統計解析 ＆ レポート自動作成 ===
    report = "==================================================\n"
    report += " 📊 【中期成長株スクリーナー】3月稼働以来の実績期待値検証レポート\n"
    report += f" 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "==================================================\n\n"

    total_signals = len(res_df)
    report += f"◆ 【1. サマリー基礎統計】\n"
    report += f"  ・累積検証シグナル総数 : {total_signals} 件\n"
    report += f"  ・今日現在(累積)の平均騰落率 : {res_df['ret_now'].mean():+.2f}%\n"
    report += f"  ・今日現在(累積)の勝率 (プラス割合) : {(res_df['ret_now'] > 0).mean()*100:.1f}%\n\n"

    # 経過期間ごとの勝率・平均リターンの推移
    periods = [("7営業日後 (約1週間)", "ret_7d"), ("14営業日後 (約2週間)", "ret_14d"), ("30営業日後 (約1ヶ月)", "ret_30d"), ("60営業日後 (約2ヶ月)", "ret_60d"), ("90営業日後 (約3ヶ月)", "ret_90d")]
    
    report += "◆ 【2. 保有期間ごとの成績推移（時間の経過による優位性の変化）】\n"
    for label, col in periods:
        p_df = res_df.dropna(subset=[col])
        if not p_df.empty:
            avg_ret = p_df[col].mean()
            win_rate = (p_df[col] > 0).mean() * 100
            report += f"  ・{label:18s} ➔ 件数: {len(p_df):4d}件 ｜ 平均利益: {avg_ret:+.2f}% ｜ 勝率: {win_rate:.1f}%\n"
    report += "\n"

    # ★ 200日移動平均線の傾きによる明確な有意差の検証
    upward_200 = res_df[res_df["ma200_slope"] > 0]
    downward_200 = res_df[res_df["ma200_slope"] <= 0]

    report += "◆ 【3. 200日移動平均線の傾き(ma200_slope)別の実績差】\n"
    if not upward_200.empty:
        up_win_30d = (upward_200["ret_30d"].dropna() > 0).mean() * 100 if not upward_200["ret_30d"].dropna().empty else 0.0
        up_avg_30d = upward_200["ret_30d"].mean()
        report += f"  🟢 【200日線が上向きの銘柄群】(件数: {len(upward_200)}件)\n"
        report += f"    - 今日現在の累積平均リターン : {upward_200['ret_now'].mean():+.2f}%\n"
        report += f"    - 30営業日後(1ヶ月)の勝率     : {up_win_30d:.1f}% (平均リターン: {up_avg_30d:+.2f}%)\n"
    if not downward_200.empty:
        down_win_30d = (downward_200["ret_30d"].dropna() > 0).mean() * 100 if not downward_200["ret_30d"].dropna().empty else 0.0
        down_avg_30d = downward_200["ret_30d"].mean()
        report += f"  🔴 【200日線が下向き・横ばいの銘柄群】(件数: {len(downward_200)}件)\n"
        report += f"    - 今日現在の累積平均リターン : {downward_200['ret_now'].mean():+.2f}%\n"
        report += f"    - 30営業日後(1ヶ月)の勝率     : {down_win_30d:.1f}% (平均リターン: {down_avg_30d:+.2f}%)\n"
    report += "\n"

    # パフォーマンス上位・下位の銘柄ランキング
    report += "◆ 【4. 伝説の合格銘柄ベスト5（今日現在の累積騰落率）】\n"
    top_5 = res_df.sort_values(by="ret_now", ascending=False).head(5)
    for idx, (_, r) in enumerate(top_5.iterrows(), 1):
        report += f"  {idx}位: **{r['name']} ({r['ticker']})** ➔ 抽出日: {r['date']} ｜ 抽出株価: {r['close_orig']:.1f}円 ➔ 累積利益: **{r['ret_now']:+.1f}%**\n"
    report += "\n"

    report += "◆ 【5. ブレイク失敗・ダマシ銘柄ワースト5（今日現在の累積騰落率）】\n"
    worst_5 = res_df.sort_values(by="ret_now", ascending=True).head(5)
    for idx, (_, r) in enumerate(worst_5.iterrows(), 1):
        report += f"  {idx}位: **{r['name']} ({r['ticker']})** ➔ 抽出日: {r['date']} ｜ 抽出株価: {r['close_orig']:.1f}円 ➔ 累積損失: **{r['ret_now']:+.1f}%**\n"
    report += "\n"

    # レポートを物理的に保存
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(report, encoding="utf-8")
    
    print("\n" + report)
    print(f"\n[検証完了] 統計レポートが {REPORT_OUTPUT} に保存されました。")


if __name__ == "__main__":
    run_verification()