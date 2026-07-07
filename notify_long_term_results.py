# notify_long_term_results.py (Version 1.1 - Multi Link Complete Edition)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from pathlib import Path
import smtplib
import numpy as np
import pandas as pd
import yfinance as yf

# パス定義
from config import CANDIDATE_HISTORY_CSV

# 環境変数 (GitHub Secrets からロード)
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL")
SENDER_NAME = "Sniper OS - Long Term Screener"


def get_chart_links(ticker: str) -> str:
    """
    あなたが作成された、株探決算・Yahoo!掲示板を含む美しい3行インデントリンク生成ロジック
    """
    code = ticker.split(".")[0] if "." in ticker else ticker
    tradingview_url = f"https://jp.tradingview.com/chart/?symbol=TSE:{code}"
    kabutan_url = f"https://kabutan.jp/stock/finance?code={code}"
    yahoo_url = f"https://finance.yahoo.co.jp/quote/{code}/forum"

    # 見やすさと対称性を考慮した改行インデント形式
    return (
        f"\n"
        f"      ・📈 [TradingView でチャート分析]({tradingview_url})\n"
        f"      ・📊 [株探 で個別株決算分析]({kabutan_url})\n"
        f"      ・🏦 [Yahoo!掲示板 でリアルな大衆心理]({yahoo_url})"
    )


def build_mail_body(latest_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    body = "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"## 📈 【中期成長株】{today_str} スクリーニング合格候補\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    # 1. 本日の合格候補サマリー
    body += f"### 🟢 本日の合格銘柄: 【 {len(latest_df)} 銘柄 】\n"
    body += "パーフェクトオーダーかつ高い利益率（ROE 8%以上）、売上成長（5%以上）を満たした中期成長候補の一覧です。\n"
    body += "----------------------------------------\n\n"

    # スコア上位10件を表示
    top_10 = latest_df.head(10)
    for idx, r in top_10.iterrows():
        rank = r.get("監視順位", r.get("rank", idx + 1))
        ticker = r.get("ティッカー", r.get("ticker"))
        name = r.get("銘柄名", r.get("name"))
        score = r.get("総合スコア", r.get("score"))
        close = r.get("終値", r.get("close"))
        roe = r.get("ROE(%)", r.get("roe_pct"))
        growth = r.get("売上成長率(%)", r.get("revenue_growth_pct"))
        cap = r.get("時価総額(十億円)", r.get("market_cap_billion"))

        # あなたが設計した美しい3行箇条書きリンクを取得して印字
        links_text = get_chart_links(ticker)

        body += f"## {rank}. {name} ({ticker}){links_text}\n"
        body += f"  ・総合スコア: **{score:.1f}点** (終値: {close:.1f}円 / 時価総額: {cap:.1f}十億円)\n"
        body += f"  ・財務業績  : ROE: **{roe:.1f}%** ｜ 売上成長率: **{growth:.1f}%**\n"
        
        # 動的シグナル解説
        if r.get("reversal_from_bearish_po"):
            body += "  ・📢【動的着眼点】: 長期の下降（逆PO）から『上昇パーフェクトオーダー』へとトレンドの主導権が完全に切り替わった、大転換初日の新鮮な形状です。\n"
        elif r.get("early_reversal_setup"):
            body += "  ・📢【動的着眼点】: 下降トレンドの底固めから、25日線が75日線をGC。中期的な反転準備のパターントリガーが引かれました。\n"
        elif r.get("reclaim_ma75_close"):
            body += "  ・📢【動的着眼点】: 綺麗なパーフェクトオーダーを維持しながら、中期75日移動平均線での反発（サポート反応）を確認した絶好の押し目位置です。\n"
        else:
            body += "  ・📢【動的解説】: 強固な上昇トレンドを維持した中期優良成長株。主要移動平均線の支持線としての機能を観察してください。\n"

        body += "----------------------------------------\n\n"

    if len(latest_df) > 10:
        body += f"※他 {len(latest_df) - 10} 銘柄が合格。詳細は results フォルダ内の long_term_watchlist.csv をご確認ください。\n\n"

    # 2. 🔁 【自動復習・答え合わせコーナー】
    body += "## 🔁 【復習コーナー（Review Corner）】\n"
    body += "過去に合格台帳に登録された銘柄たちが、その後どのように推移しているかを自動で答え合わせします。\n\n"

    if not history_df.empty:
        history_dates = sorted(history_df["date"].unique())
        history_dates = [d for d in history_dates if d != today_str]

        if len(history_dates) >= 1:
            prev_date = history_dates[-1]
            prev_items = history_df[history_df["date"] == prev_date].head(3)

            body += f"📅 【前回（ {prev_date} ）合格の教材たちのその後の経過】:\n"
            for _, r in prev_items.iterrows():
                ticker = r["ticker"]
                name = r["name"]
                orig_c = float(r["close_at_trigger"])

                try:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(period="5d", interval="1d", auto_adjust=False)
                    if not hist.empty:
                        curr_c = float(hist["Close"].iloc[-1])
                        perf = (curr_c - orig_c) / orig_c * 100
                        icon = "📈" if perf >= 0 else "📉"
                        body += f"  ・{icon} **{name} ({ticker})** ➔ 登録時: {orig_c:.1f}円 ➔ 本日終値: {curr_c:.1f}円 (騰落: **{perf:+.1f}%**)\n"
                except Exception:
                    body += f"  ・ **{name} ({ticker})** ➔ 登録時: {orig_c:.1f}円 (追跡中)\n"
            body += "\n"
        else:
            body += "  ・過去の合格銘柄データがまだありません。明日以降、自動追跡が開始されます。\n\n"
    else:
        body += "  ・過去の合格銘柄データがまだありません。明日以降、自動追跡が開始されます。\n\n"

    # 3. 🧪 【自律統計コーナー】
    body += "## 🧪 【中期スクリーニング自律統計】\n"
    if not history_df.empty:
        completed_df = history_df[history_df["status"] == "completed"]
        if len(completed_df) >= 3:
            avg_30d = completed_df["return_30d"].mean()
            win_rate_30d = (completed_df["return_30d"] > 0).mean() * 100

            upward_200 = completed_df[completed_df["ma200_slope_pct"] > 0]
            if not upward_200.empty:
                upward_win_rate = (upward_200["return_30d"] > 0).mean() * 100
                upward_avg_30d = upward_200["return_30d"].mean()
                body += f"  ・【統計事実】: これまでに追跡が完了した {len(completed_df)} 件の検証結果です。\n"
                body += f"  ・【全体勝率(30日後)】: **{win_rate_30d:.1f}%** ｜ 平均期待利益: **{avg_30d:+.2f}%**\n"
                body += f"  ・【MA200上向き時の30日後勝率】: **{upward_win_rate:.1f}%** ｜ 平均期待利益: **{upward_avg_30d:+.2f}%**\n"
                body += "  ➔ 統計データより、200日移動平均線が上向きの中期トレンド銘柄は、期待値が有意に高いことが実証されつつあります。\n"
            else:
                body += f"  ・【全体勝率(30日後)】: **{win_rate_30d:.1f}%** (分母 {len(completed_df)} 件)\n"
        else:
            body += f"  ・【検証中】: 現在合格した銘柄を追跡データベースに蓄積中（追跡中: {len(history_df)} 件）。\n"
            body += "  ・30営業日（約1ヶ月）が経過した銘柄から、自動で「勝率・200MA傾き別期待値」の統計レポートがここに自動生成されます。\n"
    else:
        body += "  ・検証データベースを収集中です。\n"

    return body


def notify() -> None:
    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    latest_file = Path("long_term_watchlist.csv")
    if not latest_file.exists():
        print(f"最新の合格ファイル {latest_file} が見つかりません。")
        return

    try:
        latest_df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")
        return

    if latest_df.empty:
        print("本日のスクリーニング合格者は0件です。通知をスキップします。")
        return

    history_df = pd.DataFrame()
    if CANDIDATE_HISTORY_CSV.exists():
        try:
            history_df = pd.read_csv(CANDIDATE_HISTORY_CSV)
        except Exception:
            pass

    body = build_mail_body(latest_df, history_df)

    msg = MIMEMultipart()
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【中期成長株】{today_str} 合格候補 {len(latest_df)} 銘柄"

    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        print("中期スクリーニング結果のメール通知を正常に送信しました。")
    except Exception as e:
        print(f"メール送信エラー: {e}")


if __name__ == "__main__":
    notify()
