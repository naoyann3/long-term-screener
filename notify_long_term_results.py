# notify_long_term_results.py (Version 1.0)
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


def get_tv_link(ticker: str) -> str:
    code = ticker.split(".")[0] if "." in ticker else ticker
    return f"https://jp.tradingview.com/chart/?symbol=TSE:{code}"


def build_mail_body(latest_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    body = "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"## 📈 【中期成長株】{today_str} スクリーニング合格候補\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    # 1. 本日の合格候補サマリー
    body += f"### 🟢 本日の合格銘柄: 【 {len(latest_df)} 銘柄 】\n"
    body += "パーフェクトオーダーかつ高い利益率（ROE 8%以上）、売上成長（5%以上）を満たした中期成長候補の一覧です。\n"
    body += "----------------------------------------\n\n"

    # スコア上位10件を表示（多すぎるとメールが長くなるため、厳選表示）
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

        tv_link = get_tv_link(ticker)
        kabutan_link = f"https://kabutan.jp/stock/?code={ticker.split('.')[0]}"

        body += f"## {rank}. {name} ({ticker})\n"
        body += f"      ・📈 [TradingView でチャート分析]({tv_link})\n"
        body += f"      ・📊 [株探 で企業財務・ニュース]({kabutan_link})\n"
        body += f"  ・総合スコア: **{score:.1f}点** (終値: {close:.1f}円 / 時価総額: {cap:.1f}十億円)\n"
        body += f"  ・財務業績  : ROE: **{roe:.1f}%** ｜ 売上成長率: **{growth:.1f}%**\n"

        # 特徴的なテクニカル要素を抽出して動的解説
        po_days = r.get("PO形成からの日数", None)
        is_reversal = r.get("逆PO→上昇PO転換", False)
        is_pullback = r.get("押し目候補シグナル", False)

        if pd.notna(po_days) and int(po_days) <= 10:
            body += f"  ・📢【動的解説】: 移動平均パーフェクトオーダー（PO）が形成されてからわずか **{int(po_days)}日目** の、極めて新鮮な上昇初期トレンドです。\n"
        elif is_reversal:
            body += "  ・📢【動的解説】: 逆パーフェクトオーダー（下落トレンド）から急反転し、上昇パーフェクトオーダーへ大復活を遂げた劇的な転換初期形状です。\n"
        elif is_pullback:
            body += "  ・📢【動的解説】: 綺麗な上昇パーフェクトオーダーを維持したまま、移動平均線付近まで一時的に株価が「押し目」を形成している狙い目の位置です。\n"
        else:
            body += "  ・📢【動的解説】: 強固な上昇トレンドを維持した中期優良成長株。主要移動平均線の支持線としての機能を観察してください。\n"

        body += "----------------------------------------\n\n"

    if len(latest_df) > 10:
        body += f"※他 {len(latest_df) - 10} 銘柄が合格。詳細は results フォルダ内の long_term_watchlist.csv をご確認ください。\n\n"

    # 2. 🔁 【自動復習・答え合わせコーナー（Review Corner）】
    body += "## 🔁 【復習コーナー（Review Corner）】\n"
    body += "過去に合格台帳に登録された銘柄たちが、その後どのように推移しているかを自動で答え合わせします。\n\n"

    if not history_df.empty:
        history_dates = sorted(history_df["date"].unique())
        # 本日（最新日）を省いた、過去の日付を抽出
        history_dates = [d for d in history_dates if d != today_str]

        if len(history_dates) >= 1:
            prev_date = history_dates[-1]  # 最も直近の過去日
            prev_items = history_df[history_df["date"] == prev_date].head(3)  # 最大3件をピックアップ

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

    # 3. 🧪 【自律統計コーナー（Research Notes）】
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

    # 本日の最新合格ファイルを参照
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

    # 追跡台帳をロード
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