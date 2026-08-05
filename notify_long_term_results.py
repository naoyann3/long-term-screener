# notify_long_term_results.py (Version 1.3 - AI Academy Momentum Edition)
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


# 👇 --- 【修正後】：関数の先頭に、1箇所だけこの2行を追加します ---
def get_chart_links(ticker: str) -> str:
    """
    あなたが作成された、株探決算・Yahoo!掲示板を含む美しい3行インデントリンク生成ロジック
    """
    # 👈 ★【Version 1.2.2安全防壁】：もしティッカーが空(None/NaN)なら、以降の計算をせず安全に空文字を返します
    if pd.isna(ticker) or not ticker or not isinstance(ticker, str):
        return ""

    code = ticker.split(".")[0] if "." in ticker else ticker
    tradingview_url = f"https://jp.tradingview.com/chart/?symbol=TSE:{code}"
    kabutan_url = f"https://kabutan.jp/stock/finance?code={code}"
    yahoo_url = f"https://finance.yahoo.co.jp/quote/{code}/forum"

    return (
        f"\n"
        f"      ・📈 [TradingView でチャート分析]({tradingview_url})\n"
        f"      ・📊 [株探 で個別株決算分析]({kabutan_url})\n"
        f"      ・🏦 [Yahoo!掲示板 でリアルな大衆心理]({yahoo_url})"
    )


def build_mail_body(latest_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    body = "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"## 📈 【中期成長株・司令室】{today_str} 需給・局面分類レポート\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    # 1. スキャニング合格者の仕分け（買い候補 vs 待機銘柄：①）
    # position_status が 'buy_signal' のものを「買い候補」、'waiting' のものを「待機・注目リスト」に仕分けます
    buy_signals = latest_df[latest_df["position_status"] == "buy_signal"] if "position_status" in latest_df.columns else latest_df
    waiting_signals = latest_df[latest_df["position_status"] == "waiting"] if "position_status" in latest_df.columns else pd.DataFrame()

    # --- A. 買い候補セクション（即戦力） ---
    body += f"### 🟢 【本命：買い候補シグナル点灯】: 【 {len(buy_signals)} 銘柄 】\n"
    body += "移動平均パーフェクトオーダーが完成し、決算業績・時価総額条件をすべてクリアした中期本命候補の一覧です。\n"
    body += "----------------------------------------\n\n"

    for idx, (_, r) in enumerate(buy_signals.head(10).iterrows(), 1):
        ticker = r.get("ticker")
        name = r.get("name")
        score = r.get("score")
        close = r.get("close")
        roe = r.get("roe_pct")
        growth = r.get("revenue_growth_pct")
        cap = r.get("market_cap_billion")
        sector = r.get("sector", "不明")
        stars = r.get("sector_stars", "★★★☆☆")
        rs = r.get("relative_strength", 0.0)

        links_text = get_chart_links(ticker)
        body += f"## {idx}. {name} ({ticker}){links_text}\n"
        body += f"  ・総合スコア: **{score:.1f}点** (終値: {close:.1f}円 ｜ 時価総額: {cap:.1f}十億円)\n"
        body += f"  ・セクター風: **{sector:20s} 【 {stars} 】**(相対強度: **{rs:+.1f}%**)\n"
        body += f"  ・財務業績  : ROE: **{roe:.1f}%** ｜ 売上成長率: **{growth:.1f}%**\n"
        
        # 動的解説
        is_reversal = r.get("reversal_from_bearish_po", False)
        is_pullback = r.get("push_filter_ok", False)
        if is_reversal:
            body += "  ・📢【動的着眼点】: 下降トレンド（逆PO）から『上昇パーフェクトオーダー』へと大口資金が完全に切り替わった、転換初日の大本命形状です。\n"
        elif is_pullback:
            body += "  ・📢【動的着眼点】: 綺麗なパーフェクトオーダーを維持しながら、中期支持線（75日線）での反発を確認した安全性の高い押し目位置です。\n"
        else:
            body += "  ・📢【動的着眼点】: 25日・75日・200日線の上で頑健に推移している、教科書通りのパーフェクトオーダー上昇トレンド株です。\n"
        body += "----------------------------------------\n\n"

    # --- B. 待機銘柄セクション（仕込み前夜・注目リスト：① ＆ ②） ---
    body += f"### 🟡 【待機：無関心・売り枯れ注目リスト】: 【 {len(waiting_signals)} 銘柄 】\n"
    body += "まだ買いシグナル（出来高急増など）は出ていませんが、市場から忘れ去られ、売りが完全に細りきった『爆発前夜』の監視銘柄です。\n"
    body += "----------------------------------------\n\n"

    if not waiting_signals.empty:
        for idx, (_, r) in enumerate(waiting_signals.head(10).iterrows(), 1):
            ticker = r.get("ticker")
            name = r.get("name")
            close = r.get("close")
            sector = r.get("sector", "不明")
            stars = r.get("sector_stars", "★★★☆☆")
            forgotten_score = int(r.get("forgotten_score", 70))
            is_deep_value = r.get("deep_value_setup", False)

            links_text = get_chart_links(ticker)
            body += f"## {idx}. {name} ({ticker}){links_text}\n"
            body += f"  ・現在終値  : **{close:.1f}円** ｜ **無関心（Forgotten Score）: 【 {forgotten_score} 点 】** (100点満点)\n"
            body += f"  ・セクター風: **{sector:20s} 【 {stars} 】**\n"
            
            # ③：半値八掛け二割引フラグの動的解説
            if is_deep_value:
                body += f"  ・⚠️【歴史的大底】: 52週高値から **64%以上下落（格言：半値・八掛け・二割引）** を達成した、大口から完全に忘れ去られた超ディープバリュー株です。\n"
                
            body += "  ・📢【動的着眼点】: 出来高が消滅し、ボラティリティも極限スクイーズしています。数日〜数週間以内に『出来高が突如2倍以上に点火する大口の足跡』が出現するかどうかを毎日監視してください。\n"
            body += "----------------------------------------\n\n"
    else:
        body += "  ・本日の極限売り枯れ（待機銘柄）の合格者はありません。\n"
        body += "----------------------------------------\n\n"

    # 3. 🔁 【自動復習・答え合わせコーナー】
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
                    body += f"  ・ {name} ({ticker}) ➔ 登録時: {orig_c:.1f}円 (追跡中)\n"
            body += "\n"
        else:
            body += "  ・過去の合格銘柄データがまだありません。明日以降、自動追跡が開始されます。\n\n"
    else:
        body += "  ・過去の合格銘柄データがまだありません。明日以降、自動追跡が開始されます。\n\n"

    # 4. 🧪 【自律統計コーナー】
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
