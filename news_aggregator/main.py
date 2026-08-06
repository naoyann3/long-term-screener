# news_aggregator/main.py
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import smtplib
import time

from config import GMAIL_USER, GMAIL_PASS, NOTIFICATION_EMAIL
from database import init_db, save_articles_to_csv, get_today_high_score_news, DB_CSV_PATH
from collector import fetch_rss_feeds, fetch_polymarket_odds
from analyzer import analyze_article_with_llm

def build_mail_html(news_rows) -> str:
    """
    【視認性最大化・ホワイトテーマ版】
    あらゆるメールアプリでの強制反転による文字崩れを防ぎ、
    白背景の中で強弱シグナルが鮮やかに美しく映える、プロ仕様のインテリジェンス・レポート
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    Ftine
    categories = {"地政学": [], "マクロ": [], "株式": [], "暗号資産": []}
    for r in news_rows:
        cat = r.get("category", "マクロ")
        if cat in categories:
            categories[cat].append(r)
        else:
            categories["マクロ"].append(r)

    html = f"""
    <html>
    <head>
      <style>
        body {{ 
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
          background-color: #ffffff; 
          color: #1a1d24; 
          padding: 20px; 
          line-height: 1.6;
        }}
        h1 {{ 
          color: #1a1d24; 
          font-size: 22px; 
          border-bottom: 3px solid #2962ff; 
          padding-bottom: 10px; 
          margin-bottom: 5px;
          font-weight: bold;
        }}
        h2 {{ 
          color: #2962ff; 
          font-size: 15px; 
          margin-top: 30px; 
          border-bottom: 1px solid #e0e3eb; 
          padding-bottom: 6px; 
          font-weight: bold;
        }}
        .news-card {{ 
          background-color: #f8f9fa; 
          padding: 14px; 
          border-radius: 6px; 
          margin-bottom: 14px; 
          border: 1px solid #e0e3eb;
          border-left: 5px solid #2962ff; 
        }}
        .score {{ 
          font-weight: bold; 
          padding: 3px 8px; 
          border-radius: 4px; 
          font-size: 11px; 
          display: inline-block;
        }}
        .score-high {{ background-color: #ffebee; color: #c62828; }}
        .score-mid {{ background-color: #fff3e0; color: #ef6c00; }}
        .score-low {{ background-color: #e8f5e9; color: #2e7d32; }}
        .fact-list {{ 
          margin: 10px 0 5px 0; 
          padding-left: 18px; 
          font-size: 13.5px; 
          color: #33353b; 
        }}
        .fact-list li {{
          margin-bottom: 4px;
        }}
        .meta-line {{ 
          font-size: 11px; 
          color: #70757f; 
          margin-top: 10px; 
          border-top: 1px dashed #e0e3eb;
          padding-top: 6px;
        }}
        .link {{ 
          color: #1a1d24; 
          text-decoration: none; 
          font-weight: bold; 
        }}
        .link:hover {{
          color: #2962ff;
          text-decoration: underline;
        }}
      </style>
    </head>
    <body>
      <h1>📬 【マクロインテリジェンス・司令室】 {today_str}</h1>
      <p style="font-size: 12px; color: #70757f; margin-top: 5px; margin-bottom: 20px;">
        主観や感情的な煽り表現を100%ノイズカットした、冷徹なファクト（客観的エビデンス）のみをお届けします。
      </p>
    """

    for cat_name, items in categories.items():
        if not items:
            continue
        html += f"<h2>■ {cat_name} ＆ 金融インフラ</h2>"
        for item in items:
            score = int(item.get("score", 50))
            if score >= 90:
                score_class = "score-high"
                score_label = f"最重要: {score}点"
            elif score >= 70:
                score_class = "score-mid"
                score_label = f"判断材料: {score}点"
            else:
                score_class = "score-low"
                score_label = f"参考情報: {score}点"
                
            sentiment = item.get("sentiment", "中立")
            
            # センチメントに応じたアクセントカラーの設定
            if sentiment == "強材料":
                sentiment_color = "#ef5350"  # 鮮やかな赤
                sentiment_bg = "#ffebee"
                sentiment_text_color = "#c62828"
            elif sentiment == "弱材料":
                sentiment_color = "#26a69a"  # 鮮やかな緑
                sentiment_bg = "#e0f2f1"
                sentiment_text_color = "#00695c"
            else:
                sentiment_color = "#2962ff"  # 鮮やかな青
                sentiment_bg = "#e3f2fd"
                sentiment_text_color = "#1565c0"
            
            html += f"""
            <div class="news-card" style="border-left-color: {sentiment_color};">
              <span class="score {score_class}">{score_label}</span> 
              <span style="background-color: {sentiment_bg}; color: {sentiment_text_color}; font-weight: bold; font-size: 11px; padding: 3px 8px; border-radius: 4px; margin-left: 6px;">【{sentiment}】</span>
              <strong style="font-size: 14px; margin-left: 8px;"><a class="link" href="{item['url']}" target="_blank">{item['title']}</a></strong>
              
              <ul class="fact-list">
                <li><b>事実：</b>{item.get('summary_1', '')}</li>
            """
            if item.get("summary_2") and pd.notna(item["summary_2"]) and str(item["summary_2"]).strip() != "":
                html += f"<li><b>事実：</b>{item['summary_2']}</li>"
            if item.get("summary_3") and pd.notna(item["summary_3"]) and str(item["summary_3"]).strip() != "":
                html += f"<li><b>事実：</b>{item['summary_3']}</li>"
                
            html += f"""
              </ul>
              <div class="meta-line">
                情報源: {item.get('source', '不明')} ｜ 関連アセット: <span style="color: #2962ff; font-weight:bold;">{item.get('related_tickers') if pd.notna(item.get('related_tickers')) and str(item.get('related_tickers')).strip() != "" else 'なし'}</span>
              </div>
            </div>
            """
            
    html += """
    </body>
    </html>
    """
    return html

def send_gmail(html_content: str, count: int):
    """Gmail配信の実行"""
    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メール認証情報(Secrets)が未設定のため、配信をスキップします。")
        return

    msg = MIMEMultipart()
    today_str = datetime.now().strftime("%Y-%m-%d")
    msg["From"] = f"Macro Intelligence Desk <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【マクロ・地政学】{today_str} 重要インテリジェンス {count}選"

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        print("📢 [配信完了] 司令室ニュース配信メールを送信しました。")
    except Exception as e:
        print(f"メール送信エラー: {e}")

def main():
    print("=== [Macro Intelligence Desk V1.0] 起動 ===")
    init_db()

    # 1. ニュース収集
    raw_articles = fetch_rss_feeds()
    polymarket_articles = fetch_polymarket_odds()
    all_articles = raw_articles + polymarket_articles

    # 既存のCSV内の既知URLをロードして「LLMの二重解析を防止」
    existing_urls = set()
    if DB_CSV_PATH.exists():
        try:
            existing_df = pd.read_csv(DB_CSV_PATH)
            existing_urls = set(existing_df["url"].astype(str).str.strip().tolist())
        except Exception:
            pass

    analyzed_list = []
    print(f"\n[解析フェーズ] 新規ニュースのLLM精査を開始します... (全 {len(all_articles)} 件中)")

    # 2. 未登録ニュースのみをLLMで解析
    for idx, art in enumerate(all_articles, 1):
        clean_url = art["url"].strip()
        if clean_url in existing_urls:
            continue  # 重複スキップ

        print(f"  ・新規解析中 [{idx}/{len(all_articles)}]: {art['title'][:25]}...")
        analysis = analyze_article_with_llm(art)
        
        if analysis:
            art.update(analysis)
            analyzed_list.append(art)
            
        time.sleep(3.0)  # 👈 1.0秒 から 3.0秒 に変更して、APIへの負荷をさらにマイルドにします

    # 3. CSVデータベースに新規マージ保存
    added_count = save_articles_to_csv(analyzed_list)
    print(f"➔ 台帳更新完了。新規に {added_count} 本のニュースを CSV に蓄積しました。")

    # 4. 直近のスコア上位15件（かつスコア50点以上の有益ニュース）をGmail配信
    today_important_news = get_today_high_score_news(limit=15)
    valid_news = [n for n in today_important_news if int(n.get("score", 0)) >= 50]

    if valid_news:
        html_mail = build_mail_html(valid_news)
        send_gmail(html_mail, len(valid_news))
    else:
        print("本日の重要ニュース（50点以上）は0件でした。メール送信をスキップします。")


if __name__ == "__main__":
    main()
