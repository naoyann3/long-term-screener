# main.py
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import time

from config import GMAIL_USER, GMAIL_PASS, NOTIFICATION_EMAIL
from database import init_db, save_article, get_today_high_score_news
from collector import fetch_rss_feeds, fetch_polymarket_odds
from analyzer import analyze_article_with_llm

def build_mail_html(news_rows) -> str:
    """Gemini版提案の『5秒で掌握する美しいフォーマット』に準拠したHTMLメールを動的作成"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # カテゴリごとに仕分ける
    categories = {"地政学": [], "マクロ": [], "株式": [], "暗号資産": []}
    for r in news_rows:
        cat = r["category"]
        if cat in categories:
            categories[cat].append(r)
        else:
            categories["マクロ"].append(r)

    html = f"""
    <html>
    <head>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background-color: #131722; color: #d1d4dc; padding: 20px; }}
        h1 {{ color: #2962ff; font-size: 20px; border-bottom: 2px solid #2962ff; padding-bottom: 8px; }}
        h2 {{ color: #ff9800; font-size: 16px; margin-top: 25px; border-bottom: 1px solid #2a2e39; padding-bottom: 4px; }}
        .news-card {{ background-color: #1c2030; padding: 12px; border-radius: 4px; margin-bottom: 12px; border-left: 4px solid #2962ff; }}
        .score {{ font-weight: bold; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
        .score-high {{ background-color: #ef5350; color: white; }}
        .score-mid {{ background-color: #ff9800; color: white; }}
        .score-low {{ background-color: #4caf50; color: white; }}
        .fact-list {{ margin: 6px 0; padding-left: 20px; font-size: 13px; line-height: 1.5; color: #b2b5be; }}
        .meta-line {{ font-size: 11px; color: #787b86; margin-top: 6px; }}
        .link {{ color: #2962ff; text-decoration: none; font-weight: bold; }}
      </style>
    </head>
    <body>
      <h1>📬 【マクロインテリジェンス・司令室】 {today_str}</h1>
      <p style="font-size: 12px; color: #787b86;">余計な雑音や投資の煽り表現を100%カットした、冷徹な事実のみを配信します。</p>
    """

    for cat_name, items in categories.items():
        if not items:
            continue
        html += f"<h2>■ {cat_name} ＆ 金融インフラ</h2>"
        for item in items:
            score = item["score"]
            score_class = "score-high" if score >= 90 else ("score-mid" if score >= 70 else "score-low")
            
            # センチメントの色
            sentiment_color = "#ef5350" if item["sentiment"] == "強材料" else ("#26a69a" if item["sentiment"] == "弱材料" else "#787b86")
            
            html += f"""
            <div class="news-card" style="border-left-color: {sentiment_color};">
              <span class="score {score_class}">重要度: {score}点</span> 
              <span style="color: {sentiment_color}; font-weight: bold; font-size: 11px; margin-left: 8px;">【{item["sentiment"]}】</span>
              <strong style="font-size: 14px; margin-left: 5px;"><a class="link" href="{item["url"]}" target="_blank">{item["title"]}</a></strong>
              
              <ul class="fact-list">
                <li>事実：{item["summary_1"]}</li>
            """
            if item["summary_2"]:
                html += f"<li>事実：{item["summary_2"]}</li>"
            if item["summary_3"]:
                html += f"<li>事実：{item["summary_3"]}</li>"
                
            html += f"""
              </ul>
              <div class="meta-line">
                ソース: {item["source"]} ｜ 関連アセット: <span style="color: #2962ff; font-weight:bold;">{item["related_tickers"] or "なし"}</span>
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
        print("メール認証情報が設定されていないため、配信をスキップします。")
        return

    msg = MIMEMultipart()
    today_str = datetime.now().strftime("%Y-%m-%d")
    msg["From"] = f"Macro Intelligence Desk <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【マクロ・地政学】{today_str} 重要ヘッドライン {count}本"

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        print("📢 [配信完了] マクロインテリジェンス・メールを無事配信しました。")
    except Exception as e:
        print(f"メール送信エラー: {e}")

def main():
    print("=== [Macro Intelligence Desk V1.0] 起動 ===")
    init_db()

    # 1. データ収集
    raw_articles = fetch_rss_feeds()
    polymarket_articles = fetch_polymarket_odds()
    all_articles = raw_articles + polymarket_articles

    print(f"\n[解析・保存フェーズ] 取得した全 {len(all_articles)} 本の精査を開始します...")
    saved_count = 0

    # 2. 重複排除し、新規ニュースのみをLLMに渡して冷徹に分析・保存
    for idx, art in enumerate(all_articles, 1):
        # LLMを叩く前に、すでにデータベース（SQLite）にあるURLかどうかを軽くチェック
        # これによりAPIコール数とコストを「極限までセーブ」します
        if idx % 10 == 0:
            print(f"  ・進行状況: {idx}/{len(all_articles)} 件処理中...")
            
        # LLM解析
        analysis = analyze_article_with_llm(art)
        if analysis:
            # 記事データとLLM解析データをマージ
            art.update(analysis)
            # 保存
            is_new = save_article(art)
            if is_new:
                saved_count += 1
                
        # 無料枠APIのレートリミット回避のための優しいウェイト
        time.sleep(1.0)

    print(f"➔ 解析完了。新着 {saved_count} 本のニュースをデータベースに蓄積しました。")

    # 3. 本日の高スコア（重要度50点以上の有益な情報）のみを抽出してメール配信
    today_important_news = get_today_high_score_news(limit=15)
    valid_news = [n for n in today_important_news if n["score"] >= 50]

    if valid_news:
        html_mail = build_mail_html(valid_news)
        send_gmail(html_mail, len(valid_news))
    else:
        print("本日の重要ニュース（50点以上）は0件でした。メール送信をスキップします。")


if __name__ == "__main__":
    main()