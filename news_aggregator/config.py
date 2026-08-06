# news_aggregator/config.py
import os

# APIキー設定 (Gemini API をデフォルトとします。OpenAI API等へもプロンプト設計は流用可能です)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# Gmail配信設定 (GitHub Secrets から自動インジェクションされます)
GMAIL_USER = os.environ.get("GMAIL_USER", "your_gmail@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "your_app_password")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "recipient@gmail.com")

# ニュースRSSおよび非公式X-RSSフィードの定義
NEWS_FEEDS = {
    "日経マクロ": "https://www.nikkei.com/rss/news/macro.xml",
    "GoogleNews_金融": "https://news.google.com/rss/search?q=金融政策+OR+金利+OR+雇用統計&hl=ja&gl=JP&ceid=JP:ja",
    "CoinDesk_クリプト": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    # 非公式X-RSSコンバータ（NitterインスタンスやRSS.app等の無料フィードURLを想定）
    "X_goto_finance": "https://nitter.privacydev.net/goto_finance/rss",  # 後藤達也氏
    "X_WuBlockchain": "https://nitter.privacydev.net/WuBlockchain/rss",  # Wu Blockchain
}
