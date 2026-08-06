# news_aggregator/config.py
import os

# APIキー設定
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# Gmail配信設定
GMAIL_USER = os.environ.get("GMAIL_USER", "your_gmail@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "your_app_password")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "recipient@gmail.com")

# ニュースRSSおよび非公式X-RSSフィードの定義（海外の一等地のマクロソースを追加）
NEWS_FEEDS = {
    "日経マクロ": "https://www.nikkei.com/rss/news/macro.xml",
    "GoogleNews_金融": "https://news.google.com/rss/search?q=金融政策+OR+金利+OR+雇用統計&hl=ja&gl=JP&ceid=JP:ja",
    "CoinDesk_クリプト": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    
    # 🌟【Version 1.1 追加】：海外の一等地マクロ・中銀金融政策RSS
    "FinancialJuice_マクロ": "https://www.financialjuice.com/rss.ashx",
    "InvestingCom_中銀": "https://www.investing.com/rss/news_285.rss",
    "IMF_国際金融": "https://www.imf.org/en/News/RSS",
    
    # 非公式X-RSSコンバータ
    "X_goto_finance": "https://nitter.privacydev.net/goto_finance/rss",
    "X_WuBlockchain": "https://nitter.privacydev.net/WuBlockchain/rss",
}
