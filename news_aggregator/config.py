# config.py
import os

# APIキー設定 (Gemini API または OpenAI API のいずれかを利用可能)
# ここでは例としてGemini API（無料枠あり）を想定しますが、OpenAIに変更も容易です。
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# Gmail配信設定 (GMAIL_APP_PASSWORDは track_long_term_positions.py と共通化可能)
GMAIL_USER = os.environ.get("GMAIL_USER", "your_gmail@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "your_app_password")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "recipient@gmail.com")

# ニュースRSSおよびX-RSSフィードの定義
NEWS_FEEDS = {
    "日経マクロ": "https://www.nikkei.com/rss/news/macro.xml",
    "GoogleNews_金融": "https://news.google.com/rss/search?q=金融政策+OR+金利+OR+雇用統計&hl=ja&gl=JP&ceid=JP:ja",
    "CoinDesk_クリプト": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    # 非公式X-RSSコンバータ（NitterインスタンスやRSS.app等で取得したXのURL）
    "X_goto_finance": "https://nitter.privacydev.net/goto_finance/rss",  # 後藤達也氏（例）
    "X_WuBlockchain": "https://nitter.privacydev.net/WuBlockchain/rss",  # ウー氏（例）
}

# Polymarketで監視したい主要マーケットID
POLYMARKET_IDS = {
    "Fed_September_Rate_Cut": "905206",  # 例: 9月利下げ予測などのマーケットID（APIから動的取得も可能）
}
