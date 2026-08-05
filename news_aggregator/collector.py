# collector.py
import feedparser
import requests
from datetime import datetime
import time
from config import NEWS_FEEDS

def fetch_rss_feeds() -> list[dict]:
    """定義されたすべてのRSSフィードから最新記事を取得（API不要・完全無料）"""
    raw_articles = []
    print("\n[収集フェーズ] 各RSSフィードの取得を開始します...")
    
    for source_name, url in NEWS_FEEDS.items():
        print(f"  ・フィード取得中: {source_name}")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:  # 各フィード最新10件に限定（ノイズ制限）
                published = entry.get("published", datetime.now().isoformat())
                raw_articles.append({
                    "title": entry.title,
                    "url": entry.link,
                    "published_at": published,
                    "source": source_name,
                    "content": entry.get("summary", entry.title) # 本文、無ければタイトル
                })
        except Exception as e:
            print(f"    [警告] {source_name} の取得中にエラーが発生しました: {e}")
            
    return raw_articles

def fetch_polymarket_odds() -> list[dict]:
    """Polymarket APIをダイレクトに叩き、予測市場の金利・地政学オッズをロード [1, 2]"""
    polymarket_data = []
    # 例として「大統領選」「FRB金利決定」などのアクティブ市場を検索
    # APIドキュメントに基づき、クエリパラメータで安全に叩く [1]
    url = "https://gamma-api.polymarket.com/markets?active=true&limit=5"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            markets = response.json()
            for m in markets:
                title = m.get("question", "")
                outcome_prices = m.get("outcomePrices", [])
                outcomes = m.get("outcomes", [])
                
                # オッズ（確率）の文字列化
                odds_str = " / ".join([f"{out}: {float(price)*100:.1f}%" for out, price in zip(outcomes, outcome_prices)])
                
                polymarket_data.append({
                    "title": f"【予測市場確率】 {title}",
                    "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                    "published_at": datetime.now().isoformat(),
                    "source": "Polymarket",
                    "content": f"現在の予測市場における確率データ：{odds_str}"
                })
    except Exception as e:
        print(f"  [警告] Polymarket API のロードに失敗しました: {e}")
        
    return polymarket_data