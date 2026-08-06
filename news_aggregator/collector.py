# news_aggregator/collector.py
import feedparser
import requests
from datetime import datetime
from config import NEWS_FEEDS

def fetch_rss_feeds() -> list[dict]:
    """定義された全RSSフィードから最新記事を取得"""
    raw_articles = []
    print("\n[収集フェーズ] 各RSSフィードの取得を開始します...")
    
    for source_name, url in NEWS_FEEDS.items():
        print(f"  ・フィード取得中: {source_name}")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:  # 各フィード最新8件に制限（無用なトラフィック・APIコールを制限）
                published = entry.get("published", datetime.now().isoformat())
                raw_articles.append({
                    "title": entry.title,
                    "url": entry.link,
                    "published_at": published,
                    "source": source_name,
                    "content": entry.get("summary", entry.title)
                })
        except Exception as e:
            print(f"    [警告] {source_name} の取得中に例外が発生しました: {e}")
            
    return raw_articles

def fetch_polymarket_odds() -> list[dict]:
    """Polymarket APIを叩き、予測市場の各種オッズデータをロード [1, 2]"""
    polymarket_data = []
    # 活発な予測市場の直近データを5件取得 [1]
    url = "https://gamma-api.polymarket.com/markets?active=true&limit=5"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            markets = response.json()
            for m in markets:
                title = m.get("question", "")
                outcome_prices = m.get("outcomePrices", [])
                outcomes = m.get("outcomes", [])
                
                # 金利引き下げ確率、選挙オッズ等の表記を生成
                odds_str = " / ".join([f"{out}: {float(price)*100:.1f}%" for out, price in zip(outcomes, outcome_prices)])
                
                polymarket_data.append({
                    "title": f"【予測市場確率】 {title}",
                    "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                    "published_at": datetime.now().isoformat(),
                    "source": "Polymarket",
                    "content": f"現在の予測市場における確率データ：{odds_str}"
                })
    except Exception as e:
        print(f"  [警告] Polymarket API のロード中に例外が発生しました: {e}")
        
    return polymarket_data
