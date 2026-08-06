# news_aggregator/collector.py
import json  # jsonモジュールを追加
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
            for entry in feed.entries[:8]:  # 各フィード最新8件
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
    """Polymarket APIを叩き、予測市場の各種オッズデータをロード"""
    polymarket_data = []
    url = "https://gamma-api.polymarket.com/markets?active=true&limit=5"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            markets = response.json()
            for m in markets:
                title = m.get("question", "")
                
                # ★【最重要修正】：文字列として埋め込まれたJSONをデコードします
                outcomes_raw = m.get("outcomes")
                prices_raw = m.get("outcomePrices")
                
                if outcomes_raw and prices_raw:
                    try:
                        outcomes = json.loads(outcomes_raw)
                        outcome_prices = json.loads(prices_raw)
                        
                        # オッズ（確率）の文字列化
                        odds_str = " / ".join([f"{out}: {float(price)*100:.1f}%" for out, price in zip(outcomes, outcome_prices)])
                        
                        polymarket_data.append({
                            "title": f"【予測市場確率】 {title}",
                            "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                            "published_at": datetime.now().isoformat(),
                            "source": "Polymarket",
                            "content": f"現在の予測市場における確率データ：{odds_str}"
                        })
                    except Exception as inner_err:
                        print(f"    [デコードエラー] {title[:15]}... のパースに失敗: {inner_err}")
                        
        else:
            print(f"  [警告] Polymarket APIがステータスコード {response.status_code} を返しました。")
    except Exception as e:
        print(f"  [警告] Polymarket API のロード中に例外が発生しました: {e}")
        
    return polymarket_data
