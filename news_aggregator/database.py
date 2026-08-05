# database.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "news_intelligence.db"

def init_db():
    """データベースの初期化とテーブル作成"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT,
            title TEXT,
            url TEXT UNIQUE,
            source TEXT,
            category TEXT,       -- マクロ, 株式, 暗号資産, 地政学
            score INTEGER,       -- 0〜100
            sentiment TEXT,      -- 強材料, 弱材料, 中立
            summary_1 TEXT,      -- ファクト1
            summary_2 TEXT,      -- ファクト2
            summary_3 TEXT,      -- ファクト3
            related_tickers TEXT, -- 関連銘柄 (カンマ区切り)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_article(article: dict) -> bool:
    """ニュースを保存（URL重複時はスキップ）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO articles (
                published_at, title, url, source, category, score, sentiment, summary_1, summary_2, summary_3, related_tickers
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article.get("published_at"),
            article.get("title"),
            article.get("url"),
            article.get("source"),
            article.get("category", "マクロ"),
            article.get("score", 50),
            article.get("sentiment", "中立"),
            article.get("summary_1", ""),
            article.get("summary_2", ""),
            article.get("summary_3", ""),
            article.get("related_tickers", "")
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # 重複（すでにURLが保存されている）
        return False
    finally:
        conn.close()

def get_today_high_score_news(limit=10):
    """今日の高スコアニュースを取得"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM articles 
        WHERE published_at >= date('now', '-1 day')
        ORDER BY score DESC, published_at DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows