# news_aggregator/database.py
from pathlib import Path
import pandas as pd

DB_CSV_PATH = Path(__file__).resolve().parent / "news_intelligence.csv"

def init_db():
    """CSVデータベースの初期化（無ければ新規作成）"""
    if not DB_CSV_PATH.exists():
        df = pd.DataFrame(columns=[
            "published_at", "title", "url", "source", "category", 
            "score", "sentiment", "summary_1", "summary_2", "summary_3", "related_tickers"
        ])
        df.to_csv(DB_CSV_PATH, index=False, encoding="utf-8-sig")

def save_articles_to_csv(new_articles: list[dict]) -> int:
    """新規ニュースの一括マージおよびURL重複排除"""
    if not new_articles:
        return 0
        
    init_db()
    
    # 既存データのロード
    try:
        existing_df = pd.read_csv(DB_CSV_PATH)
    except Exception:
        # 破損等のフォールバック
        existing_df = pd.DataFrame(columns=[
            "published_at", "title", "url", "source", "category", 
            "score", "sentiment", "summary_1", "summary_2", "summary_3", "related_tickers"
        ])
    
    # 新規データのDataFrame化
    new_df = pd.DataFrame(new_articles)
    
    # 結合してURL重複を排除（古いデータを優先し、同一ニュースの多重解析を防止）
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df["url"] = combined_df["url"].astype(str).str.strip()
    
    # URL重複排除
    final_df = combined_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    
    # 差分件数を計算
    added_count = len(final_df) - len(existing_df)
    
    # CSVファイルへの書き戻し
    final_df.to_csv(DB_CSV_PATH, index=False, encoding="utf-8-sig")
    return added_count

def get_today_high_score_news(limit=15) -> list[dict]:
    """直近に収集したニュースから、スコア順にソートして上限数で切り出し"""
    init_db()
    try:
        df = pd.read_csv(DB_CSV_PATH)
        if df.empty:
            return []
            
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        
        # 直近の追加データを優先的にソート
        df_sorted = df.sort_values(by=["score", "published_at"], ascending=[False, False])
        return df_sorted.head(limit).to_dict(orient="records")
    except Exception as e:
        print(f"[エラー] ニュースデータのロードに失敗しました: {e}")
        return []
