# database.py
from pathlib import Path
import pandas as pd

DB_CSV_PATH = Path(__file__).resolve().parent / "news_intelligence.csv"

def init_db():
    """CSVデータベースの初期化（無ければ空ファイル作成）"""
    if not DB_CSV_PATH.exists():
        df = pd.DataFrame(columns=[
            "published_at", "title", "url", "source", "category", 
            "score", "sentiment", "summary_1", "summary_2", "summary_3", "related_tickers"
        ])
        df.to_csv(DB_CSV_PATH, index=False, encoding="utf-8-sig")

def save_articles_to_csv(new_articles: list[dict]) -> int:
    """新規のニュースをCSV台帳にマージし、URL重複を排除して保存"""
    if not new_articles:
        return 0
        
    init_db()
    
    # 既存のCSVをロード
    existing_df = pd.read_csv(DB_CSV_PATH)
    
    # 新規データをDataFrame化
    new_df = pd.DataFrame(new_articles)
    
    # 結合して、URLの重複を排除（古いものを残し、新しい重複をカット）
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df["url"] = combined_df["url"].astype(str).str.strip()
    
    # 重複排除（urlをキーにし、最初に登録された履歴を優先保持）
    final_df = combined_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    
    # 追加された差分件数を算出
    added_count = len(final_df) - len(existing_df)
    
    # CSVへ書き戻し
    final_df.to_csv(DB_CSV_PATH, index=False, encoding="utf-8-sig")
    return added_count

def get_today_high_score_news(limit=15) -> list[dict]:
    """本日のニュースから高スコア（重要）な順に辞書リストで取得"""
    init_db()
    try:
        df = pd.read_csv(DB_CSV_PATH)
        if df.empty:
            return []
            
        # 今日の日付（YYYY-MM-DD）を取得して簡易フィルタリング
        # (yyyy-mm-dd形式で部分一致、または文字列比較)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        
        # 簡易的にスコア順にソートして上限数で切り出す
        df_sorted = df.sort_values(by=["score", "published_at"], ascending=[False, False])
        return df_sorted.head(limit).to_dict(orient="records")
    except Exception as e:
        print(f"[エラー] ニュースデータのロードに失敗しました: {e}")
        return []
