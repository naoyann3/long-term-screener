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
        existing_df = pd.DataFrame(columns=[
            "published_at", "title", "url", "source", "category", 
            "score", "sentiment", "summary_1", "summary_2", "summary_3", "related_tickers"
        ])
    
    # 新規データのDataFrame化
    new_df = pd.DataFrame(new_articles)
    
    # 結合してURL重複を排除
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
    """
    【Version 1.3 修正：日付フィルターの実装】
    直近48時間以内（土日対策として5件未満なら72時間までフォールバック）に発行された
    最新のニュースのみを、スコアの高い順に抽出して配信します。
    """
    init_db()
    try:
        df = pd.read_csv(DB_CSV_PATH)
        if df.empty:
            return []
            
        # 1. published_at を安全に日付型（UTCタイムゾーン付）に共通パース
        df["pub_date"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        
        # 破損した日付データやパース失敗行を排除
        df_valid = df.dropna(subset=["pub_date"]).copy()
        if df_valid.empty:
            return []

        # 2. 直近48時間（2日間）のニュースをフィルタリング
        now = pd.Timestamp.now(tz="UTC")
        forty_eight_hours_ago = now - pd.Timedelta(hours=48)
        
        df_filtered = df_valid[df_valid["pub_date"] >= forty_eight_hours_ago].copy()
        
        # 🌟【自律フォールバック】：ニュースが極端に少なければ、直近72時間（3日間）に広げて再検索
        if len(df_filtered) < 5:
            seventy_two_hours_ago = now - pd.Timedelta(hours=72)
            df_filtered = df_valid[df_valid["pub_date"] >= seventy_two_hours_ago].copy()
            
        # 最終フォールバック：それすら空、または数日ぶりの実行なら、CSVに登録されている最新20件から抽出
        if df_filtered.empty:
            df_filtered = df_valid.sort_values(by="pub_date", ascending=False).head(20).copy()

        # 3. フィルターされた最新ニュースの中から、スコアの高い順にソートして切り出し
        df_filtered["score"] = pd.to_numeric(df_filtered["score"], errors="coerce").fillna(0).astype(int)
        df_sorted = df_filtered.sort_values(by=["score", "pub_date"], ascending=[False, False])
        
        # 不要になった日付オブジェクト列を削除して、純粋な辞書型として返却
        df_sorted = df_sorted.drop(columns=["pub_date"])
        return df_sorted.head(limit).to_dict(orient="records")
        
    except Exception as e:
        print(f"[エラー] 最新ニュース抽出中に例外が発生しました: {e}")
        return []
