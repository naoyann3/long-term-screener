# recover_github_artifacts.py (Version 1.0 - Auto Artifacts Recovery Engine)
import os
import requests
from pathlib import Path
import zipfile
import io
import time

# あなたのGitHubのアカウント名とリポジトリ名
GITHUB_OWNER = "naoyann3"
GITHUB_REPO = "long-term-screener"
OUTPUT_DIR = Path("results/long_term_watchlists")

# 🔒 【最重要・設定】：GitHubの個人用アクセストークン（PAT）をここに貼り付けます
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or "YOUR_PERSONAL_ACCESS_TOKEN_HERE"


def recover_artifacts() -> None:
    if GITHUB_TOKEN == "YOUR_PERSONAL_ACCESS_TOKEN_HERE":
        print("🚨 エラー: 成果物を自動ダウンロードするには、GitHubのアクセストークン（PAT）の設定が必要です。")
        print("\n🔑 【1分でできる、GitHubアクセストークン（PAT）の作成手順】:")
        print("  1. ブラウザで GitHub の右上アイコン ➔ 『Settings』 を開く")
        print("  2. 左サイドバーの一番下にある 『Developer Settings』 ➔ 『Personal access tokens』 ➔ 『Tokens (classic)』 を開く")
        print("  3. 右上の 『Generate new token』 ➔ 『Generate new token (classic)』 をクリック")
        print("  4. Noteに「Screener Token」等と入力し、有効期限（Expiration）を適当に設定（7days等）")
        print("  5. スコープ（Select scopes）の 【 ☐ repo 】 (一番上のチェックボックス) にチェックを入れる")
        print("  6. 一番下の緑の 『Generate token』 ボタンをクリック")
        print("  7. 画面に表示される 『ghp_...』 で始まる長い暗号キーをコピーし、このコードの GITHUB_TOKEN に貼り付けてください。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    # 過去100件の成果物リストを一挙に取得
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/artifacts?per_page=100"
    
    print("=== GitHubから過去のActions成果物リストを調査中... ===")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"エラー: GitHub APIへの接続に失敗しました (ステータス: {response.status_code}): {response.text}")
            return
            
        data = response.json()
        artifacts = data.get("artifacts", [])
        
        # 今回のスクリーナー結果（long-term-screener-results）を含むものだけに自動フィルタリング
        target_artifacts = [a for a in artifacts if "long-term-screener-results" in a["name"]]
        
        if not target_artifacts:
            print("過去のスクリーニング成果物（ZIP）が見つかりません。すでに保存期間（90日）を過ぎて消滅している可能性があります。")
            return
            
        print(f"➔ 救出可能な過去データ: 【 {len(target_artifacts)} 日分 】 を検出しました (※直近90日間の生存データ)")
        print("\n=== クラウドからの一括自動ダウンロード ＆ メモリ上での超高速解凍を開始します ===")
        
        success_count = 0
        for idx, a in enumerate(target_artifacts, 1):
            name = a["name"]  # 例：long-term-screener-results-2026-07-07
            download_url = a["archive_download_url"]
            
            # ファイル名から「日付」の部分を自動抽出
            date_part = name.replace("long-term-screener-results-", "")
            
            print(f"  [{idx}/{len(target_artifacts)}] 自動救出中... {date_part}")
            
            # ZIPファイルのバイナリデータをメモリ上に直接ダウンロード
            res_dl = requests.get(download_url, headers=headers)
            if res_dl.status_code == 200:
                # メモリ上でZIPを展開し、中にある long_term_watchlist.csv を日付付きの名前に書き換えて抽出保存
                with zipfile.ZipFile(io.BytesIO(res_dl.content)) as zip_file:
                    for filename in zip_file.namelist():
                        if filename == "long_term_watchlist.csv":
                            data_content = zip_file.read(filename)
                            # 保存名: 例 /results/long_term_watchlists/2026-07-07_lt_v1_recovered.csv
                            dest_path = OUTPUT_DIR / f"{date_part}_lt_v1_recovered.csv"
                            dest_path.write_bytes(data_content)
                            success_count += 1
            else:
                print(f"    ➔ ❌ [ダウンロード失敗] ステータス: {res_dl.status_code}")
                
            time.sleep(0.5)  # API制限を回避するためのウェイト
            
        print(f"\n🎉 [救出ミッション完了] 合計 【 {success_count} 日分 】 の過去合格データを全自動で回収・解凍し、{OUTPUT_DIR} にマージ完了しました！")
        
    except Exception as e:
        print(f"例外エラーが発生しました: {e}")


if __name__ == "__main__":
    recover_artifacts()