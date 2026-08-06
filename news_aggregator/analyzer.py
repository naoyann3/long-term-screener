# news_aggregator/analyzer.py
import json
import time
import requests
from config import GEMINI_API_KEY

def analyze_article_with_llm(article: dict, max_retries=3) -> dict | None:
    """
    LLM (Gemini API) を用いて、ニュースのノイズカット・スコアリングを実行。
    ※入力ニュースが【英語】であっても、出力されるJSONデータ（事実、分類、強弱判定）は
    すべて流暢かつ冷徹な【日本語】に翻訳・要約して一貫して出力するようプロンプトを補強。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
あなたは冷徹な機関投資家のチーフ・リサーチアナリストです。
提供されたニュースから、主観、感情論、投資の煽り文句、不要な修飾表現を【100%排除】し、
市場参加者が事実関係を5秒で把握できるよう、客観的な「事実（エビデンス）」のみを抽出して、指定のJSONフォーマットで出力してください。

【⚠️超重要：言語統制ルール】
入力されるニュースの情報源、タイトル、本文が「英語」で書かれている場合であっても、
出力するJSON内のテキスト（summary_1〜3、category、sentiment、related_tickers 等のすべてのフィールドの値）は、
必ず流暢でプロフェッショナルな【日本語】で翻訳・要約した上で出力してください。

【入力ニュース情報】
情報源: {article['source']}
タイトル: {article['title']}
本文: {article['content']}

【出力フォーマット】
以下のキーを持つ純粋なJSONデータのみを出力してください（Markdownの ```json 等の囲みコードブロックは一切含めず、プレーンテキストのJSONのみを出力すること）。

{{
  "category": "分類を「地政学」「マクロ」「株式」「暗号資産」から1つ選択（必ず日本語で「地政学」「マクロ」「株式」「暗号資産」のいずれかにすること）",
  "score": 0〜100の整数。市場インパクト、対象への関連性、過去の価格反応傾向から冷静に算出（90以上: 極めて重要、70〜89: 投資判断材料、50〜69: 参考情報、49以下: ノイズ・無風）,
  "sentiment": "材料の方向性を日本語で「強材料」「弱材料」「中立」から1つ選択",
  "summary_1": "1つ目のファクト（必ず日本語で、3行以内の純粋な事実・客観的な数値情報）",
  "summary_2": "2つ目のファクト（必ず日本語で、3行以内の純粋な事実・客観的な数値情報。無ければ空欄）",
  "summary_3": "3つ目のファクト（必ず日本語で、3行以内の純粋な事実・客観的な数値情報。無ければ空欄）",
  "related_tickers": "関連する上場銘柄コード（例: 7203.T, NVDA）や主要通貨（BTC, ETH）。特定できなければ空欄"
}}
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    # 最大3回のリトライループ
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                res_json = response.json()
                text_response = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                if text_response.startswith("```"):
                    text_response = text_response.split("\n", 1)[1]
                    if text_response.endswith("```"):
                        text_response = text_response.rsplit("\n", 1)[0]
                        
                parsed_data = json.loads(text_response.strip())
                return parsed_data
                
            elif response.status_code == 429:
                wait_sec = 33
                print(f"    [警告] 429 クォータ制限（5 RPM）超過を検知しました。{wait_sec}秒間待機し、自動リトライします (試行 {attempt + 1}/{max_retries})...")
                time.sleep(wait_sec)
                continue
                
            else:
                print(f"    [APIエラー] ステータス {response.status_code} を返しました。応答: {response.text}")
                break
                
        except Exception as e:
            print(f"    [解析スキップ] {article['title'][:15]}... のLLM解析に失敗: {e}")
            break
            
    return None
