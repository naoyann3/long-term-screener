# analyzer.py
import json
import requests
from config import GEMINI_API_KEY

def analyze_article_with_llm(article: dict) -> dict | None:
    """
    LLM (Gemini API) を用いて、感情論・煽りを【100%ノイズカット】し、
    ファクト分類、0-100点スコアリング、強弱判定、関連銘柄を構造化JSONで一瞬で出力します。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # 2つの提案（ファクト3行、100点スコアリング、銘柄抽出、強弱センチメント）を完全にマージしたプロンプト設計
    prompt = f"""
あなたは極めて冷徹な機関投資家のチーフ・リサーチアナリストです。
提供されたニューステキストから、一切の「主観、感情論、投資の煽り文句、ポエジーな表現」を【100%排除（ノイズカット）】し、
市場に影響を与える客観的「事実（エビデンス）」のみを抽出して、指定のJSONフォーマットで出力してください。

【入力ニュース情報】
情報源: {article['source']}
タイトル: {article['title']}
本文: {article['content']}

【出力フォーマット】
以下のキーを持つ純粋なJSONデータのみを出力してください（Markdownの ```json 等の囲みも不要です）。

{{
  "category": "分類を「地政学」「マクロ」「株式」「暗号資産」から1つ選択",
  "score": 0〜100の整数。市場インパクト（大口投資家の資金移動可能性）、関連性、過去の価格反応傾向から冷静に算出（90以上: 即確認の激震、70〜89: 投資判断材料、50〜69: 参考、49以下: ほぼノイズ・無風）,
  "sentiment": "材料の方向性を「強材料」「弱材料」「中立」から1つ選択",
  "summary_1": "1つ目のファクト（3行以内の純粋な事実・数値）",
  "summary_2": "2つ目のファクト（3行以内の純粋な事実・数値。無ければ空欄）",
  "summary_3": "3つ目のファクト（3行以内の純粋な事実・数値。無ければ空欄）",
  "related_tickers": "関連する上場銘柄コード（例: 7203.T, NVDA）や主要通貨（BTC, ETH）。無ければ空欄"
}}
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"  # JSONモードの強制
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]
            # 解析データをパースして返す
            parsed_data = json.loads(text_response.strip())
            return parsed_data
    except Exception as e:
        print(f"    [解析エラー] {article['title'][:15]}... のLLM解析に失敗しました: {e}")
        
    return None