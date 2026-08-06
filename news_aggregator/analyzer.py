# news_aggregator/analyzer.py
import json
import requests
from config import GEMINI_API_KEY

def analyze_article_with_llm(article: dict) -> dict | None:
    """
    LLM (Gemini API) を用いて、ニュースのノイズカット・スコアリングを実行。
    エラー時には沈黙せず、ステータスコードと原因をログに出力します。
    """
    # 2026年時点で最も安定しているモデル識別子 (gemini-2.5-flash / gemini-1.5-flash) を指定
    # (APIキーが未登録または無効の場合、URLは叩けてもエラーが返ります)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = f"""
あなたは冷徹な機関投資家のチーフ・リサーチアナリストです。
提供されたニュースから、主観、感情論、投資の煽り文句、不要な修飾表現を【100%排除】し、
市場参加者が事実関係を5秒で把握できるよう、客観的な「事実（エビデンス）」のみを抽出して、指定のJSONフォーマットで出力してください。

【入力ニュース情報】
情報源: {article['source']}
タイトル: {article['title']}
本文: {article['content']}

【出力フォーマット】
以下のキーを持つ純粋なJSONデータのみを出力してください（Markdownの ```json 等の囲みコードブロックは一切含めず、プレーンテキストのJSONのみを出力すること）。

{{
  "category": "分類を「地政学」「マクロ」「株式」「暗号資産」から1つ選択",
  "score": 0〜100の整数。市場インパクト、対象への関連性、過去の価格反応傾向から冷静に算出（90以上: 極めて重要、70〜89: 投資判断材料、50〜69: 参考情報、49以下: ノイズ・無風）,
  "sentiment": "材料の方向性を「強材料」「弱材料」「中立」から1つ選択",
  "summary_1": "1つ目のファクト（3行以内の純粋な事実・客観的な数値情報）",
  "summary_2": "2つ目のファクト（3行以内の純粋な事実・客観的な数値情報。無ければ空欄）",
  "summary_3": "3つ目のファクト（3行以内の純粋な事実・客観的な数値情報。無ければ空欄）",
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
        else:
            # ★【最重要追加】：沈黙せず、ステータスコードとAPIからのエラー本文を出力
            print(f"    [APIエラー] ステータス {response.status_code} を返しました。応答: {response.text}")
    except Exception as e:
        print(f"    [解析スキップ] {article['title'][:15]}... のLLM解析に失敗: {e}")
        
    return None
