# dashboard.py (Version 1.7 - Streamlit Custom Interactive Plotly Edition)
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ページ基本設定：黒テーマを引き立たせるワイドレイアウト
st.set_page_config(page_title="Sniper OS - Central Command", layout="wide", initial_sidebar_state="collapsed")

# CSSでマークダウンや全体のスタイルをTradingView風のダークモード（#131722）に完全統一
st.markdown("""
    <style>
    .reportview-container { background: #131722; }
    h1, h2, h3, h4 { color: #2962ff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    div.stButton > button:first-child { background-color: #2962ff; color: white; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Sniper OS - Central Command Dashboard")
st.caption("毎日21:00に自動更新される、あなたの保有・監視株の一元管理コックピット")

# 1. GitHubに自動保存されている最新の健康診断データをロード
RAW_CSV_URL = "https://raw.githubusercontent.com/naoyann3/long-term-screener/master/long_term_tracking.csv"

@st.cache_data(ttl=60) # 1分間キャッシュ
def load_data():
    try:
        return pd.read_csv(RAW_CSV_URL)
    except Exception:
        # フィードバックとしてローカルのCSVを読み込む
        return pd.read_csv("long_term_tracking.csv")

df = load_data()

if df.empty:
    st.error("現在、トラッキングデータがありません。")
else:
    # 左右の2ペイン（左: 11, 右: 9）に画面を美しく分割
    col_left, col_right = st.columns([11, 9])
    
    with col_left:
        st.subheader("📊 アクティブ・ポートフォリオ健康診断")
        
        # 表示用のセレクトボックスを生成
        options_list = []
        for idx, row in df.iterrows():
            options_list.append(f"{row['ティッカー']} : {row['銘柄名']} ({row['判定']})")
            
        selected_option = st.selectbox(
            "🔍 チャートを分析する銘柄をリストから選択してください（一瞬で同期します）:",
            options=options_list,
            index=0
        )
        
        # 選択されたティッカーを特定
        selected_ticker = selected_option.split(" : ")[0]
        selected_row = df[df["ティッカー"] == selected_ticker].iloc[0]
        
        # 綺麗な並びでテーブル表示
        display_cols = ["判定", "警戒スコア", "ティッカー", "銘柄名", "種別", "取得日", "取得単価", "終値", "取得単価比(%)", "25日線乖離(%)", "75日線乖離(%)", "推奨アクション", "警戒サイン", "メモ"]
        existing_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[existing_cols],
            use_container_width=True,
            hide_index=True
        )

    with col_right:
        st.subheader(f"📊 {selected_row['銘柄名']} ({selected_ticker}) の多次元定点観測")
        clean_code = selected_ticker.split(".")[0]
        
        # ==========================================
        # 📈 【Version 1.7 最終破壊的イノベーション】：Plotly による本格的なローカル自律描画
        # ==========================================
        st.markdown(f"#### 📸 [📈 TradingView で大画面チャートを開く (TSE:{clean_code})](https://jp.tradingview.com/chart/?symbol=TSE:{clean_code})")
        
        # 1. yfinanceから日足データをロード（ローカル環境のためIPブロックの規制を受けずに100%安全に一瞬で取得）
        try:
            with st.spinner("リアルタイム株価データをレンダリング中..."):
                ticker_obj = yf.Ticker(selected_ticker)
                hist = ticker_obj.history(period="12mo", interval="1d", auto_adjust=False)
                
            if hist.empty:
                st.error("株価データのロードに失敗しました。")
            else:
                # 25日・75日・200日移動平均線を算出
                hist["ma25"] = hist["Close"].rolling(25).mean()
                hist["ma75"] = hist["Close"].rolling(75).mean()
                hist["ma200"] = hist["Close"].rolling(200).mean()
                
                # サブプロットの作成（上段：ローソク足 ＆ MA、下段：出来高）
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03, 
                    row_width=[0.25, 0.75]  # 比率
                )
                
                # 上段に「ローソク足」を配置（陽線＝赤、陰線＝青緑の TradingView 仕様）
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
                    name="株価",
                    increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
                    increasing_fillcolor='#ef5350', decreasing_fillcolor='#26a69a'
                ), row=1, col=1)
                
                # 上段に「移動平均線（25日:青、75日:オレンジ、200日:緑）」を追加
                fig.add_trace(go.Scatter(x=hist.index, y=hist["ma25"], name="25日線", line=dict(color="#2962ff", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist["ma75"], name="75日線", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist["ma200"], name="200日線", line=dict(color="#4caf50", width=1.5)), row=1, col=1)
                
                # 下段に「出来高の棒グラフ」を配置
                fig.add_trace(go.Bar(
                    x=hist.index, y=hist["Volume"], name="出来高",
                    marker=dict(color="#3f4251")
                ), row=2, col=1)
                
                # 3. レイアウトの徹底的な美化（TradingViewと完璧に調和するスタイリッシュ・ダークテーマ）
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#131722",
                    plot_bgcolor="#131722",
                    xaxis_rangeslider_visible=False, # スライダーを非表示にして表示領域を拡大
                    margin=dict(l=5, r=5, t=5, b=5),
                    height=340,
                    showlegend=False
                )
                
                # X軸とY軸のグリッド線を少し暗くして視認性を最大化
                fig.update_xaxes(showgrid=True, gridcolor="#2a2e39", row=1, col=1)
                fig.update_yaxes(showgrid=True, gridcolor="#2a2e39", row=1, col=1)
                fig.update_xaxes(showgrid=True, gridcolor="#2a2e39", row=2, col=1)
                fig.update_yaxes(showgrid=True, gridcolor="#2a2e39", row=2, col=1)
                
                # 描画
                st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
                
        except Exception as e:
            st.error(f"チャートの自律描画中に例外が発生しました: {e}")
        
        # 3. 株探決算 ＆ Yahoo!掲示板へのスマートダイレクトゲート
        st.markdown("### 🔗 クオンツ・ダイレクトポータル")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.markdown(f'<a href="https://kabutan.jp/stock/finance?code={clean_code}" target="_blank" style="display:block; text-align:center; background-color:#2962ff; color:white; padding:10px; border-radius:4px; text-decoration:none; font-weight:bold; font-size:12px;">📊 株探で決算を分析</a>', unsafe_allow_html=True)
        with col_btn2:
            st.markdown(f'<a href="https://finance.yahoo.co.jp/quote/{clean_code}/forum" target="_blank" style="display:block; text-align:center; background-color:#7b00cc; color:white; padding:10px; border-radius:4px; text-decoration:none; font-weight:bold; font-size:12px;">🏦 Yahoo!掲示板で大衆心理</a>', unsafe_allow_html=True)

        # 4. 健康診断詳細カルテの描画
        warning_sign = selected_row['警戒サイン'] if pd.notna(selected_row['警戒サイン']) else 'なし'
        
        st.markdown(f"""
        <div style="background-color:#1c2030; padding:12px; border-radius:4px; border-left:4px solid #2962ff; font-size:12px; line-height:1.5; margin-top:10px;">
          <strong>🏥 本日の健康診断カルテ: {clean_code}</strong><br>
          ・現在の判定  : <span style="color:#2962ff; font-weight:bold;">{selected_row['判定']}</span> (警戒スコア: {selected_row['警戒スコア']}点)<br>
          ・現在終値    : <strong>{selected_row['終値']} 円</strong> (取得単価比: {selected_row['取得単価比(%)']})<br>
          ・25日線乖離 : {selected_row['25日線乖離(%)']} / 75日線乖離: {selected_row['75日線乖離(%)']}<br>
          ・推奨アクション : <span style="background-color:#2962ff; padding:2px 6px; border-radius:3px; font-weight:bold;">{selected_row['推奨アクション']}</span><br>
          ・警戒サイン  : <span style="color:#ff0050;">{warning_sign}</span>
        </div>
        """, unsafe_allow_html=True)
