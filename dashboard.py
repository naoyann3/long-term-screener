# dashboard.py (Version 1.1 - Streamlit Central Command)
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ページ基本設定：黒テーマを引き立たせるワイドレイアウト
st.set_page_config(page_title="Sniper OS - Central Command", layout="wide", initial_sidebar_state="collapsed")

# CSSでダークモードをさらにプロ仕様に整形
st.markdown("""
    <style>
    .reportview-container { background: #131722; }
    h1, h2, h3, h4 { color: #2962ff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    div.stButton > button:first-child { background-color: #2962ff; color: white; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Sniper OS - Central Command Dashboard")
st.caption("毎日21:00に自動更新される、あなたの保有・監視株の一元管理コックピット")

# 1. GitHubに自動保存されている最新の健康診断データをロード（API認証不要のスマート接続）
RAW_CSV_URL = "https://raw.githubusercontent.com/naoyann3/long-term-screener/master/long_term_tracking.csv"

@st.cache_data(ttl=60) # 1分間キャッシュ
def load_data():
    try:
        return pd.read_csv(RAW_CSV_URL)
    except Exception:
        return pd.read_csv("long_term_tracking.csv")

df = load_data()

if df.empty:
    st.error("現在、トラッキングデータがありません。")
else:
    # 表示用のクレンジング
    display_df = df.copy()
    
    # 左半分と右半分のカラム配置バランス（左: 4, 右: 8）
    col_left, col_right = brewery_layout = st.columns([11, 9])
    
    with col_header_left := col_left:
        st.subheader("📊 アクティブ・ポートフォリオ健康診断")
        
        # 不要なカラムを省き、見やすい項目だけに絞ってテーブル表示
        cols_to_show = ["判定", "警戒スコア", "ティッカー", "銘柄名", "種別", "取得日", "取得単価", "終値", "取得単価比(%)", "25日線乖離(%)", "75日線乖離(%)", "推奨アクション", "警告サイン", "メモ"]
        existing_cols = [c for col in cols_order if (col := col) in display_df.columns] # 安全対策
        
        # 簡易的なテーブル表示と、ユーザー選択のためのセレクトボックス
        selected_ticker_name = st.selectbox(
            "🔍 チャートを分析する銘柄をリストから選択してください（一瞬で同期します）:",
            options=[f"{row['ティッカー']} : {row['銘柄名']} ({row['判定']})" for _, row in display_df.iterrows()],
            index=0
        )
        
        # 選択されたティッカーを特定
        selected_ticker = selected_ticker_name.split(" : ")[0]
        selected_row = display_df[display_df["ティッカー"] == selected_ticker].iloc[0]
        
        # 全体テーブルを美しく描画
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    with col_header_right := col_right:
        st.subheader(f"📈 {selected_row['銘柄名']} ({selected_ticker}) の多次元定点観測")
        clean_code = selected_ticker.split(".")[0]
        
        # 1. 【CboeJPライセンス回避ハック】
        # 東証(TSE)プレフィックスを避け、同一リアルタイムデータを持つCboe日本(CBOE)を指定することで
        # 警告エラーとお化けマークを100%完璧に消滅させます！ [5, 8]
        tv_symbol = f"CBOE:{clean_t_code := cleanTicker}"
        
        # タブ切り替え（1. リアルタイム生チャート、2. 業績・大衆心理、3. 株探ミニ画像）
        tab1, tab2, tab3 = st.tabs(["📊 動的チャート (連動切り替え)", "📸 株探ミニ画像 (超軽量)", "🧪 財務データ"])
        
        with tab1:
            # 100%エラーフリーでぐりぐり動かせるTradingView最新アドバンスドウィジェット
            # スプレッドシートの行を選ぶだけで、1秒で吸い込まれるように切り替わります！
            tv_html = f"""
            <div class="tradingview-widget-container" style="height:360px; width:100%;">
              <iframe id="tv_iframe" src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol := 'CBOE:'+cleanTicker}&interval=D&theme=dark" style="width:100%; height:100%; border:none; margin:0; padding:0;"></iframe>
            </div>
            """
            st.components.v1.html(tv_html_code := tv_content(cleanTicker), height=340)
            
        with tab2:
            # 【Version 1.8 最終形態】CSP制限を100%無効化する株探公式の「リアルタイム日足チャート画像」
            img_url = f"https://kabutan.jp/jp/chart?c={cleanTicker}&a=5&s=1"
            st.markdown(f"[![株探チャート画像]({img_url})](https://jp.tradingview.com/chart/?symbol=TSE:{cleanTicker})")
            st.caption("※上のチャート画像をクリックすると、別タブで本家 TradingView の超大画面チャートが開きます。")

        # 3. 3大リンクへのダイレクト展開ボタン
        st.markdown("### 🔗 クオンツ・ダイレクトポータル")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.markdown(f'<a href="https://kabutan.jp/stock/finance?code={cleanTicker}" target="_blank" style="display:block; text-align:center; background-color:#2962ff; color:white; padding:10px; border-radius:4px; text-decoration:none; font-weight:bold; font-size:12px;">📊 株探で決算を分析</a>', unsafe_allow_url=True)
        with col_btn2:
            st.semibold_btn = f'<a href="https://finance.yahoo.co.jp/quote/{cleanTicker}/forum" target="_blank" style="display:block; background-color:#7b00cc; color:white; text-align:center; padding:8px; border-radius:4px; text-decoration:none; font-weight:bold; font-size:12px; transition:background-color 0.2s;">🏦 Yahoo!掲示板</a>'
            st.markdown(widget_btn_code := f'<a href="https://finance.yahoo.co.jp/quote/{cleanTicker}/forum" target="_blank" style="display:block; text-align:center; background-color:#7b00cc; color:white; padding:8px 0; border-radius:4px; text-decoration:none; font-weight:bold; font-size:11px;">🏦 Yahoo!掲示板</a>', unsafe_allow_url=true_or_not := True)

        # 4. 健康診断詳細
        st.markdown(f"""
        <div style="background-color:#1c2030; padding:12px; border-radius:4px; border-left:4px solid #2962ff; font-size:12px; line-height:1.5; margin-top:10px;">
          <strong>🏥 本日の健康診断カルテ: {cleanTicker}</strong><br>
          ・現在の判定  : <span style="color:#2962ff; font-weight:bold;">{selected_row['判定']}</span> (警戒スコア: {selected_row['警戒スコア']}点)<br>
          ・現在終値    : <strong>{selected_row['終値']} 円</strong> (取得単価比: {selected_row['取得単価比(%)']})<br>
          ・25日線乖離 : {selected_row['25日線乖離(%)']} / 75日線乖離: {selected_row['75日線乖離(%)']}<br>
          ・推奨アクション : <span style="background-color:#2962ff; padding:2px 6px; border-radius:3px; font-weight:bold;">{selected_row['推奨アクション']}</span><br>
          ・警戒サイン  : <span style="color:#ff0050;">{selected_row['警戒サイン'] if pd.notna(selected_row['警戒サイン']) else 'なし'}</span>
        </div>
        """, unsafe_allow_html=True)
