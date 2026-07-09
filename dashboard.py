# dashboard.py (Version 1.0 - Streamlit Central Command)
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ページ基本設定：黒テーマを引き立たせるワイドレイアウト
st.set_page_config(page_title="Sniper OS - Central Command", layout="wide", initial_sidebar_state="collapsed")

# CSSでダークモードをさらにプロ仕様に整形
st.markdown("""
    <style>
    .reportview-container { background: #131722; }
    h1, h2, h3 { color: #2962ff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    div.stButton > button:first-child { background-color: #2962ff; color: white; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True) # 👈 ★【修正点】：このように書き換えます

st.title("🚀 Sniper OS - Central Command Dashboard")
st.caption("毎日21:00に自動更新される、あなたの保有・監視株の一元管理コックピット")

# 1. GitHubに自動保存されている最新の健康診断データをロード（API認証不要のスマート接続）
# ※あなたのGitHubのRaw URLを指定することで、常に最新状態をロードします。
RAW_CSV_URL = "https://raw.githubusercontent.com/naoyann3/long-term-screener/master/long_term_tracking.csv"

@st.cache_data(ttl=60) # 1分間キャッシュ
def load_data():
    try:
        # もしGitHubから直接ロードできない場合は、ローカルのCSVを読み込むロバスト仕様
        return pd.read_csv(RAW_CSV_URL)
    except Exception:
        return pd.read_csv("long_term_tracking.csv")

df = load_data()

if df.empty:
    st.error("現在、トラッキングデータがありません。")
else:
    # 表示用のクレンジング
    display_cols = ["判定", "警戒スコア", "ティッカー", "銘柄名", "種別", "取得日", "取得単価", "終値", "取得単価比(%)", "25日線乖離(%)", "75日線乖離(%)", "推奨アクション", "警戒サイン", "メモ"]
    df_display = df[[col for col in display_cols if col in df.columns]].copy()

    # 左右の2ペイン（左に表、右に TradingView と 株探・Yahoo）に画面を美しく分割
    col_left, col_right = st.columns([4, 3])

    with col_left:
        st.subheader("📊 ポートフォリオ健康診断台帳")
        st.write("健康診断結果の一覧です。行をクリックして選択すると、右側のチャートが一瞬で切り替わります。")
        
        # Streamlit標準のインタラクティブなデータテーブル（セル選択を検知可能な最新エディション）
        event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",  # 👈 【最重要】：行を選択した瞬間に画面を再描画させて切り替えるトリガー [2]
            selection_mode="single-row"
        )
        
        # 選択された行の銘柄コードを特定
        selected_ticker = "4063.T"  # デフォルト（信越化学）
        selected_name = "信越化学工業"
        
        if event and event.get("selection") and event["selection"].get("rows"):
            selected_row_idx = event["selection"]["rows"][0]
            selected_ticker = df_display.iloc[selected_row_idx]["ティッカー"]
            selected_name = df_display.iloc[selected_row_idx]["銘柄名"]

    with col_right:
        st.subheader(f"📈 {selected_name} ({selected_ticker}) の定点観測")
        clean_code = selected_ticker.split(".")[0]
        
        # 2. 【本物のTradingView大画面ウィジェット】
        # サンドボックスの外から呼び出すため、東証（TSE:）データであってもエラーを1発完殺してリアルタイム描画！
        tv_widget_html = f"""
        <div class="tradingview-widget-container" style="height:400px; width:100%;">
          <div id="tradingview_chart" style="height:100%; width:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "autosize": true,
            "symbol": "TSE:{clean_code}",
            "interval": "D",
            "timezone": "Asia/Tokyo",
            "theme": "dark",
            "style": "1",
            "locale": "ja",
            "toolbar_bg": "#131722",
            "enable_publishing": false,
            "hide_side_toolbar": true,
            "allow_symbol_change": false,
            "save_image": false,
            "container_id": "tradingview_chart"
          }});
          </script>
        </div>
        """
        components.html(tv_widget_html, height=400)
        
        # 3. あなたが開発した、株探決算 ＆ Yahoo!掲示板へのスマートダイレクトゲート
        st.markdown(f"#### 🔗 クオンツ・ダイレクトポータル")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.link_button("📊 株探決算で財務をデバッグ", f"https://kabutan.jp/stock/finance?code={clean_code}", use_container_width=True)
        with btn_col2:
            st.link_button("🏦 Yahoo!掲示板で大衆心理を追跡", f"https://finance.yahoo.co.jp/quote/{clean_code}/forum", use_container_width=True)
