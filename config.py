# config.py (Version 1.1)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
LONG_TERM_WATCHLISTS_DIR = RESULTS_DIR / "long_term_watchlists"

# ★【新規追加】：中期合格者・累積追跡台帳
CANDIDATE_HISTORY_CSV = RESULTS_DIR / "candidate_history.csv"

LONG_TERM_SCREEN_VERSION = "lt_v1"


def ensure_results_dirs():
    for path in (RESULTS_DIR, LONG_TERM_WATCHLISTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

# ==========================================
# 📅 【中期成長株：決算イベントリスク ＆ 決算ショック検出設定】
# ==========================================
# 決算日不明時の扱い
EARNINGS_UNKNOWN_PENALTY = 0.0     # 減点はないが、警告フラグ「⚪ 決算日不明」を明示
EARNINGS_UNKNOWN_NOTE = "EARNINGS_UNKNOWN"

# 決算接近時の段階的減点ペナルティ（総合スコアから差し引く値）
PENALTY_DIRECT_PRE_EARNINGS = -4.0   # 1〜3営業日：🔴 新規買い非推奨（大幅減点）
PENALTY_CLOSE_PRE_EARNINGS = -2.0    # 4〜7営業日：🟠 決算接近（中度減点）
PENALTY_WARN_PRE_EARNINGS = -0.5     # 8〜14営業日：🟡 決算注意（軽度減点）

# 【決算ショック急落候補】のテクニカル検知閾値
EARNINGS_SHOCK_DROP_PCT = -8.0       # 決算直後（1日）の大幅下落率閾値（%）
EARNINGS_SHOCK_VOLUME_RATIO = 1.5    # 出来高急増倍率（20日平均の1.5倍以上）
EARNINGS_SHOCK_LOOKBACK_DAYS = 5     # 直近決算日からの最大経過日数（または急落検知期間）
