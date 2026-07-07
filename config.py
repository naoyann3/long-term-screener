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
