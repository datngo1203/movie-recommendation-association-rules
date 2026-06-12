from pathlib import Path


APP_TITLE = "Hệ thống gợi ý phim bằng luật kết hợp"
APP_ICON = "🎬"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

MOVIES_FILE = RAW_DATA_DIR / "movies.csv"

RULE_OUTPUTS = {
    "Apriori": MODELS_DIR / "apriori_rules.csv",
    "FP-Growth": MODELS_DIR / "fpgrowth_rules.csv",
}
