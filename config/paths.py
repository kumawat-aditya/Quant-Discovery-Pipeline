# config/paths.py
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Core dirs (Use / instead of os.path.join)
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR   = PROJECT_ROOT / "data"
SRC_DIR    = PROJECT_ROOT / "src"
LOGS_DIR   = PROJECT_ROOT / "logs"

# Data layers
RAW_DATA_DIR    = DATA_DIR / "raw"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
SILVER_DATA_FEATURES_DIR = SILVER_DATA_DIR / "features"
SILVER_DATA_CHUNKED_OUTCOMES_DIR = SILVER_DATA_DIR / "chunked_outcomes"
GOLD_DATA_DIR   = DATA_DIR / "gold"
GOLD_DATA_FEATURES_DIR = GOLD_DATA_DIR / "features"
PLATINUM_DATA_DIR = DATA_DIR / "platinum"
PLATINUM_DATA_COMBINATIONS_DIR = PLATINUM_DATA_DIR / "combinations"
PLATINUM_DATA_TEMP_DIR = PLATINUM_DATA_DIR / "temp_targets"
PLATINUM_DATA_TARGETS_DIR = PLATINUM_DATA_DIR / "targets"

# Source dirs
SRC_UTILS_DIR    = SRC_DIR / "utils"
SRC_PIPELINE_DIR = SRC_DIR / "pipeline"
SRC_LAYERS_DIR   = SRC_DIR / "layers"

# Script layers
LAYERS_BRONZE_DIR   = SRC_LAYERS_DIR / "bronze"
LAYERS_SILVER_DIR   = SRC_LAYERS_DIR / "silver"
LAYERS_GOLD_DIR     = SRC_LAYERS_DIR / "gold"
LAYERS_PLATINUM_DIR = SRC_LAYERS_DIR / "platinum"
LAYERS_DIAMOND_DIR  = SRC_LAYERS_DIR / "diamond"

# Ensure dirs exist
ALL_DIRS = [
    DATA_DIR,
    LOGS_DIR,
    RAW_DATA_DIR,
    BRONZE_DATA_DIR,
    SILVER_DATA_DIR,
    SILVER_DATA_FEATURES_DIR,
    SILVER_DATA_CHUNKED_OUTCOMES_DIR,
    GOLD_DATA_DIR,
    GOLD_DATA_FEATURES_DIR,
    PLATINUM_DATA_DIR,
    PLATINUM_DATA_COMBINATIONS_DIR,
    PLATINUM_DATA_TEMP_DIR,
    PLATINUM_DATA_TARGETS_DIR
]

def ensure_directories():
    for d in ALL_DIRS:
        # Now 'd' is a Path object, so mkdir works!
        d.mkdir(parents=True, exist_ok=True)

# if __name__ == "__main__":
#     ensure_directories()