# src/utils/paths.py

import os
from pathlib import Path

# --- PROJECT ROOT ---
# Assuming this file is in src/utils/paths.py
# .parents[2] goes up from 'paths.py' -> 'utils' -> 'src' -> 'trading-strategy-finder'
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- TOP LEVEL DIRECTORIES ---
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR   = PROJECT_ROOT / "data"
SRC_DIR    = PROJECT_ROOT / "src"
LOGS_DIR   = PROJECT_ROOT / "logs"

# --- DATA LAYERS (The Pipeline) ---

# 1. Raw
RAW_DATA_DIR = DATA_DIR / "raw"

# 2. Bronze
BRONZE_DATA_DIR = DATA_DIR / "bronze"

# 3. Silver
SILVER_DATA_DIR          = DATA_DIR / "silver"
SILVER_FEATURES_DIR      = SILVER_DATA_DIR / "features"
SILVER_CHUNKED_DIR       = SILVER_DATA_DIR / "chunked_outcomes"

# 4. Gold
GOLD_DATA_DIR            = DATA_DIR / "gold"
GOLD_FEATURES_DIR        = GOLD_DATA_DIR / "features"

# 5. Platinum (Strategy Discovery)
PLATINUM_DATA_DIR        = DATA_DIR / "platinum"
PLATINUM_COMBINATIONS    = PLATINUM_DATA_DIR / "combinations"
PLATINUM_TARGETS         = PLATINUM_DATA_DIR / "targets"
PLATINUM_TEMP_TARGETS    = PLATINUM_DATA_DIR / "temp_targets" # Internal use
PLATINUM_DISCOVERED      = PLATINUM_DATA_DIR / "discovered_strategies"
PLATINUM_BLACKLISTS      = PLATINUM_DATA_DIR / "blacklists"
PLATINUM_LOGS            = PLATINUM_DATA_DIR / "discovery_log"

# 6. Diamond (Validation & Production)
DIAMOND_DATA_DIR         = DATA_DIR / "diamond"
DIAMOND_STRATEGIES       = DIAMOND_DATA_DIR / "strategies"   # Saved XGBoost Models (.json)
DIAMOND_TRIGGERS         = DIAMOND_DATA_DIR / "triggers"     # Threshold configs
DIAMOND_TRADE_LOGS       = DIAMOND_DATA_DIR / "trade_logs"   # Backtest results
DIAMOND_REPORTS          = DIAMOND_DATA_DIR / "reports"
DIAMOND_VALIDATION       = DIAMOND_DATA_DIR / "validation"

# --- LIST OF CRITICAL DIRS FOR SETUP ---
ALL_DIRS = [
    LOGS_DIR,
    RAW_DATA_DIR,
    BRONZE_DATA_DIR,
    SILVER_FEATURES_DIR,
    SILVER_CHUNKED_DIR,
    GOLD_FEATURES_DIR,
    PLATINUM_COMBINATIONS,
    PLATINUM_TARGETS,
    PLATINUM_TEMP_TARGETS,
    PLATINUM_DISCOVERED,
    PLATINUM_BLACKLISTS,
    PLATINUM_LOGS,
    DIAMOND_STRATEGIES,
    DIAMOND_TRIGGERS,
    DIAMOND_TRADE_LOGS,
    DIAMOND_REPORTS,
    DIAMOND_VALIDATION
]

def ensure_directories():
    """Creates the directory structure if it doesn't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)