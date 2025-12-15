# config/paths.py
from pathlib import Path
import os

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Core dirs
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
SRC_DIR    = os.path.join(PROJECT_ROOT, "src")
LOGS_DIR   = os.path.join(PROJECT_ROOT, "logs")

# Data layers
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
BRONZE_DATA_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DATA_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DATA_DIR   = os.path.join(DATA_DIR, "gold")
PLATINUM_DATA_DIR = os.path.join(DATA_DIR, "platinum")

# Source dirs
SRC_UTILS_DIR = os.path.join(SRC_DIR, "utils")
SRC_PIPELINE_DIR = os.path.join(SRC_DIR, "pipeline")
SRC_LAYERS_DIR = os.path.join(SRC_DIR, "layers")

# script layers
LAYERS_BRONZE_DIR = os.path.join(SRC_LAYERS_DIR, "bronze")
LAYERS_SILVER_DIR = os.path.join(SRC_LAYERS_DIR, "silver")
LAYERS_GOLD_DIR = os.path.join(SRC_LAYERS_DIR, "gold")
LAYERS_PLATINUM_DIR = os.path.join(SRC_LAYERS_DIR, "platinum")
LAYERS_DIAMOND_DIR = os.path.join(SRC_LAYERS_DIR, "diamond")

# Ensure dirs exist (safe to call multiple times)
ALL_DIRS = [
    DATA_DIR,
    RAW_DATA_DIR,
    BRONZE_DATA_DIR,
    SILVER_DATA_DIR,
    GOLD_DATA_DIR,
    PLATINUM_DATA_DIR,
    LOGS_DIR,
]

def ensure_directories():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
