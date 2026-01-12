# src/config.py

import logging
import numpy as np
from multiprocessing import cpu_count

# --- GLOBAL EXECUTION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
CONSOLE_LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = logging.DEBUG

# --- BRONZE LAYER ---
BRONZE_INPUT_CHUNK_SIZE: int = 5_000
BRONZE_OUTPUT_CHUNK_SIZE: int = 500_000
BRONZE_GENERATION_MODE: str = 'BALANCED' # 'WINS_ONLY', 'BALANCED', 'ALL'
BRONZE_MAX_SAMPLES_PER_CHUNK: int = 200_000
RAW_DATA_COLUMNS: list[str] = ["time", "open", "high", "low", "close", "volume"]

# The core simulation grid. Defines SL/TP ratios and the max trade holding
# period (`MAX_LOOKFORWARD`) for different chart timeframes.
TIMEFRAME_PRESETS: dict[str, dict] = {
    "1m": {"SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005), "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005), "MAX_LOOKFORWARD": 200},
    "5m": {"SL_RATIOS": np.arange(0.001, 0.0155, 0.0005), "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005), "MAX_LOOKFORWARD": 300},
    "15m": {"SL_RATIOS": np.arange(0.002, 0.0255, 0.001), "TP_RATIOS": np.arange(0.002, 0.0505, 0.001), "MAX_LOOKFORWARD": 400},
    "30m": {"SL_RATIOS": np.arange(0.003, 0.0355, 0.001), "TP_RATIOS": np.arange(0.003, 0.0705, 0.001), "MAX_LOOKFORWARD": 500},
    "60m": {"SL_RATIOS": np.arange(0.005, 0.0505, 0.001), "TP_RATIOS": np.arange(0.005, 0.1005, 0.001), "MAX_LOOKFORWARD": 600},
    "240m": {"SL_RATIOS": np.arange(0.010, 0.1005, 0.001), "TP_RATIOS": np.arange(0.010, 0.2005, 0.001), "MAX_LOOKFORWARD": 800},
}

# --- 3. SILVER LAYER CONFIGURATION ---
# Settings specific to `silver_data_generator.py`.

# The number of initial candles to discard to ensure indicator stability.
# This should be >= the longest indicator period used.
SILVER_INDICATOR_WARMUP_PERIOD: int = 200

# The number of rows to read from the Bronze Parquet file in each batch during enrichment.
SILVER_PARQUET_BATCH_SIZE: int = 500_000

# --- Technical Indicator Parameters (Feature Space) ---
# These lists and values define the feature engineering space.

# 1. Trend & Moving Averages
SMA_PERIODS: list[int] = [20, 50, 100, 200]
EMA_PERIODS: list[int] = [8, 13, 21, 50]

# 2. Volatility
BBANDS_PERIODS: list[int] = [20]
BBANDS_STD_DEV: float = 2.0
ATR_PERIODS: list[int] = [14]

# 3. Momentum
RSI_PERIODS: list[int] = [14]
ADX_PERIODS: list[int] = [14]
CCI_PERIODS: list[int] = [20]  # Changed from hardcoded 20 to a list
MOM_PERIODS: list[int] = [10]  # Changed from hardcoded 10 to a list

# 4. MACD
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# --- Market Structure & Regimes (Dynamic Logic) ---

# The lookback window on each side for identifying fractal S/R points.
PIVOT_WINDOW: int = 10

# The rolling window sizes for calculating price action features (e.g., avg_body_last_10).
PAST_LOOKBACKS: list[int] = [3, 5, 10, 20, 50]

# ADX Threshold: Values above this are considered "Trend", below is "Range"
ADX_TREND_THRESHOLD: int = 25

# Volatility Regime: The rolling window to calculate the average ATR.
# Current ATR > Average ATR(Window) = "High Volatility"
ATR_MA_WINDOW: int = 50

# ATR Bands: Multipliers to create dynamic S/R levels (e.g., Close + 1*ATR, Close + 2*ATR)
# This allows the ML to find relationships with dynamic volatility bands.
ATR_BAND_MULTIPLIERS: list[float] = [1.0, 2.0]

# Use a dynamic threshold multiplier (e.g., 2.0 or 3.0 ATRs to confirm a swing)
SWING_ATR_MULTIPLIER: 2.0

# Market Sessions Configuration (UTC)
# Bins define the hour cutoffs. Labels define the session names.
# Bins: [-1, 0, 8, 9, 13, 17, 22, 23] covers 0-23 hours.
SESSION_BINS: list[int] = [-1, 0, 8, 9, 13, 17, 22, 23]
SESSION_LABELS: list[str] = [
    'Tokyo',                 # Hour 0
    'Tokyo_London_Overlap',  # Hours 1-8 (Asian Session)
    'London',                # Hour 9 (London Open)
    'London_NY_Overlap',     # Hours 10-13
    'New_York',              # Hours 14-17
    'Sydney',                # Hours 18-22
    'Sydney'                 # Hour 23 (Merged into Sydney)
]

# --- 4. GOLD LAYER CONFIGURATION ---

# The size of the rolling window used for time-series standardization (scaling).
GOLD_SCALER_ROLLING_WINDOW: int = 200

# --- Multi-Anchor Normalization Strategy ---
# A list of transformations. The script will iterate through this list.
# For each entry, it finds columns matching 'targets_regex' and normalizes them
# against the 'anchor_col'.
# Formula: (Target - Anchor) / Anchor
# New Column Name: {Target}_rel_{Anchor}

GOLD_NORMALIZATION_CONFIG: list[dict] = [
    # 1. The Standard: Everything vs Close
    # Helps the AI see where levels are relative to current price.
    {
        "anchor_col": "close",
        "targets_regex": r"^(open|high|low|SMA_.*|EMA_.*|BB_.*|support|resistance|ATR_level.*)$"
    },

    # 2. Wick Dynamics: High/Low vs Open
    # Helps the AI understand intra-candle volatility and rejection.
    {
        "anchor_col": "open",
        "targets_regex": r"^(high|low)$"
    },
    
    # 3. Trend Deviation: Close vs Long-Term MA
    # Helps the AI spot overextended trends.
    {
        "anchor_col": "SMA_200", 
        "targets_regex": r"^(close)$"
    },
    
    # 4. Volatility Compression: Bands vs SMA
    # Helps the AI see if bands are squeezing (narrow) or expanding.
    {
        "anchor_col": "SMA_20",
        "targets_regex": r"^(BB_upper_20|BB_lower_20)$"
    }
]

# --- 5. PLATINUM LAYER CONFIGURATION ---
# Settings for `platinum_preprocessor.py` and `platinum_strategy_discoverer.py`

# -- Pre-Processor Settings --
# The size of the bin in Basis Points for bucketing relational distance features.
PLATINUM_BPS_BIN_SIZE: float = 5.0

# The number of aggregated trade records to hold in memory before flushing to temp files.
# Larger values are faster but use more RAM.
PLATINUM_BUFFER_FLUSH_THRESHOLD: int = 2_000_000

# The number of temporary shard files to use during the shuffle phase.
# More shards can improve parallelism but may increase filesystem overhead.
# A power of 2 is often a good choice.
PLATINUM_NUM_SHARDS: int = 128

# The prefix for temporary files created during the shuffle phase.
PLATINUM_TEMP_SHARD_PREFIX: str = "_temp_shard_"

# -- Strategy Discoverer Settings --
# The minimum number of candles a blueprint must have appeared on to be considered for discovery.
PLATINUM_MIN_CANDLE_LIMIT: int = 100

# The maximum depth of the Decision Tree. Deeper trees can find more complex rules
# but risk overfitting. 6-8 is often a good starting range.
PLATINUM_DT_MAX_DEPTH: int = 7

# The minimum number of historical candles a final rule must apply to.
# This is a key parameter to prevent rules based on statistically insignificant samples.
PLATINUM_MIN_CANDLES_PER_RULE: int = 50

# A rule's average profitability must be this many times greater than the blueprint's
# overall average profitability. Controls how much of an "edge" a rule must have.
PLATINUM_DENSITY_LIFT_THRESHOLD: float = 1.5

# The number of blueprints to process in a single batch by each worker process.
PLATINUM_DISCOVERY_BATCH_SIZE: int = 20
# --- DIAMOND LAYER (Model Trainer) ---
# Set to True if you have 64GB+ RAM or a high-end Server.
# It loads all data at once for faster training.
# Set to False to use Iterative Loading (slower, but works on 8GB RAM).
DIAMOND_LOAD_FULL_DATASET_IN_MEMORY: bool = False

# Parameters for the Unified XGBoost Model
DIAMOND_XGB_PARAMS: dict = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,             # Learning Rate
    'max_depth': 8,          # Depth
    'subsample': 0.8,        # Stochastic row sampling
    'colsample_bytree': 0.8, # Feature sampling
    'tree_method': 'hist',   # Use 'hist' for CPU. Change to 'gpu_hist' or device='cuda' for GPU.
    'nthread': MAX_CPU_USAGE
    # 'device': 'cuda'       # Uncomment this line if using a GPU Server
}

DIAMOND_BOOST_ROUNDS: int = 1000
DIAMOND_EARLY_STOPPING: int = 50
DIAMOND_TEST_SIZE: float = 0.2  # Last 20% of data used for validation (Time Series Split)

# --- SIMULATION CONSTANTS ---
SIMULATION_SPREAD_PIPS: dict[str, float] = {
    "DEFAULT": 3.0, "EURUSD": 1.5, "GBPUSD": 2.0, "AUDUSD": 2.5,
    "USDJPY": 2.0, "USDCAD": 2.5, "XAUUSD": 20.0,
}

# --- PIP SIZE MAP (NEW) ---
# Explicitly defines the pip size for instruments to avoid guessing.
# 0.01 for JPY pairs and Metals/Indices, 0.0001 for standard Forex.
PIP_SIZE_MAP: dict[str, float] = {
    "DEFAULT": 0.0001,
    "JPY": 0.01,
    "XAU": 0.01,
    "XAG": 0.01,
    "BTC": 1.0, # Just in case
    "ETH": 1.0,
    "US30": 1.0,
    "SPX": 0.1
}

# Assumed round-trip commission cost per standard lot (100,000 units).
SIMULATION_COMMISSION_PER_LOT: float = 7.0

# The maximum number of candles a trade will be held in the Diamond layer sims.
SIMULATION_MAX_LOOKFORWARD: int = 500

# --- Performance Filters for "Master Strategy" Status ---
# Used by the diamond_backtester to filter for quality.
DIAMOND_MIN_PROFIT_FACTOR: float = 1.5
DIAMOND_MAX_DRAWDOWN_PCT: float = 20.0
DIAMOND_MIN_TOTAL_TRADES: int = 50
DIAMOND_MIN_SQN: float = 1.8