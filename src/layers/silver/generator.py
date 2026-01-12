# silver_data_generator.py (V7.0 - Advanced Math & Structural Logic)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the strategy discovery
pipeline. It transforms the raw, high-volume trade simulations from the Bronze
Layer Parquet files into an intelligent, context-rich dataset ready for
machine learning.

Updates in V7.0:
- Structural S/R: Implements ZigZag (Swing) Pivots based on ATR volatility.
- Dynamic Normalization: Uses ATR-based distances instead of static BPS.
- Vector Placement: Uses Linear Rescaling for SL/TP positioning.
"""

import gc
import logging
import math
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Manager, Pool
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

# Check for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("CRITICAL: 'pyarrow' library not found. Please run 'pip install pyarrow' to continue.")
    sys.exit(1)

try:
    import ta
except ImportError:
    print("CRITICAL: 'ta' library not found. Please run 'pip install ta'.")
    sys.exit(1)
try:
    import talib
except ImportError:
    print("CRITICAL: 'talib' library not found. See library docs for installation.")
    sys.exit(1)

# --- CONFIGURATION IMPORT ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

try:
    import config.config as c
    from src.utils import paths as p
    from src.utils.logger import setup_logging 
    from src.utils.file_selector import scan_new_files, select_files_interactively # type: ignore
    from src.utils.raw_data_loader import load_and_clean_raw_ohlc_csv # type: ignore
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
worker_feature_values_np: np.ndarray
worker_time_to_idx_lookup: pd.Series
worker_col_to_idx: Dict[str, int]
worker_levels_for_positioning: List[str]
worker_chunked_outcomes_dir: str
worker_atr_values_np: np.ndarray # New global for ATR normalization


def init_worker(
    feature_values_np: np.ndarray, time_to_idx_lookup: pd.Series, col_to_idx: Dict[str, int],
    levels_for_positioning: List[str], chunked_outcomes_dir: str, atr_values_np: np.ndarray
) -> None:
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_feature_values_np, worker_time_to_idx_lookup, worker_col_to_idx
    global worker_levels_for_positioning, worker_chunked_outcomes_dir, worker_atr_values_np
    
    worker_feature_values_np = feature_values_np
    worker_time_to_idx_lookup = time_to_idx_lookup
    worker_col_to_idx = col_to_idx
    worker_levels_for_positioning = levels_for_positioning
    worker_chunked_outcomes_dir = chunked_outcomes_dir
    worker_atr_values_np = atr_values_np


# --- ADVANCED MATH: STRUCTURAL ANALYSIS ---

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes DataFrame memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

@njit
def calculate_zigzag_levels_numba(highs: np.ndarray, lows: np.ndarray, atrs: np.ndarray, multiplier: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Structural Support and Resistance using a ZigZag (Swing) algorithm.
    Logic:
    - We track the current trend direction.
    - If price reverses by (ATR * multiplier), a Swing Point is confirmed.
    - We forward-fill the last confirmed Swing High as 'Resistance' and Swing Low as 'Support'.
    """
    n = len(highs)
    resistance = np.full(n, np.nan, dtype=np.float32)
    support = np.full(n, np.nan, dtype=np.float32)
    
    # ZigZag State
    trend = 1 # 1: Up, -1: Down
    last_high = highs[0]
    last_low = lows[0]
    last_high_idx = 0
    last_low_idx = 0
    
    # Initialize output with first bar
    curr_res = last_high
    curr_sup = last_low

    for i in range(1, n):
        threshold = atrs[i] * multiplier
        
        if trend == 1: # Uptrend
            if highs[i] > last_high:
                last_high = highs[i]
                last_high_idx = i
            elif lows[i] < last_high - threshold:
                # Reversal to Downtrend: The previous high is now a confirmed Resistance Pivot
                trend = -1
                curr_res = last_high 
                last_low = lows[i]
                last_low_idx = i
        
        else: # Downtrend
            if lows[i] < last_low:
                last_low = lows[i]
                last_low_idx = i
            elif highs[i] > last_low + threshold:
                # Reversal to Uptrend: The previous low is now a confirmed Support Pivot
                trend = 1
                curr_sup = last_low
                last_high = highs[i]
                last_high_idx = i
        
        resistance[i] = curr_res
        support[i] = curr_sup
        
    return support, resistance

def add_structural_support_resistance(df: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Calculates and adds ZigZag-based S/R levels to the DataFrame."""
    highs = df["high"].values.astype(np.float32)
    lows = df["low"].values.astype(np.float32)
    atrs = atr_series.values.astype(np.float32)
    
    # Use a dynamic threshold multiplier (e.g., 2.0 or 3.0 ATRs to confirm a swing)
    # Falling back to BBANDS_STD_DEV if not specifically defined for Swings
    swing_mult = getattr(c, 'SWING_ATR_MULTIPLIER', 2.0) 
    
    support_pts, resistance_pts = calculate_zigzag_levels_numba(highs, lows, atrs, swing_mult)
    
    df["support"] = support_pts
    df["resistance"] = resistance_pts
    
    # Handle initial NaNs (backward fill)
    df["support"] = df["support"].bfill()
    df["resistance"] = df["resistance"].bfill()
    
    return df

# --- STANDARD FEATURES ---

def map_market_sessions(hour_series: pd.Series) -> pd.Series:
    """
    Maps a Series of UTC hours to their corresponding Forex market sessions
    based on definitions in config.py.
    """
    # Uses dynamic bins and labels from config
    return pd.cut(
        hour_series, 
        bins=c.SESSION_BINS, 
        labels=c.SESSION_LABELS, 
        ordered=False, 
        right=True
    )

def _add_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a batch of standard technical indicators based on config lists."""
    indicator_df = pd.DataFrame(index=df.index)
    
    # Simple Moving Averages
    for p in c.SMA_PERIODS:
        indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
        
    # Exponential Moving Averages
    for p in c.EMA_PERIODS:
        indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
        
    # Bollinger Bands
    for p in c.BBANDS_PERIODS:
        bb = ta.volatility.BollingerBands(df["close"], p, c.BBANDS_STD_DEV)
        indicator_df[f"BB_upper_{p}"] = bb.bollinger_hband()
        indicator_df[f"BB_lower_{p}"] = bb.bollinger_lband()
        
    # RSI
    for p in c.RSI_PERIODS:
        indicator_df[f"RSI_{p}"] = ta.momentum.RSIIndicator(df["close"], p).rsi()
        
    # MACD (Uses global config, not lists)
    indicator_df[f"MACD_hist_{c.MACD_FAST}_{c.MACD_SLOW}_{c.MACD_SIGNAL}"] = ta.trend.MACD(df["close"], c.MACD_SLOW, c.MACD_FAST, c.MACD_SIGNAL).macd_diff()
        
    # ATR
    for p in c.ATR_PERIODS:
        indicator_df[f"ATR_{p}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], p).average_true_range()
        
    # ADX
    for p in c.ADX_PERIODS:
        indicator_df[f"ADX_{p}"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], p).adx()
        
    # Momentum (ROC) - Now Dynamic
    for p in c.MOM_PERIODS:
        indicator_df[f"MOM_{p}"] = ta.momentum.ROCIndicator(df["close"], window=p).roc()
        
    # CCI - Now Dynamic
    for p in c.CCI_PERIODS:
        indicator_df[f"CCI_{p}"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=p).cci()
        
    # Dynamic ATR Bands (e.g., Close + 1*ATR, Close + 1.5*ATR)
    for p in c.ATR_PERIODS:
        atr_series = indicator_df[f"ATR_{p}"]
        for mult in c.ATR_BAND_MULTIPLIERS:
            mult_str = str(mult).replace('.', 'p')
            indicator_df[f"ATR_level_up_{mult_str}x_{p}"] = df["close"] + (atr_series * mult)
            indicator_df[f"ATR_level_down_{mult_str}x_{p}"] = df["close"] - (atr_series * mult)
            
    return indicator_df

def _add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a batch of TA-Lib candlestick pattern recognitions."""
    pattern_names = talib.get_function_groups().get("Pattern Recognition", [])
    patterns_df = pd.DataFrame(index=df.index)
    for p in pattern_names:
        patterns_df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
    return patterns_df

def _add_time_and_pa_features(df: pd.DataFrame, indicator_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates time-based, price-action, and market regime features."""
    pa_df = pd.DataFrame(index=df.index)
    
    # 1. Market Sessions (Dynamic)
    pa_df['session'] = map_market_sessions(df['time'].dt.hour)
    pa_df['hour'], pa_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    
    # 2. Price Action Stats
    is_bullish = (df['close'] > df['open']).astype(int)
    body_size = np.abs(df['close'] - df['open'])
    
    for n in c.PAST_LOOKBACKS:
        pa_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        pa_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        pa_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        
    # 3. Regimes (Dynamic Thresholds)
    # Trend Regime based on ADX Threshold
    for p in c.ADX_PERIODS:
        if f'ADX_{p}' in indicator_df.columns:
            pa_df[f'trend_regime_{p}'] = np.where(
                indicator_df[f'ADX_{p}'] > c.ADX_TREND_THRESHOLD, 'trend', 'range'
            )
            
    # Volatility Regime based on ATR Moving Average Window
    for p in c.ATR_PERIODS:
        if f'ATR_{p}' in indicator_df.columns:
            # Current ATR vs Average ATR over ATR_MA_WINDOW
            avg_atr = indicator_df[f'ATR_{p}'].rolling(c.ATR_MA_WINDOW).mean()
            pa_df[f'vol_regime_{p}'] = np.where(
                indicator_df[f'ATR_{p}'] > avg_atr, 'high_vol', 'low_vol'
            )
            
    return pa_df

def add_all_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates feature calculation."""
    logger.info("  - Calculating indicators...")
    indicator_df = _add_standard_indicators(df)
    
    # REORDER: S/R now depends on ATR, so indicators must be done first
    logger.info("  - Calculating structural S/R (ZigZag)...")
    atr_col = f"ATR_{c.ATR_PERIODS[0]}"
    df_with_sr = add_structural_support_resistance(df.copy(), indicator_df[atr_col])
    
    logger.info("  - Calculating patterns & context...")
    patterns_df = _add_candlestick_patterns(df)

    logger.info("  - Calculating time-based and price-action features...")
    pa_df = _add_time_and_pa_features(df, indicator_df)
    
    return pd.concat([df, indicator_df, patterns_df, df_with_sr[['support', 'resistance']], pa_df], axis=1)

def create_feature_lookup_structures(
    features_df: pd.DataFrame, level_cols: List[str]
) -> Tuple[np.ndarray, pd.Series, Dict[str, int], np.ndarray]:
    """Pre-computes data into highly efficient lookup structures."""
    features_df = features_df.sort_values('time').reset_index(drop=True)
    col_to_idx = {col: i for i, col in enumerate(level_cols)}
    feature_values_np = features_df[level_cols].to_numpy(dtype=np.float32)
    time_to_idx_lookup = pd.Series(features_df.index, index=features_df['time'])
    
    # Extract ATR for Volatility Normalization
    atr_col = f"ATR_{c.ATR_PERIODS[0]}"
    if atr_col in features_df.columns:
        atr_values_np = features_df[atr_col].to_numpy(dtype=np.float32)
        # Prevent division by zero
        atr_values_np[atr_values_np == 0] = 1e-5 
    else:
        atr_values_np = np.ones(len(features_df), dtype=np.float32)
        
    return feature_values_np, time_to_idx_lookup, col_to_idx, atr_values_np

# --- ADVANCED ENRICHMENT LOGIC ---

def add_positioning_features(
    bronze_chunk: pd.DataFrame, feature_values_np: np.ndarray, time_to_idx_lookup: pd.Series,
    col_to_idx: Dict[str, int], levels_for_positioning: List[str]
) -> pd.DataFrame:
    """Enriches trades with Volatility-Normalized and Vector-Scaled features."""
    indices = time_to_idx_lookup.reindex(bronze_chunk['entry_time']).values
    valid_mask = ~np.isnan(indices)
    if not valid_mask.any(): return pd.DataFrame()
        
    bronze_chunk = bronze_chunk.loc[valid_mask].copy()
    indices = indices[valid_mask].astype(int)
    
    features_for_chunk = feature_values_np[indices]
    atr_for_chunk = worker_atr_values_np[indices]
    
    entry_prices = bronze_chunk['entry_price'].values
    sl_prices = bronze_chunk['sl_price'].values
    tp_prices = bronze_chunk['tp_price'].values

    # Helper for safe division
    def safe_divide(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(num, den)
        result[den == 0] = 0.0 # Standardize 'on-top' to 0 distance
        return result

    for level_name in levels_for_positioning:
        if level_name not in col_to_idx: continue
        level_prices = features_for_chunk[:, col_to_idx[level_name]]
        
        # --- 1. Volatility Normalized Distance (ATR Units) ---
        # "How many ATRs away is the SL/TP?"
        # Much better than BPS for ML across different years.
        bronze_chunk[f'sl_dist_to_{level_name}_atr'] = safe_divide(sl_prices - level_prices, atr_for_chunk)
        bronze_chunk[f'tp_dist_to_{level_name}_atr'] = safe_divide(tp_prices - level_prices, atr_for_chunk)
        
        # --- 2. Linear Vector Scaling ---
        # "Where is SL relative to the vector [Entry -> Level]?"
        # 0 = At Entry, 1 = At Level, 1.5 = Beyond Level, -0.5 = Opposite side
        denom = level_prices - entry_prices
        
        # If Level == Entry, we can't define a vector. Use 0.
        denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
        
        sl_scaled = (sl_prices - entry_prices) / denom
        tp_scaled = (tp_prices - entry_prices) / denom
        
        bronze_chunk[f'sl_place_scale_{level_name}'] = np.nan_to_num(sl_scaled, nan=0.0)
        bronze_chunk[f'tp_place_scale_{level_name}'] = np.nan_to_num(tp_scaled, nan=0.0)
        
    return bronze_chunk

def queue_worker(task_queue) -> int:
    """The 'Consumer' worker function."""
    total = 0
    while True:
        try:
            task = task_queue.get()
            if task is None: break
            chunk_df, chunk_num = task
            if chunk_df.empty: continue
            
            enriched = add_positioning_features(
                chunk_df, worker_feature_values_np, worker_time_to_idx_lookup, 
                worker_col_to_idx, worker_levels_for_positioning
            )
            
            if not enriched.empty:
                enriched = downcast_dtypes(enriched)
                path = os.path.join(worker_chunked_outcomes_dir, f"chunk_{chunk_num}.parquet")
                enriched.to_parquet(path, index=False)
                total += len(enriched)
        except Exception:
            logger.error("Worker Error", exc_info=True)
    return total

def _get_level_columns(all_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Dynamically identifies columns needed for positioning and NumPy lookup."""
    base = ['time', 'open', 'high', 'low', 'close', 'support', 'resistance']
    # Dynamic patterns matching the generated columns
    patterns = ['SMA_', 'EMA_', 'BB_upper', 'BB_lower', 'ATR_level']
    
    level_cols = set(base)
    for col in all_columns:
        if any(p in col for p in patterns): level_cols.add(col)
        
    cols_for_numpy = [c for c in level_cols if c != 'time']
    # Level columns are subsets of numpy columns that act as 'Levels'
    levels_for_pos = [c for c in cols_for_numpy if c not in ['open', 'high', 'low', 'close']]
    return cols_for_numpy, levels_for_pos

# --- MAIN ORCHESTRATOR ---
def create_silver_data(
    raw_path: str, features_path: str, bronze_path: str = None,
    chunked_outcomes_dir: str = None, features_only: bool = False
) -> None:
    """Orchestrates the Silver Layer generation process."""
    instrument_name = os.path.splitext(os.path.basename(raw_path))[0]
    logger.info(f"--- Processing: {instrument_name} ---")

    # --- Stage 1: Market Feature Generation ---
    logger.info("STEP 1: Creating Silver Features dataset...")
    try:
        raw_df = load_and_clean_raw_ohlc_csv(raw_path)
        # Remove the 'volume' column if it exists
        if 'volume' in raw_df.columns:
            raw_df.drop(columns=['volume'], inplace=True)
    except Exception as e:
        logger.error(f"Failed to load '{instrument_name}': {e}")
        return

    # Check for warmup period
    if len(raw_df) < c.SILVER_INDICATOR_WARMUP_PERIOD + 100:
        logger.error("Not enough data for warmup.")
        return
        
    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect()
    
    # TODO we should allow the non indicator filled rows to let go on for future uses
    # Remove warmup period rows
    features_df = features_df.iloc[c.SILVER_INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    logger.info(f"SUCCESS: Features saved to: {os.path.basename(features_path)}")
    
    if features_only:
        logger.info("Features-only mode complete.")
        return

    # --- Stage 2 & 3: Trade Enrichment ---
    logger.info("STEP 2: Preparing for PARALLEL chunk enrichment...")
    if os.path.exists(chunked_outcomes_dir): 
        shutil.rmtree(chunked_outcomes_dir)
    os.makedirs(chunked_outcomes_dir)
    
    logger.info("  - Creating feature lookup structures...")
    cols_for_numpy, levels_for_pos = _get_level_columns(features_df.columns)
    lookup_df = features_df[['time'] + cols_for_numpy]
    
    # NEW: Now returns ATR values too
    feature_values_np, time_to_idx, col_to_idx, atr_values_np = create_feature_lookup_structures(lookup_df, cols_for_numpy)
    del features_df, lookup_df; gc.collect()

    logger.info("STEP 3: Enriching Bronze data...")
    try:
        pq_file = pq.ParquetFile(bronze_path)
        num_rows = pq_file.metadata.num_rows
    except Exception as e:
        logger.error(f"Could not read Bronze Parquet: {e}")
        return
        
    if num_rows <= 0:
        logger.info("Bronze file is empty. No trades to process.")
        return

    logger.info(f"Found {num_rows:,} trades. Processing in batches...")
    manager = Manager()
    task_queue = manager.Queue(maxsize=c.MAX_CPU_USAGE * 2)
    
    # NEW: Init args now includes atr_values_np
    pool_init_args = (feature_values_np, time_to_idx, col_to_idx, levels_for_pos, chunked_outcomes_dir, atr_values_np)

    with Pool(processes=c.MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        worker_results = [pool.apply_async(queue_worker, (task_queue,)) for _ in range(c.MAX_CPU_USAGE)]
        iterator = pq_file.iter_batches(batch_size=c.SILVER_PARQUET_BATCH_SIZE)
        
        for i, batch in enumerate(tqdm(iterator, desc="Feeding Batches", unit="batch"), 1):
            task_queue.put((batch.to_pandas(), i))
            
        for _ in range(c.MAX_CPU_USAGE): task_queue.put(None)
        total_trades_processed = sum(res.get() for res in worker_results)

    logger.info(f"SUCCESS: Enriched {total_trades_processed:,} trades.")

def main() -> None:
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "silver_layer")
    p.ensure_directories()
    start_time = time.time()
    logger.info("--- Silver Layer: The Enrichment Engine (V7.0 - Advanced Math) ---")

    features_only_mode = '--features-only' in sys.argv
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
    
    if features_only_mode:
        logger.info("Running in FEATURES-ONLY mode.")
        if not target_file_arg:
            logger.error("--features-only requires filename.")
            return
        
        raw_path = os.path.join(p.RAW_DATA_DIR, f"{target_file_arg}.csv")
        features_path = os.path.join(p.SILVER_FEATURES_DIR, f"{target_file_arg}.parquet")
        
        if not os.path.exists(raw_path):
            logger.error(f"Raw data file not found: {raw_path}")
            return
        
        create_silver_data(raw_path, features_path, features_only=True)
            
    else:  # --- Normal (Full) Mode ---
        if target_file_arg:
            target_path = os.path.join(p.BRONZE_DATA_DIR, f"{target_file_arg}.parquet")
            if os.path.exists(target_path):
                files_to_process = [f"{target_file_arg}.parquet"]
                logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
            else:
                logger.error(f"Target file not found: {target_path}")
                return
        else:
            new_files = scan_new_files(p.BRONZE_DATA_DIR, p.SILVER_CHUNKED_DIR)
            # TODO we can use silver features too...
            files_to_process = select_files_interactively(new_files)

        if not files_to_process:
            logger.info("No files selected. Exiting.")
            return

        logger.info(f"Processing {len(files_to_process)} file(s)...")
        for filename in files_to_process:
            name = filename.replace('.parquet', '')
            raw_filename = filename.replace('.parquet', '.csv')
            paths = {
                "bronze_path": os.path.join(p.BRONZE_DATA_DIR, filename),
                "raw_path": os.path.join(p.RAW_DATA_DIR, raw_filename),
                "features_path": os.path.join(p.SILVER_FEATURES_DIR, filename),
                "chunked_outcomes_dir": os.path.join(p.SILVER_CHUNKED_DIR, name),
                "features_only": False
            }
            
            if os.path.exists(paths["raw_path"]):
                create_silver_data(**paths)
            else:
                logger.warning(f"SKIPPING {filename}: Corresponding raw_data file ('{raw_filename}') is missing.")
                continue

    end_time = time.time()
    logger.info(f"Silver Layer generation finished. Total Runtime: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()