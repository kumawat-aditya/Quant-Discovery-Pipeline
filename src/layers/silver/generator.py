# silver_data_generator.py (V6.0 - Fully Dynamic Config)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the strategy discovery
pipeline. It transforms the raw, high-volume trade simulations from the Bronze
Layer Parquet files into an intelligent, context-rich dataset ready for
machine learning.

It is now fully driven by `config.py`, allowing for dynamic adjustment of
indicator periods, regime thresholds, and market session definitions.
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
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
config_dir = os.path.join(project_root, "config")
utils_dir = os.path.join(project_root, "src", "utils")

sys.path.append(config_dir)
sys.path.append(utils_dir)

try:
    import config as c
    import paths as p # type: ignore
    from logger_setup import setup_logging # type: ignore
    from file_selector import scan_new_files, select_files_interactively # type: ignore
    from raw_data_loader import load_and_clean_raw_ohlc_csv # type: ignore
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules. Ensure config.py and utils are accessible: {e}")
    sys.exit(1)

# Initialize logger for this module
logger = logging.getLogger(__name__)


# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
worker_feature_values_np: np.ndarray
worker_time_to_idx_lookup: pd.Series
worker_col_to_idx: Dict[str, int]
worker_levels_for_positioning: List[str]
worker_chunked_outcomes_dir: str


def init_worker(
    feature_values_np: np.ndarray, time_to_idx_lookup: pd.Series, col_to_idx: Dict[str, int],
    levels_for_positioning: List[str], chunked_outcomes_dir: str
) -> None:
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_feature_values_np, worker_time_to_idx_lookup, worker_col_to_idx
    global worker_levels_for_positioning, worker_chunked_outcomes_dir
    worker_feature_values_np, worker_time_to_idx_lookup, worker_col_to_idx, \
    worker_levels_for_positioning, worker_chunked_outcomes_dir = \
        feature_values_np, time_to_idx_lookup, col_to_idx, \
        levels_for_positioning, chunked_outcomes_dir


# --- UTILITY & FEATURE FUNCTIONS ---

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes DataFrame memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

@numba.njit
def _calculate_s_r_numba(lows: np.ndarray, highs: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies fractal support and resistance points using Numba."""
    n = len(lows)
    support, resistance = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        ws = slice(i - window, i + window + 1)
        if lows[i] == np.min(lows[ws]): support[i] = lows[i]
        if highs[i] == np.max( highs[ws]): resistance[i] = highs[i]
    return support, resistance

def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds forward-filled S/R levels to the DataFrame."""
    lows, highs = df["low"].values.astype(np.float32), df["high"].values.astype(np.float32)
    # Uses config for pivot window
    support_pts, resistance_pts = _calculate_s_r_numba(lows, highs, c.PIVOT_WINDOW)
    df["support"] = pd.Series(support_pts, index=df.index).ffill()
    df["resistance"] = pd.Series(resistance_pts, index=df.index).ffill()
    return df

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
            # Replaced "1x" hardcoding with dynamic multiplier naming
            mult_str = str(mult).replace('.', 'p') # 1.5 -> 1p5
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
    """Orchestrates the calculation of a comprehensive suite of market features."""
    logger.info("  - Calculating standard indicators...")
    indicator_df = _add_standard_indicators(df)
    
    logger.info("  - Calculating candlestick patterns...")
    patterns_df = _add_candlestick_patterns(df)
    
    logger.info("  - Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy())
    
    logger.info("  - Calculating time-based and price-action features...")
    pa_df = _add_time_and_pa_features(df, indicator_df)
    
    return pd.concat([df, indicator_df, patterns_df, df_with_sr[['support', 'resistance']], pa_df], axis=1)

def create_feature_lookup_structures(
    features_df: pd.DataFrame, level_cols: List[str]
) -> Tuple[np.ndarray, pd.Series, Dict[str, int]]:
    """Pre-computes data into highly efficient lookup structures."""
    features_df = features_df.sort_values('time').reset_index(drop=True)
    col_to_idx = {col: i for i, col in enumerate(level_cols)}
    feature_values_np = features_df[level_cols].to_numpy(dtype=np.float32)
    time_to_idx_lookup = pd.Series(features_df.index, index=features_df['time'])
    return feature_values_np, time_to_idx_lookup, col_to_idx

def add_positioning_features(
    bronze_chunk: pd.DataFrame, feature_values_np: np.ndarray, time_to_idx_lookup: pd.Series,
    col_to_idx: Dict[str, int], levels_for_positioning: List[str]
) -> pd.DataFrame:
    """Enriches a chunk of Bronze trade data with relational positioning features."""
    # Find indices in the feature array corresponding to trade entry times
    indices = time_to_idx_lookup.reindex(bronze_chunk['entry_time']).values
    valid_mask = ~np.isnan(indices)
    if not valid_mask.any(): return pd.DataFrame()
        
    bronze_chunk = bronze_chunk.loc[valid_mask].copy()
    indices = indices[valid_mask].astype(int)
    
    # Extract feature values for these specific times
    features_for_chunk_np = feature_values_np[indices]
    
    sl_prices = bronze_chunk['sl_price'].values
    tp_prices = bronze_chunk['tp_price'].values
    candle_close_price = features_for_chunk_np[:, col_to_idx['close']]

    def safe_divide(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(num, den)
        result[den == 0] = np.nan
        return result

    # Calculate distance features for every level defined
    for level_name in levels_for_positioning:
        if level_name not in col_to_idx: continue
            
        level_price = features_for_chunk_np[:, col_to_idx[level_name]]
        
        # 1. Distance in Basis Points
        bronze_chunk[f'sl_dist_to_{level_name}_bps'] = safe_divide(sl_prices - level_price, candle_close_price) * 10000
        bronze_chunk[f'tp_dist_to_{level_name}_bps'] = safe_divide(tp_prices - level_price, candle_close_price) * 10000
        
        # 2. Relational Placement (Pct)
        # How far is the SL from entry, relative to how far the level is from entry?
        total_dist_to_level = level_price - candle_close_price
        sl_dist_from_entry = sl_prices - candle_close_price
        tp_dist_from_entry = tp_prices - candle_close_price
        
        bronze_chunk[f'sl_place_pct_to_{level_name}'] = safe_divide(sl_dist_from_entry, total_dist_to_level)
        bronze_chunk[f'tp_place_pct_to_{level_name}'] = safe_divide(tp_dist_from_entry, total_dist_to_level)
        
    return bronze_chunk

def queue_worker(task_queue) -> int:
    """The 'Consumer' worker function, which runs in a continuous loop."""
    total = 0
    while True:
        try:
            task = task_queue.get()
            if task is None: break
            chunk_df, chunk_num = task
            if chunk_df.empty: continue
            enriched = add_positioning_features(chunk_df, worker_feature_values_np, worker_time_to_idx_lookup, worker_col_to_idx, worker_levels_for_positioning)
            if not enriched.empty:
                enriched = downcast_dtypes(enriched)
                path = os.path.join(worker_chunked_outcomes_dir, f"chunk_{chunk_num}.parquet")
                enriched.to_parquet(path, index=False)
                total += len(enriched)
        except Exception:
            logger.error(f"An error occurred in a worker process.", exc_info=True)
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
    except Exception as e:
        logger.error(f"ERROR: Failed to load '{instrument_name}': {e}")
        return

    # Check for warmup period
    if len(raw_df) < c.SILVER_INDICATOR_WARMUP_PERIOD + 100:
        logger.error(f"Not enough data for indicator warmup ({len(raw_df)} rows). Skipping.")
        return
        
    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect()
    
    # TODO we should allow the non indicator filled rows to let go on for future uses
    # Remove warmup period rows
    features_df = features_df.iloc[c.SILVER_INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    logger.info(f"SUCCESS: Silver Features saved to: {os.path.basename(features_path)}")
    
    if features_only:
        logger.info("Features-only mode complete.")
        return

    # --- Stage 2 & 3: Trade Enrichment ---
    logger.info("STEP 2: Preparing for PARALLEL chunk enrichment...")
    if os.path.exists(chunked_outcomes_dir): shutil.rmtree(chunked_outcomes_dir)
    os.makedirs(chunked_outcomes_dir)
    
    logger.info("  - Creating feature lookup structures...")
    cols_for_numpy, levels_for_pos = _get_level_columns(features_df.columns)
    lookup_df = features_df[['time'] + cols_for_numpy]
    feature_values_np, time_to_idx, col_to_idx = create_feature_lookup_structures(lookup_df, cols_for_numpy)
    del features_df, lookup_df; gc.collect()

    logger.info("STEP 3: Enriching Bronze data...")
    try:
        pq_file = pq.ParquetFile(bronze_path)
        num_rows = pq_file.metadata.num_rows
    except Exception as e:
        logger.error(f"Could not read Bronze Parquet file at {bronze_path}: {e}")
        return
        
    if num_rows <= 0:
        logger.info("Bronze file is empty. No trades to process.")
        return

    logger.info(f"Found {num_rows:,} trades. Processing in batches...")
    manager = Manager()
    task_queue = manager.Queue(maxsize=c.MAX_CPU_USAGE * 2)
    pool_init_args = (feature_values_np, time_to_idx, col_to_idx, levels_for_pos, chunked_outcomes_dir)

    with Pool(processes=c.MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        worker_results = [pool.apply_async(queue_worker, (task_queue,)) for _ in range(c.MAX_CPU_USAGE)]
        
        # Read Bronze Parquet file in batches to save RAM
        iterator = pq_file.iter_batches(batch_size=c.SILVER_PARQUET_BATCH_SIZE)
        
        for i, batch in enumerate(tqdm(iterator, desc="Feeding Batches", unit="batch"), 1):
            task_queue.put((batch.to_pandas(), i))
            
        for _ in range(c.MAX_CPU_USAGE): task_queue.put(None)
        total_trades_processed = sum(res.get() for res in worker_results)

    logger.info(f"SUCCESS: Enriched and chunked {total_trades_processed:,} trades.")
    logger.info(f"Output saved to: {chunked_outcomes_dir}")

def main() -> None:
    """Main execution function."""
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "silver_layer")
    
    start_time = time.time()
    logger.info("--- Silver Layer: The Enrichment Engine (V6.0 - Dynamic Config) ---")

    features_only_mode = '--features-only' in sys.argv
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
    
    if features_only_mode:
        logger.info("Running in FEATURES-ONLY mode.")
        if not target_file_arg:
            logger.error("--features-only mode requires a target .csv file argument.")
            return
        
        raw_path = os.path.join(p.RAW_DATA_DIR, f"{target_file_arg}.csv")
        features_path = os.path.join(p.SILVER_DATA_FEATURES_DIR, f"{target_file_arg}.parquet")
        
        if not os.path.exists(raw_path):
            logger.error(f"Raw data file not found: {raw_path}")
            return
            
        paths = {
            "raw_path": raw_path,
            "features_path": features_path,
            "features_only": True
        }
        create_silver_data(**paths)
        
    else: # --- Normal (Full) Mode ---
        if target_file_arg:
            target_path = os.path.join(p.BRONZE_DATA_DIR, f"{target_file_arg}.parquet")
            if os.path.exists(target_path):
                files_to_process = [f"{target_file_arg}.parquet"]
                logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
            else:
                logger.error(f"Target file not found: {target_path}")
                return
        else:
            new_files = scan_new_files(p.BRONZE_DATA_DIR, p.SILVER_DATA_CHUNKED_OUTCOMES_DIR)
            # TODO we can use silver features too...
            files_to_process = select_files_interactively(new_files)

        if not files_to_process:
            logger.info("No files selected. Exiting.")
            return
        else:
            logger.info(f"Processing {len(files_to_process)} file(s)...")
            for filename in files_to_process:
                name = filename.replace('.parquet', '')
                raw_filename = filename.replace('.parquet', '.csv')
                paths = {
                    "bronze_path": os.path.join(p.BRONZE_DATA_DIR, filename),
                    "raw_path": os.path.join(p.RAW_DATA_DIR, raw_filename),
                    "features_path": os.path.join(p.SILVER_DATA_FEATURES_DIR, filename),
                    "chunked_outcomes_dir": os.path.join(p.SILVER_DATA_CHUNKED_OUTCOMES_DIR, name),
                    "features_only": False
                }
                
                if not os.path.exists(paths["raw_path"]):
                    logger.warning(f"SKIPPING {filename}: Corresponding raw_data file ('{raw_filename}') is missing.")
                    continue
                    
                try:
                    create_silver_data(**paths)
                except Exception:
                    logger.critical(f"A fatal error occurred while processing {filename}.", exc_info=True)

    end_time = time.time()
    logger.info(f"Silver Layer generation finished. Total Runtime: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()