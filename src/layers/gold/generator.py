# gold_data_generator.py (V4.0 - Corrected Scaling, Config & Logging)

"""
Gold Layer: The Machine Learning Preprocessor

This script represents the crucial final stage of data preparation in the
pipeline. It acts as a specialized transformer, converting the human-readable,
context-rich Silver `features` dataset into a purely numerical, normalized, and
standardized Parquet file that is perfectly optimized for machine learning.

Its sole purpose is to "translate" market context into the mathematical
language that ML models understand, performing several key transformations:
- Relational Transformation: Converts absolute price levels into a normalized
  distance from the current close price, making features scale-invariant.
- Categorical Encoding: Converts text-based features (e.g., 'session', 'regime')
  into binary (0/1) columns via one-hot encoding.
- Pattern Compression: Bins noisy candlestick pattern scores into a simple,
  discrete 5-point scale to reduce noise.
- Standardization: Rescales all other numerical features to a common scale
  (mean 0, std 1) using StandardScaler.
"""

import os
import re
import sys
import time
import logging
import traceback
from typing import List, Tuple

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

# --- CONFIGURATION IMPORT ---
# Calculate paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming structure: project_root/src/data_processing/bronze/bronze_data_generator.py
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
    # Fallback logging if setup hasn't run
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules. Ensure config.py and utils are accessible: {e}")
    sys.exit(1)

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS ---

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes a DataFrame's memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def _transform_relational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts absolute price columns to be relative to the current close price."""
    logger.info("  - Applying relational transformation...")
    if 'close' not in df.columns:
        raise KeyError("The required 'close' column is not present in the input DataFrame.")

    abs_price_patterns = re.compile(
        r'^(open|high|low|close)$|^(SMA|EMA)_\d+$|^BB_(upper|lower)_\d+$|'
        r'^(support|resistance)$|^ATR_level_.+_\d+$'
    )
    abs_price_cols = [col for col in df.columns if abs_price_patterns.match(col)]
    
    close_series = df['close']
    for col in abs_price_cols:
        if col != 'close':
            df[f'{col}_dist_norm'] = (df[col] - close_series) / close_series.replace(0, np.nan)
    
    df.drop(columns=list(set(abs_price_cols) | {'volume'}), inplace=True, errors='ignore')
    return df

def _encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Converts text-based categorical columns into numerical binary columns."""
    logger.info("  - Encoding categorical features...")
    categorical_cols = [
        col for col in df.columns
        if col.startswith(('session', 'trend_regime', 'vol_regime'))
    ]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=float)
    return df, categorical_cols

def _compress_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Bins noisy TA-Lib candlestick scores into a simple, discrete 5-point scale."""
    logger.info("  - Compressing candlestick patterns...")
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    
    def compress(v):
        if v >= 100: return 1.0   # Strong Bullish
        if v > 0: return 0.5     # Weak Bullish
        if v <= -100: return -1.0  # Strong Bearish
        if v < 0: return -0.5    # Weak Bearish
        return 0.0               # Neutral
        
    for col in candle_cols:
        df[col] = df[col].fillna(0).apply(compress)
    return df


def _scale_numeric_features_corrected(
    df: pd.DataFrame, original_cat_cols: List[str], window_size: int
) -> pd.DataFrame:
    """
    Standardizes numerical features using a rolling window to prevent look-ahead bias.
    This is the only methodologically sound way to scale time-series data for ML.
    """
    logger.info(f"  - Scaling numeric features using a {window_size}-period rolling window...")
    candle_cols = {col for col in df.columns if col.startswith("CDL")}
    
    one_hot_cols = {
        col for col in df.columns
        if any(cat_col_base in col for cat_col_base in original_cat_cols)
    }
    
    non_scalable_cols = candle_cols | one_hot_cols | {'time'}
    
    cols_to_scale = [
        col for col in df.columns
        if col not in non_scalable_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if not cols_to_scale:
        logger.warning("No numerical columns found to scale.")
        return df
        
    # Using a rolling window approach for time-series-safe standardization.
    for col in tqdm(cols_to_scale, desc="    -> Scaling columns", unit="col"):
        # Fill initial NaNs that might exist before calculating rolling stats
        df[col] = df[col].fillna(0)
        
        rolling_mean = df[col].rolling(window=window_size, min_periods=max(2, int(window_size*0.1))).mean()
        rolling_std = df[col].rolling(window=window_size, min_periods=max(2, int(window_size*0.1))).std()
        
        # Standardize: (value - rolling_mean) / rolling_std
        df[col] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # After rolling, the first 'window_size-1' rows will be NaN.
        # We backfill them, which is a reasonable approximation for the start of the series.
        # Then, fill any remaining NaNs (e.g., if std was zero) with 0.
        df[col] = df[col].bfill().fillna(0)
        
    logger.info("  - Scaling complete.")
    return df

def create_gold_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates the full preprocessing pipeline."""
    df = features_df.copy()
    
    df = _transform_relational_features(df)
    df, original_cat_cols = _encode_categorical_features(df)
    df = _compress_candlestick_patterns(df)
    df = _scale_numeric_features_corrected(df, original_cat_cols, c.GOLD_SCALER_ROLLING_WINDOW)
    
    logger.info("  - Finalizing and downcasting data types...")
    return downcast_dtypes(df)

def _process_single_file(paths_tuple: Tuple[str, str]) -> str:
    """Reads, processes, and saves a single Silver feature file into the Gold format."""
    silver_path, gold_path = paths_tuple
    fname = os.path.basename(silver_path)
    try:
        features_df = pd.read_parquet(silver_path)
        # Ensure 'time' column is datetime
        features_df['time'] = pd.to_datetime(features_df['time'])

        gold_dataset = create_gold_features(features_df)
        
        gold_dataset.to_parquet(gold_path, index=False)
        return f"SUCCESS: Gold data generated for {fname}."
    except Exception:
        logger.error(f"A fatal error occurred while processing {fname}.", exc_info=True)
        return f"ERROR: FAILED to process {fname}."


def main() -> None:
    """Main execution function."""
    # Setup Logging using Config levels
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "gold_layer")

    start_time = time.time()
    logger.info("--- Gold Layer: The ML Preprocessor (Parquet Edition) ---")

    # File Selection Logic
    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_file_arg:
        # User passed a specific file
        silver_path = os.path.join(p.SILVER_DATA_FEATURES_DIR, f"{target_file_arg}.parquet")
        if os.path.exists(silver_path):
            files_to_process = [f"{target_file_arg}.parquet"]
            logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
        else:
            logger.error(f"Target file not found: {silver_path}")
    else:
        # Standard Mode: Scan for new files
        new_files = scan_new_files(p.SILVER_DATA_FEATURES_DIR, p.GOLD_DATA_FEATURES_DIR)
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    logger.info(f"Queued {len(files_to_process)} file(s): {', '.join(files_to_process)}")
    for filename in files_to_process:
        logger.info(f"--- Processing {filename} ---")
        silver_path = os.path.join(p.SILVER_DATA_FEATURES_DIR, filename)
        gold_path = os.path.join(p.GOLD_DATA_FEATURES_DIR, filename) # Output name is the same
        result = _process_single_file((silver_path, gold_path))
        logger.info(result)

    end_time = time.time()
    logger.info(f"Gold Layer generation complete. Total Runtime: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()