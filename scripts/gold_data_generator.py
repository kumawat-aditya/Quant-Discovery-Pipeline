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

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    logging.critical("'pyarrow' library not found. Please run 'pip install pyarrow'.")
    sys.exit(1)

# --- PROJECT-LEVEL IMPORTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    import config
    from scripts.logger_setup import setup_logging
except ImportError as e:
    logging.critical(f"Failed to import project modules. Ensure config.py and logger_setup.py are accessible: {e}")
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

# ### <<< CRITICAL CHANGE: Corrected scaling logic to prevent look-ahead bias.
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
    # ### <<< CHANGE: Calling the new, corrected scaling function.
    df = _scale_numeric_features_corrected(df, original_cat_cols, config.GOLD_SCALER_ROLLING_WINDOW)
    
    logger.info("  - Finalizing and downcasting data types...")
    return downcast_dtypes(df)

def _process_single_file(paths_tuple: Tuple[str, str]) -> str:
    """Reads, processes, and saves a single Silver feature file into the Gold format."""
    silver_path, gold_path = paths_tuple
    fname = os.path.basename(silver_path)
    try:
        # ### <<< CHANGE: Read Parquet instead of CSV.
        features_df = pd.read_parquet(silver_path)
        # Ensure 'time' column is datetime
        features_df['time'] = pd.to_datetime(features_df['time'])

        gold_dataset = create_gold_features(features_df)
        
        gold_dataset.to_parquet(gold_path, index=False)
        return f"SUCCESS: Gold data generated for {fname}."
    except Exception:
        logger.error(f"A fatal error occurred while processing {fname}.", exc_info=True)
        return f"ERROR: FAILED to process {fname}."

def _select_files_interactively(silver_dir: str, gold_dir: str) -> List[str]:
    """Scans for new Silver files and prompts the user to select which to process."""
    logger.info("Interactive Mode: Scanning for new files...")
    try:
        # ### <<< CHANGE: Look for .parquet files from the Silver layer.
        silver_files = sorted([f for f in os.listdir(silver_dir) if f.endswith('.parquet')])
        gold_bases = {os.path.splitext(f)[0] for f in os.listdir(gold_dir) if f.endswith('.parquet')}
        new_files = [f for f in silver_files if os.path.splitext(f)[0] not in gold_bases]

        if not new_files:
            logger.info("No new Silver feature files to process.")
            return []

        print("\n--- Select File(s) to Process ---")
        for i, f in enumerate(new_files): print(f"  [{i+1}] {f}")
        print("  [a] Process All New Files")
        print("\nEnter selection (e.g., 1,3 or a):")
        
        user_input = input("> ").strip().lower()
        if not user_input: return []
        if user_input == 'a': return new_files

        selected_files = []
        try:
            indices = {int(i.strip()) - 1 for i in user_input.split(',')}
            for idx in sorted(indices):
                if 0 <= idx < len(new_files): selected_files.append(new_files[idx])
                else: logger.warning(f"Invalid selection '{idx + 1}' ignored.")
            return selected_files
        except ValueError:
            logger.error("Invalid input. Please enter numbers (e.g., 1,3) or 'a'.")
            return []
    except FileNotFoundError:
        logger.error(f"The Silver features directory was not found at: {silver_dir}")
        return []

def main() -> None:
    """Main execution function."""
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    LOGS_DIR = os.path.join(PROJECT_ROOT, config.LOG_DIR)
    setup_logging(LOGS_DIR, config.CONSOLE_LOG_LEVEL, config.FILE_LOG_LEVEL)

    start_time = time.time()
    SILVER_FEATURES_DIR = os.path.join(PROJECT_ROOT, 'silver_data', 'features')
    GOLD_FEATURES_DIR = os.path.join(PROJECT_ROOT, 'gold_data', 'features')
    os.makedirs(GOLD_FEATURES_DIR, exist_ok=True)

    logger.info("--- Gold Layer: The ML Preprocessor (Parquet Edition) ---")

    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_file_arg:
        logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
        # ### <<< CHANGE: Look for .parquet file extension.
        if not target_file_arg.endswith('.parquet'): target_file_arg += '.parquet'
        silver_path_check = os.path.join(SILVER_FEATURES_DIR, target_file_arg)
        files_to_process = [target_file_arg] if os.path.exists(silver_path_check) else []
        if not files_to_process: logger.error(f"Target file not found: {silver_path_check}")
    else:
        files_to_process = _select_files_interactively(SILVER_FEATURES_DIR, GOLD_FEATURES_DIR)

    if not files_to_process:
        logger.info("No files selected or found for processing. Exiting.")
    else:
        logger.info(f"Queued {len(files_to_process)} file(s): {', '.join(files_to_process)}")
        for filename in files_to_process:
            logger.info(f"--- Processing {filename} ---")
            silver_path = os.path.join(SILVER_FEATURES_DIR, filename)
            gold_path = os.path.join(GOLD_FEATURES_DIR, filename) # Output name is the same
            result = _process_single_file((silver_path, gold_path))
            logger.info(result)

    end_time = time.time()
    logger.info(f"Gold Layer generation finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()