# gold_data_generator.py (V7.1 - Strict Scaling & Multi-Anchor)

"""
Gold Layer: The Machine Learning Preprocessor

This script represents the crucial final stage of data preparation in the
pipeline. It features a Multi-Anchor Normalization Engine and a Strict Scaling
policy to prevent signal corruption.
"""

import os
import re
import sys
import time
import logging
import traceback
from typing import List, Tuple, Set

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
except ImportError as e:
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

def _transform_relational_features_multi_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multiple normalization passes based on GOLD_NORMALIZATION_CONFIG.
    Generates features like 'high_rel_close', 'high_rel_open'.
    """
    logger.info("  - Applying Multi-Anchor Relational Transformation...")
    
    if not hasattr(c, "GOLD_NORMALIZATION_CONFIG"):
        raise AttributeError("GOLD_NORMALIZATION_CONFIG is missing from config.py")

    absolute_cols_to_drop: Set[str] = set()
    
    # Iterate through the config list
    for i, rule in enumerate(c.GOLD_NORMALIZATION_CONFIG):
        anchor_name = rule.get("anchor_col")
        targets_regex = rule.get("targets_regex")
        
        if not anchor_name or not targets_regex:
            continue
            
        if anchor_name not in df.columns:
            logger.debug(f"Skipping rule #{i+1}: Anchor '{anchor_name}' not found.")
            continue
            
        anchor_series = df[anchor_name]
        regex = re.compile(targets_regex)
        
        # Find all columns matching the regex
        target_cols = [col for col in df.columns if regex.match(col)]
        
        if not target_cols:
            continue
            
        logger.info(f"    -> Anchor: '{anchor_name}' | Normalizing {len(target_cols)} targets.")
        
        for target in target_cols:
            # Don't normalize a column against itself
            if target == anchor_name:
                absolute_cols_to_drop.add(target)
                continue
            
            # Create Relative Feature
            new_col = f"{target}_rel_{anchor_name}"
            
            # Formula: (Target - Anchor) / Anchor
            df[new_col] = (df[target] - anchor_series) / anchor_series.replace(0, np.nan)
            
            # Mark for later deletion
            absolute_cols_to_drop.add(target)
            absolute_cols_to_drop.add(anchor_name)

    # Cleanup: Drop original absolute price columns
    if 'volume' in df.columns:
        absolute_cols_to_drop.add('volume')
        
    cols_existing = [c for c in absolute_cols_to_drop if c in df.columns]
    logger.info(f"  - Dropping {len(cols_existing)} absolute price columns.")
    df.drop(columns=cols_existing, inplace=True)
    
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
    Standardizes numerical features using a rolling window.
    STRICTLY avoids scaling Relational and Time columns to preserve signal integrity.
    """
    logger.info(f"  - Scaling numeric features using a {window_size}-period rolling window...")
    
    # 1. Identify what NOT to scale
    
    # A. Candlestick Scores (-1 to 1)
    candle_cols = {col for col in df.columns if col.startswith("CDL")}
    
    # B. One-Hot Encoded Columns (0 or 1)
    one_hot_cols = {
        col for col in df.columns
        if any(cat_col_base in col for cat_col_base in original_cat_cols)
    }
    
    # C. Relational Columns (Percentage Distances)
    # Scaling these flips the sign relative to local history, destroying 'Above/Below' signals.
    rel_cols = {col for col in df.columns if "_rel_" in col}
    
    # D. Time Features (Cyclical/Raw)
    # Z-scoring these destroys the 0-23 cycle or 0-6 weekday logic.
    time_cols = {'hour', 'weekday', 'time'}
    
    non_scalable_cols = candle_cols | one_hot_cols | rel_cols | time_cols
    
    # 2. Select Columns to Scale
    # This leaves: Oscillators (RSI, CCI), Statistics (avg_body), etc.
    cols_to_scale = [
        col for col in df.columns
        if col not in non_scalable_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if not cols_to_scale:
        logger.warning("No numerical columns found to scale.")
        return df
        
    logger.info(f"    -> Identified {len(cols_to_scale)} columns for Z-Score scaling (Oscillators/Stats).")
    
    # 3. Apply Rolling Z-Score
    for col in tqdm(cols_to_scale, desc="    -> Scaling columns", unit="col"):
        df[col] = df[col].fillna(0)
        
        rolling_mean = df[col].rolling(window=window_size, min_periods=max(2, int(window_size*0.1))).mean()
        rolling_std = df[col].rolling(window=window_size, min_periods=max(2, int(window_size*0.1))).std()
        
        # Calculate Z-Score
        df[col] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # Backfill warmup period
        df[col] = df[col].bfill().fillna(0)
        
    logger.info("  - Scaling complete.")
    return df

def create_gold_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates the full preprocessing pipeline."""
    df = features_df.copy()
    
    # 1. Multi-Anchor Relational Transform
    df = _transform_relational_features_multi_anchor(df)
    
    # 2. Categorical Encoding
    df, original_cat_cols = _encode_categorical_features(df)
    
    # 3. Compression
    df = _compress_candlestick_patterns(df)
    
    # 4. Rolling Standardization (Strict Mode)
    df = _scale_numeric_features_corrected(df, original_cat_cols, c.GOLD_SCALER_ROLLING_WINDOW)
    
    logger.info("  - Finalizing and downcasting data types...")
    return downcast_dtypes(df)

def _process_single_file(paths_tuple: Tuple[str, str]) -> str:
    """Reads, processes, and saves a single Silver feature file into the Gold format."""
    silver_path, gold_path = paths_tuple
    fname = os.path.basename(silver_path)
    try:
        features_df = pd.read_parquet(silver_path)
        features_df['time'] = pd.to_datetime(features_df['time'])
        gold_dataset = create_gold_features(features_df)
        gold_dataset.to_parquet(gold_path, index=False)
        return f"SUCCESS: Gold data generated for {fname}."
    except Exception:
        logger.error(f"A fatal error occurred while processing {fname}.", exc_info=True)
        return f"ERROR: FAILED to process {fname}."


def main() -> None:
    """Main execution function."""
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "gold_layer")

    start_time = time.time()
    logger.info("--- Gold Layer: The ML Preprocessor (V7.1 - Strict Scaling) ---")

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        silver_path = os.path.join(p.SILVER_DATA_FEATURES_DIR, f"{target_file_arg}.parquet")
        if os.path.exists(silver_path):
            files_to_process = [f"{target_file_arg}.parquet"]
            logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
        else:
            logger.error(f"Target file not found: {silver_path}")
    else:
        new_files = scan_new_files(p.SILVER_DATA_FEATURES_DIR, p.GOLD_DATA_FEATURES_DIR)
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    logger.info(f"Queued {len(files_to_process)} file(s): {', '.join(files_to_process)}")
    for filename in files_to_process:
        logger.info(f"--- Processing {filename} ---")
        silver_path = os.path.join(p.SILVER_DATA_FEATURES_DIR, filename)
        gold_path = os.path.join(p.GOLD_DATA_FEATURES_DIR, filename)
        
        result = _process_single_file((silver_path, gold_path))
        logger.info(result)

    end_time = time.time()
    logger.info(f"Gold Layer generation complete. Total Runtime: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()