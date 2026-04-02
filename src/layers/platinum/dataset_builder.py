# platinum_dataset_builder.py (V1.0 - Unified XGBoost Dataset)

"""
Platinum Layer: The Dataset Builder

This script constructs the massive training matrix required for the 
Unified XGBoost Model.

Process:
1. Loads 'Gold Features' (Market Context per Candle).
2. Loads 'Silver Trades' (55M+ Individual Trade Outcomes).
3. Performs a High-Performance Inner Join on 'entry_time'.
4. Selects relevant features:
   - Market Context (RSI, SMA_rel_close, etc.)
   - Trade Parameters (SL_Dist_BPS, TP_Dist_BPS)
   - Target (Outcome: 1/0)
5. Saves the result as a sharded Parquet dataset for memory-efficient training.
"""

import os
import sys
import gc
import logging
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --- CONFIGURATION IMPORT ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

try:
    import config.config as c
    from src.utils import paths as p
    from src.utils.logger import setup_logging 
    from src.utils.file_selector import scan_new_files, select_files_interactively # type: ignore
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- CORE LOGIC ---

def get_sl_tp_columns(df_columns: list) -> tuple:
    """
    Dynamically identifies the SL/TP distance columns from Silver data.
    We want the BPS columns, not the Percentage ones.
    """
    sl_cols = [col for col in df_columns if col.startswith("sl_dist_to_") and col.endswith("_bps")]
    tp_cols = [col for col in df_columns if col.startswith("tp_dist_to_") and col.endswith("_bps")]
    return sl_cols, tp_cols

def build_dataset_for_instrument(instrument_name: str):
    """
    Joins Gold (Market) and Silver (Trades) to create the Training Matrix.
    Writes output to Platinum/Datasets/{Instrument}.
    """
    # 1. Path Setup
    gold_path = os.path.join(p.GOLD_FEATURES_DIR, f"{instrument_name}.parquet")
    silver_dir = os.path.join(p.SILVER_CHUNKED_DIR, instrument_name)
    output_dir = os.path.join(p.PLATINUM_TARGETS, instrument_name) # Reusing targets dir for dataset
    
    if not os.path.exists(gold_path):
        logger.error(f"Gold features not found: {gold_path}")
        return
    if not os.path.exists(silver_dir):
        logger.error(f"Silver trades not found: {silver_dir}")
        return
        
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Gold Features (Market Context)
    # This is small (~100MB), so we keep it in memory for fast joining.
    logger.info("  - Loading Gold Features...")
    gold_df = pd.read_parquet(gold_path)
    # Ensure join key is datetime
    gold_df['entry_time'] = pd.to_datetime(gold_df['time'])
    gold_df = gold_df.drop(columns=['time']) # Remove duplicate time col
    
    # 3. Stream Silver Trades (The Massive Dataset)
    silver_files = sorted([f for f in os.listdir(silver_dir) if f.endswith('.parquet')])
    if not silver_files:
        logger.error("No Silver chunks found.")
        return

    logger.info(f"  - Processing {len(silver_files)} trade chunks...")
    
    total_rows = 0
    
    for i, f_name in enumerate(tqdm(silver_files, desc="Building Dataset")):
        chunk_path = os.path.join(silver_dir, f_name)
        
        # Load Trade Chunk
        silver_chunk = pd.read_parquet(chunk_path)
        if silver_chunk.empty: continue
        
        # Prepare Join Key
        silver_chunk['entry_time'] = pd.to_datetime(silver_chunk['entry_time'])
        
        # 4. The Join (Market Context + Trade Outcome)
        # Inner join means we only keep trades where we have valid market features.
        merged_df = pd.merge(
            silver_chunk, 
            gold_df, 
            on='entry_time', 
            how='inner'
        )
        
        if merged_df.empty: continue
        
        # 5. Feature Cleaning & encoding
        # Convert Outcome 'win'/'loss' to 1/0
        if 'outcome' in merged_df.columns:
            merged_df['target'] = (merged_df['outcome'] == 'win').astype('int8')
            merged_df.drop(columns=['outcome'], inplace=True)
        
        # Drop high-cardinality ID columns that confuse XGBoost
        cols_to_drop = ['entry_time', 'exit_time', 'entry_price', 'sl_price', 'tp_price']
        merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns], inplace=True)
        
        # Convert 'trade_type' (buy/sell) to 1/0
        if 'trade_type' in merged_df.columns:
            merged_df['is_buy'] = (merged_df['trade_type'] == 'buy').astype('int8')
            merged_df.drop(columns=['trade_type'], inplace=True)

        # 6. Save Shard
        # We save as a partitioned Parquet dataset for easy Dask/XGBoost loading later.
        out_path = os.path.join(output_dir, f"part_{i}.parquet")
        merged_df.to_parquet(out_path, index=False)
        
        total_rows += len(merged_df)
        
        # Memory Management
        del silver_chunk, merged_df
        gc.collect()

    logger.info(f"SUCCESS: Dataset built for {instrument_name}. Total Rows: {total_rows:,}")

def main():
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "platinum_builder")
    p.ensure_directories()
    logger.info("--- Platinum Dataset Builder (V1.0 - Unified XGBoost) ---")

    target_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_arg:
        # Non-interactive: orchestrator passed the instrument name directly
        logger.info(f"Targeted Mode: Processing '{target_arg}'")
        files_to_process = [target_arg]
    else:
        # Interactive: scan for unprocessed instrument directories
        all_dirs = [d for d in os.listdir(p.SILVER_CHUNKED_DIR)
                    if os.path.isdir(os.path.join(p.SILVER_CHUNKED_DIR, d))]
        to_process = [d for d in all_dirs
                      if not os.path.exists(os.path.join(p.PLATINUM_TARGETS, d))]
        files_to_process = select_files_interactively(to_process)

    if not files_to_process:
        logger.info("No instruments selected.")
        return

    for instr in files_to_process:
        logger.info(f"--- Processing {instr} ---")
        build_dataset_for_instrument(instr)

if __name__ == "__main__":
    main()