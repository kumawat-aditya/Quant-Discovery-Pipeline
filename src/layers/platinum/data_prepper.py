# platinum_preprocessor.py (V6.0 - Config Aligned)

"""
Platinum Pre-Processor: Unified Blueprint Discovery & Target Extraction

This script executes the Map-Reduce logic to discover strategy blueprints
from the Silver Layer trade chunks.

It consumes: Silver Layer Parquet Chunks (Trade Data)
It produces: 
    1. A 'Combinations' file (List of all valid strategies found)
    2. 'Target' files (Compressed binary files of trade outcomes for each strategy)
"""

import hashlib
import logging
import os
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool, current_process
from typing import Dict, List, Tuple

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
    logging.critical(f"Failed to import project modules: {e}")
    sys.exit(1)

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- PHASE 1 HELPERS ---

def nested_dd():
    """A pickle-safe helper for creating nested defaultdicts."""
    return defaultdict(int)

def _apply_binning(chunk: pd.DataFrame, all_levels: List[str]) -> pd.DataFrame:
    """
    Vectorized calculation of binning features.
    This creates the discrete grid for the Strategy Discovery.
    """
    binned_cols = {}
    bin_size_bps = c.PLATINUM_BPS_BIN_SIZE
    
    for level in all_levels:
        for sltp in ['sl', 'tp']:
            # Percentage Placement Binning (e.g., 50% -> Bin 5)
            pct_col = f"{sltp}_place_pct_to_{level}"
            if pct_col in chunk.columns:
                # Binning by 10% steps (0.1)
                binned_cols[f'{pct_col}_bin'] = np.floor(chunk[pct_col] * 10)
            
            # Basis Points Distance Binning (e.g., 12 bps -> Bin 10 if bin_size=5)
            bps_col = f"{sltp}_dist_to_{level}_bps"
            if bps_col in chunk.columns:
                binned_cols[f'{bps_col}_bin'] = np.floor(chunk[bps_col] / bin_size_bps) * bin_size_bps
                
    return pd.DataFrame(binned_cols)

def _aggregate_blueprints(agg_chunk: pd.DataFrame, all_levels: List[str]) -> Dict:
    """Performs groupby operations to discover unique strategy keys."""
    chunk_results = defaultdict(nested_dd)
    if 'trade_type' not in agg_chunk.columns:
        return chunk_results

    # Optimizing types for grouping speed
    agg_chunk['trade_type'] = agg_chunk['trade_type'].astype('category')
    
    for level in all_levels:
        # 1. Strategy Type: SL placed by Pct, TP by Ratio
        col = f'sl_place_pct_to_{level}_bin'
        if col in agg_chunk.columns:
            # Filter outliers (-200% to +200%)
            df_filt = agg_chunk[agg_chunk[col].between(-20, 20)]
            if not df_filt.empty:
                groups = df_filt.groupby([col, 'tp_ratio', 'trade_type', 'entry_time'], observed=True).size()
                for (sl_bin, tp_ratio, trade_type, time), count in groups.items():
                    if count > 0:
                        bp = ('SL-Pct', level, int(sl_bin), 'ratio', tp_ratio, trade_type)
                        chunk_results[bp][time] += count

        # 2. Strategy Type: TP placed by Pct, SL by Ratio
        col = f'tp_place_pct_to_{level}_bin'
        if col in agg_chunk.columns:
            # Filter for rows where the TP placement is within a reasonable range (-200% to +200%).
            df_filt = agg_chunk[agg_chunk[col].between(-20, 20)]
            if not df_filt.empty:
                groups = df_filt.groupby([col, 'sl_ratio', 'trade_type', 'entry_time'], observed=True).size()
                for (tp_bin, sl_ratio, trade_type, time), count in groups.items():
                    if count > 0:
                        bp = ('TP-Pct', 'ratio', sl_ratio, level, int(tp_bin), trade_type)
                        chunk_results[bp][time] += count

        # 3. Strategy Type: SL placed by BPS, TP by Ratio
        col = f'sl_dist_to_{level}_bps_bin'
        if col in agg_chunk.columns:
            # Filter outliers (-50 to +50 bins * 5bps = -250 to +250 bps)
            df_filt = agg_chunk[agg_chunk[col].between(-50, 50)]
            if not df_filt.empty:
                groups = df_filt.groupby([col, 'tp_ratio', 'trade_type', 'entry_time'], observed=True).size()
                for (sl_bin, tp_ratio, trade_type, time), count in groups.items():
                    if count > 0:
                        bp = ('SL-BPS', level, int(sl_bin), 'ratio', tp_ratio, trade_type)
                        chunk_results[bp][time] += count

        # 4. Strategy Type: TP placed by BPS, SL by Ratio
        col = f'tp_dist_to_{level}_bps_bin'
        if col in agg_chunk.columns:
            # Filter for rows where the TP distance is within a reasonable range (-50 to +50 bps).
            df_filt = agg_chunk[agg_chunk[col].between(-50, 50)]
            if not df_filt.empty:
                groups = df_filt.groupby([col, 'sl_ratio', 'trade_type', 'entry_time'], observed=True).size()
                for (tp_bin, sl_ratio, trade_type, time), count in groups.items():
                    if count > 0:
                        bp = ('TP-BPS', 'ratio', sl_ratio, level, int(tp_bin), trade_type)
                        chunk_results[bp][time] += count
                        
    return chunk_results

# --- PHASE 1: WORKER FUNCTION (MAPPER) ---
def discover_and_aggregate_chunk(task_tuple: Tuple[str, List[str]]) -> Dict:
    """Worker for Phase 1. Discovers blueprints from one chunk."""
    chunk_path, all_levels = task_tuple
    try:
        chunk = pd.read_parquet(chunk_path)
        if chunk.empty: return {'status': 'success', 'result': {}}
        
        # Rounding for consistency in grouping
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
        
        binned_df = _apply_binning(chunk, all_levels)
        
        # Combine necessary columns
        agg_chunk = pd.concat([chunk[['entry_time', 'sl_ratio', 'tp_ratio', 'trade_type']], binned_df], axis=1)
        
        chunk_results = _aggregate_blueprints(agg_chunk, all_levels)
        return {'status': 'success', 'result': chunk_results}
        
    except Exception as e:
        return {
            'status': 'error', 
            'worker': current_process().name, 
            'chunk': os.path.basename(chunk_path),
            'error': str(e), 
            'traceback': traceback.format_exc()
        }

# --- PHASE 2: SHARDED BUFFER FLUSHING ---
def flush_buffer_to_shards(buffer: List[Tuple], temp_dir: str):
    """Converts the buffer to a DataFrame and appends it to sharded Parquet files."""
    if not buffer: return
    
    df = pd.DataFrame(buffer, columns=['key', 'entry_time', 'trade_count'])
    
    # Simple sharding: Hash Key -> Integer Modulo N
    df['shard_id'] = df['key'].apply(lambda x: int(x, 16) % c.PLATINUM_NUM_SHARDS)
    
    for shard_id, group in df.groupby('shard_id'):
        filename = f"{c.PLATINUM_TEMP_SHARD_PREFIX}{shard_id}.parquet"
        filepath = os.path.join(temp_dir, filename)
        
        new_table = pa.Table.from_pandas(
            group[['key', 'entry_time', 'trade_count']], 
            preserve_index=False
        )
        
        if os.path.exists(filepath):
            existing_table = pq.read_table(filepath)
            combined_table = pa.concat_tables([existing_table, new_table])
            pq.write_table(combined_table, filepath)
        else:
            pq.write_table(new_table, filepath)

# --- PHASE 3: WORKER FUNCTION (REDUCER) ---
def consolidate_shard_file(task_tuple: Tuple[str, str]) -> Dict[str, int]:
    """Worker for Phase 3. Consolidates one shard file into final target files."""
    shard_path, final_dir = task_tuple
    candle_counts = {}
    try:
        df = pd.read_parquet(shard_path)
        if not df.empty:
            for key, group in df.groupby('key'):
                final_df = group.groupby('entry_time')['trade_count'].sum().reset_index()
                
                # Validation: Does this strategy have enough volume?
                if len(final_df) >= c.PLATINUM_MIN_CANDLE_LIMIT:
                    candle_counts[key] = len(final_df)
                    final_target_path = os.path.join(final_dir, f"{key}.parquet")
                    final_df.to_parquet(final_target_path, index=False)
                    
        os.remove(shard_path) # Cleanup temp file
    except Exception:
        logger.warning(f"Failed to consolidate shard {os.path.basename(shard_path)}.", exc_info=True)
    return candle_counts

# --- MAIN ORCHESTRATOR ---
def run_preprocessor_for_instrument(instrument_name: str) -> None:
    """Orchestrates the entire Map-Reduce process for a single instrument."""
    
    # Define Paths using 'paths.py' where possible, or constructing them relative
    chunked_outcomes_dir = os.path.join(p.SILVER_DATA_CHUNKED_OUTCOMES_DIR, instrument_name)
    combinations_path = os.path.join(p.PLATINUM_DATA_COMBINATIONS_DIR, f"{instrument_name}.parquet")
    temp_targets_dir = os.path.join(p.PLATINUM_DATA_TEMP_TARGETS_DIR, instrument_name)
    final_targets_dir = os.path.join(p.PLATINUM_DATA_TARGETS_DIR, instrument_name)
    
    # Cleanup / Prep
    if os.path.exists(combinations_path):
        logger.info(f"{instrument_name} already has a combinations file. Skipping.")
        return
    
    for d in [temp_targets_dir, final_targets_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Validate Input
    try:
        if not os.path.exists(chunked_outcomes_dir):
            logger.error(f"Silver chunks directory not found: {chunked_outcomes_dir}")
            return
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.parquet')]
        if not chunk_files:
            logger.error(f"No Silver chunk files found for {instrument_name}.")
            return
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
        return

    # Dynamic Level Detection
    try:
        pq_file = pq.ParquetFile(chunk_files[0])
        column_names = pq_file.schema.names
    except Exception as e:
        logger.error(f"Could not read schema from '{chunk_files[0]}': {e}")
        return
        
    level_pattern = re.compile(r'(?:sl|tp)_(?:place_pct_to|dist_to)_([a-zA-Z0-9_]+)')
    all_levels = sorted({match.group(1) for col in column_names for match in [level_pattern.match(col)] if match})
    logger.info(f"Discovered {len(all_levels)} market levels for {instrument_name}.")

    # --- Phase 1 & 2: Map & Shuffle ---
    logger.info("--- Phase 1: Discovering Blueprints and Streaming to Shards ---")
    tasks = [(path, all_levels) for path in chunk_files]
    master_blueprint_keys = {}
    master_buffer = []

    try:
        with Pool(processes=c.MAX_CPU_USAGE) as pool:
            with tqdm(total=len(tasks), desc="Phase 1: Processing Chunks") as pbar:
                for result in pool.imap_unordered(discover_and_aggregate_chunk, tasks):
                    
                    if result['status'] == 'error':
                        logger.critical(f"Worker '{result['worker']}' failed on chunk '{result['chunk']}'. Error: {result['error']}")
                        pool.terminate()
                        sys.exit(1)
                    
                    # Accumulate Results
                    for blueprint, counts in result['result'].items():
                        if blueprint not in master_blueprint_keys:
                            key_str = '-'.join(map(str, blueprint))
                            master_blueprint_keys[blueprint] = hashlib.sha256(key_str.encode()).hexdigest()[:16]
                        
                        key = master_blueprint_keys[blueprint]
                        for time_val, count in counts.items():
                            master_buffer.append((key, time_val, count))
                    
                    # Memory Management Flush
                    if len(master_buffer) >= c.PLATINUM_BUFFER_FLUSH_THRESHOLD:
                        flush_buffer_to_shards(master_buffer, temp_targets_dir)
                        master_buffer.clear()
                    
                    pbar.update(1)
                    
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(1)
    except Exception:
        logger.critical("Fatal exception in Phase 1.", exc_info=True)
        sys.exit(1)

    # Final Flush
    if master_buffer:
        flush_buffer_to_shards(master_buffer, temp_targets_dir)
    
    logger.info(f"Phase 1 Complete. Discovered {len(master_blueprint_keys)} unique blueprints.")

    # --- Phase 3: Reduce ---
    logger.info("--- Phase 2: Consolidating Shard Files ---")
    shard_files = [os.path.join(temp_targets_dir, f) for f in os.listdir(temp_targets_dir) if f.startswith(c.PLATINUM_TEMP_SHARD_PREFIX)]
    master_candle_counts = {}
    
    if not shard_files:
        logger.warning("No shard files generated. Skipping Phase 2.")
    else:
        tasks = [(path, final_targets_dir) for path in shard_files]
        with Pool(processes=c.MAX_CPU_USAGE) as pool:
            for result_dict in tqdm(pool.imap_unordered(consolidate_shard_file, tasks), total=len(tasks), desc="Phase 2: Consolidating"):
                master_candle_counts.update(result_dict)
    
    logger.info("Phase 2 Complete. Final targets saved.")
    
    # Save Combinations Definitions
    definitions = []
    for bp, key in master_blueprint_keys.items():
        # Only save definitions for strategies that met the candle limit in consolidation
        # If master_candle_counts doesn't have the key, it means it had < PLATINUM_MIN_CANDLE_LIMIT
        count = master_candle_counts.get(key, 0)
        if count > 0:
            definitions.append({
                'key': key, 
                'type': bp[0], 
                'sl_def': bp[1], 'sl_bin': bp[2], 
                'tp_def': bp[3], 'tp_bin': bp[4], 
                'trade_type': bp[5],
                'num_candles': count
            })

    if definitions:
        definitions_df = pd.DataFrame(definitions)
        definitions_df.to_parquet(combinations_path, index=False)
        logger.info(f"Saved {len(definitions)} valid strategy combinations to {os.path.basename(combinations_path)}")
    else:
        logger.warning("No valid strategies found after filtering.")
    
    # Cleanup Temp Directory
    if os.path.exists(temp_targets_dir):
        shutil.rmtree(temp_targets_dir)

def main() -> None:
    """Main execution function."""
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "platinum_preprocessor")
    
    start_time = time.time()
    logger.info("--- Platinum Pre-Processor: Unified Blueprint Discovery (V6.0) ---")

    # File Selection Logic
    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # User passed a specific file
        target_path = os.path.join(p.SILVER_DATA_CHUNKED_OUTCOMES_DIR, target_file_arg)
        if os.path.exists(target_path):
            files_to_process = [target_file_arg]
            logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
        else:
            logger.error(f"Target file not found: {target_path}")
    else:
        # Standard Mode: Scan for new files
        new_files = scan_new_files(p.SILVER_DATA_CHUNKED_OUTCOMES_DIR, p.PLATINUM_DATA_COMBINATIONS_DIR)
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    logger.info(f"Processing {len(files_to_process)} instruments...")
    for instrument in files_to_process:
        logger.info(f"--- Processing: {instrument} ---")
        run_preprocessor_for_instrument(instrument)
    
    end_time = time.time()
    logger.info(f"Job finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()