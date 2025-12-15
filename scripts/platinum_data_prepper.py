# platinum_preprocessor.py (V5.0 - Central Config & Logging)

"""
Platinum Pre-Processor: Unified Blueprint Discovery & Target Extraction

This high-performance script unifies the discovery of strategy "blueprints" and
the pre-computation of their performance data (targets). It replaces the two
previous Platinum stages (`combinations_generator` and `target_extractor`) with
a single, more efficient Map-Reduce style architecture.

Phase 1: Parallel Discovery (The "Mapper")
- It reads the enriched Silver data Parquet chunks in parallel.
- Each worker (mapper) discovers all unique strategy blueprints within its
  assigned chunk and aggregates the trade counts for each blueprint per candle.
- The results are streamed from all workers to a large in-memory buffer in the
  main process.

Phase 2: Sharded Streaming (The "Shuffle")
- When the buffer reaches a size threshold, a "flush" operation is triggered.
- The buffer is "sharded" based on a hash of the strategy's unique key.
- Results are appended in large, efficient batches to a small, fixed number of
  temporary Parquet shard files. This solves the I/O bottleneck of writing to
  thousands of tiny files by concentrating all writes.

Phase 3: Parallel Consolidation (The "Reducer")
- After all chunks have been processed, a final parallel process begins.
- Each worker (reducer) is assigned one temporary shard file.
- It loads its shard, performs a final aggregation (groupby('key')), and writes
  the final, clean Parquet target files, one for each unique strategy.
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
from functools import partial
from multiprocessing import Pool, current_process
from typing import Dict, List, Tuple

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
    import config.config as config
    from config.logger_setup import setup_logging
except ImportError as e:
    logging.critical(f"Failed to import project modules. Ensure config.py and logger_setup.py are accessible: {e}")
    sys.exit(1)

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- PICKLE-SAFE HELPER ---
def nested_dd():
    """A pickle-safe helper for creating nested defaultdicts."""
    return defaultdict(int)

# --- REFACTORED PHASE 1 HELPERS ---

def _apply_binning(chunk: pd.DataFrame, all_levels: List[str]) -> pd.DataFrame:
    """Vectorized calculation of all binning features for a chunk."""
    binned_cols = {}
    for level in all_levels:
        for sltp in ['sl', 'tp']:
            pct_col = f"{sltp}_place_pct_to_{level}"
            bps_col = f"{sltp}_dist_to_{level}_bps"
            if pct_col in chunk.columns:
                binned_cols[f'{pct_col}_bin'] = np.floor(chunk[pct_col] * 10)
            if bps_col in chunk.columns:
                # ### <<< CHANGE: Using bin size from config.
                bin_size = config.PLATINUM_BPS_BIN_SIZE
                binned_cols[f'{bps_col}_bin'] = np.floor(chunk[bps_col] / bin_size) * bin_size
    return pd.DataFrame(binned_cols)

def _aggregate_blueprints(agg_chunk: pd.DataFrame, all_levels: List[str]) -> Dict:
    """Performs groupby operations to discover and count all blueprint occurrences."""
    chunk_results = defaultdict(nested_dd)
    if 'trade_type' not in agg_chunk.columns:
        raise KeyError("'trade_type' column not found in chunk. Cannot create directional blueprints.")

    for level in all_levels:
        # --- SL-Pct ---
        col = f'sl_place_pct_to_{level}_bin'
        if col in agg_chunk.columns:
            df_filtered = agg_chunk[agg_chunk[col].between(-20, 20, inclusive='both')]
            groups = df_filtered.dropna(subset=[col, 'tp_ratio']).groupby([col, 'tp_ratio', 'trade_type', 'entry_time'], observed=True).size()
            for (sl_bin, tp_ratio, trade_type, time), count in groups.items():
                blueprint = ('SL-Pct', level, int(sl_bin), 'ratio', tp_ratio, trade_type)
                chunk_results[blueprint][time] += count
        # --- TP-Pct ---
        col = f'tp_place_pct_to_{level}_bin'
        if col in agg_chunk.columns:
            # Filter for rows where the TP placement is within a reasonable range (-200% to +200%).
            df_filtered = agg_chunk[agg_chunk[col].between(-20, 20, inclusive='both')]
            groups = df_filtered.dropna(subset=[col, 'sl_ratio']).groupby([col, 'sl_ratio', 'trade_type', 'entry_time'], observed=True).size()
            for (tp_bin, sl_ratio, trade_type, time), count in groups.items():
                blueprint = ('TP-Pct', 'ratio', sl_ratio, level, int(tp_bin), trade_type)
                chunk_results[blueprint][time] += count
        # --- SL-BPS ---
        col = f'sl_dist_to_{level}_bps_bin'
        if col in agg_chunk.columns:
            df_filtered = agg_chunk[agg_chunk[col].between(-50, 50, inclusive='both')]
            groups = df_filtered.dropna(subset=[col, 'tp_ratio']).groupby([col, 'tp_ratio', 'trade_type', 'entry_time'], observed=True).size()
            for (sl_bin, tp_ratio, trade_type, time), count in groups.items():
                blueprint = ('SL-BPS', level, sl_bin, 'ratio', tp_ratio, trade_type)
                chunk_results[blueprint][time] += count
        # --- TP-BPS ---
        col = f'tp_dist_to_{level}_bps_bin'
        if col in agg_chunk.columns:
            # Filter for rows where the TP distance is within a reasonable range (-50 to +50 bps).
            df_filtered = agg_chunk[agg_chunk[col].between(-50, 50, inclusive='both')]
            groups = df_filtered.dropna(subset=[col, 'sl_ratio']).groupby([col, 'sl_ratio', 'trade_type', 'entry_time'], observed=True).size()
            for (tp_bin, sl_ratio, trade_type, time), count in groups.items():
                blueprint = ('TP-BPS', 'ratio', sl_ratio, level, tp_bin, trade_type)
                chunk_results[blueprint][time] += count
    return chunk_results

# --- PHASE 1: WORKER FUNCTION (MAPPER) ---
def discover_and_aggregate_chunk(task_tuple: Tuple[str, List[str]]) -> Dict:
    """Worker for Phase 1. Discovers blueprints from one chunk."""
    chunk_path, all_levels = task_tuple
    try:
        chunk = pd.read_parquet(chunk_path)
        if chunk.empty: return {'status': 'success', 'result': {}}
        
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
        
        binned_df = _apply_binning(chunk, all_levels)
        agg_chunk = pd.concat([chunk[['entry_time', 'sl_ratio', 'tp_ratio', 'trade_type']], binned_df], axis=1)
        
        chunk_results = _aggregate_blueprints(agg_chunk, all_levels)
        return {'status': 'success', 'result': chunk_results}
    except Exception as e:
        return {'status': 'error', 'worker': current_process().name, 'chunk': os.path.basename(chunk_path),
                'error': str(e), 'traceback': traceback.format_exc()}

# --- PHASE 2: SHARDED BUFFER FLUSHING ---
def flush_buffer_to_shards(buffer: List[Tuple], temp_dir: str):
    """Converts the buffer to a DataFrame and appends it to sharded Parquet files."""
    if not buffer: return
    
    df = pd.DataFrame(buffer, columns=['key', 'entry_time', 'trade_count'])
    # ### <<< CHANGE: Using num shards from config.
    df['shard_id'] = df['key'].apply(lambda x: int(x, 16) % config.PLATINUM_NUM_SHARDS)
    
    for shard_id, group in df.groupby('shard_id'):
        # ### <<< CHANGE: Using shard prefix from config.
        filepath = os.path.join(temp_dir, f"{config.PLATINUM_TEMP_SHARD_PREFIX}{shard_id}.parquet")
        new_table = pa.Table.from_pandas(group[['key', 'entry_time', 'trade_count']], preserve_index=False)
        
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
                candle_counts[key] = len(final_df)
                final_target_path = os.path.join(final_dir, f"{key}.parquet")
                final_df.to_parquet(final_target_path, index=False)
        os.remove(shard_path)
    except Exception:
        logger.warning(f"Failed to consolidate shard {os.path.basename(shard_path)}.", exc_info=True)
    return candle_counts

# --- MAIN ORCHESTRATOR ---
def run_preprocessor_for_instrument(instrument_name: str, base_dirs: Dict[str, str]) -> None:
    """Orchestrates the entire Map-Reduce process for a single instrument."""
    chunked_outcomes_dir = os.path.join(base_dirs['silver'], instrument_name)
    combinations_path = os.path.join(base_dirs['platinum_combo'], f"{instrument_name}.parquet")
    temp_targets_dir = os.path.join(base_dirs['platinum_temp'], instrument_name)
    final_targets_dir = os.path.join(base_dirs['platinum_final'], instrument_name)
    
    if os.path.exists(combinations_path):
        logger.info(f"{instrument_name} already has a combinations file. Skipping.")
        return
    
    for d in [temp_targets_dir, final_targets_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    try:
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.parquet')]
        if not chunk_files:
            logger.error(f"No Silver chunk files found for {instrument_name}.")
            return
    except FileNotFoundError:
        logger.error(f"Silver chunks directory not found: {chunked_outcomes_dir}")
        return

    try:
        pq_file = pq.ParquetFile(chunk_files[0])
        column_names = pq_file.schema.names
    except Exception as e:
        logger.error(f"Could not read schema from chunk file '{chunk_files[0]}': {e}")
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
        # ### <<< CHANGE: Using CPU count from config.
        with Pool(processes=config.MAX_CPU_USAGE) as pool:
            with tqdm(total=len(tasks), desc="Phase 1: Processing Chunks") as pbar:
                for result in pool.imap_unordered(discover_and_aggregate_chunk, tasks):
                    if result['status'] == 'error':
                        logger.critical(f"Worker '{result['worker']}' failed on chunk '{result['chunk']}'. Error: {result['error']}\nTerminating.")
                        pool.terminate()
                        sys.exit(1)
                    
                    for blueprint, counts in result['result'].items():
                        if blueprint not in master_blueprint_keys:
                            key_str = '-'.join(map(str, blueprint))
                            master_blueprint_keys[blueprint] = hashlib.sha256(key_str.encode()).hexdigest()[:16]
                        key = master_blueprint_keys[blueprint]
                        for time, count in counts.items():
                            master_buffer.append((key, time, count))
                    
                    # ### <<< CHANGE: Using buffer threshold from config.
                    if len(master_buffer) >= config.PLATINUM_BUFFER_FLUSH_THRESHOLD:
                        flush_buffer_to_shards(master_buffer, temp_targets_dir)
                        master_buffer.clear()
                    pbar.update(1)
    except Exception:
        logger.critical("The main process encountered a fatal exception.", exc_info=True)
        sys.exit(1)

    if master_buffer:
        flush_buffer_to_shards(master_buffer, temp_targets_dir)
    logger.info(f"Phase 1 Complete. Discovered {len(master_blueprint_keys)} unique blueprints.")

    # --- Phase 3: Reduce ---
    logger.info("--- Phase 2: Consolidating Shard Files ---")
    shard_files = [os.path.join(temp_targets_dir, f) for f in os.listdir(temp_targets_dir) if f.startswith(config.PLATINUM_TEMP_SHARD_PREFIX)]
    master_candle_counts = {}
    if not shard_files:
        logger.warning("No temporary shard files were generated. Skipping consolidation.")
    else:
        tasks = [(path, final_targets_dir) for path in shard_files]
        with Pool(processes=config.MAX_CPU_USAGE) as pool:
            for result_dict in tqdm(pool.imap_unordered(consolidate_shard_file, tasks), total=len(tasks), desc="Phase 2: Consolidating Shards"):
                master_candle_counts.update(result_dict)
    logger.info("Phase 2 Complete. Final targets saved.")
    
    definitions = [{'key': key, 'type': bp[0], 'sl_def': bp[1], 'sl_bin': bp[2], 
                    'tp_def': bp[3], 'tp_bin': bp[4], 'trade_type': bp[5]} 
                   for bp, key in master_blueprint_keys.items()]
    definitions_df = pd.DataFrame(definitions)
    definitions_df['num_candles'] = definitions_df['key'].map(master_candle_counts).fillna(0).astype(int)
    definitions_df.to_parquet(combinations_path, index=False)
    logger.info(f"Saved final combinations file to {os.path.basename(combinations_path)}")
    
    if os.path.exists(temp_targets_dir):
        shutil.rmtree(temp_targets_dir)

def _select_instruments_interactively(silver_dir: str, combo_dir: str) -> List[str]:
    """Scans for new instrument folders and prompts the user for selection."""
    logger.info("Interactive Mode: Scanning for new instruments...")
    try:
        all_instruments = sorted([d for d in os.listdir(silver_dir) if os.path.isdir(os.path.join(silver_dir, d))])
        processed_bases = {os.path.splitext(f)[0] for f in os.listdir(combo_dir) if f.endswith('.parquet')}
        new_instruments = [inst for inst in all_instruments if inst not in processed_bases]

        if not new_instruments:
            logger.info("No new instruments to process.")
            return []

        print("\n--- Select Instrument(s) to Process ---")
        for i, f in enumerate(new_instruments): print(f"  [{i+1}] {f}")
        print("  [a] Process All New Instruments")
        print("\nEnter selection (e.g., 1,3 or a):")
        user_input = input("> ").strip().lower()
        if not user_input: return []
        if user_input == 'a': return new_instruments

        selected = []
        try:
            indices = {int(i.strip()) - 1 for i in user_input.split(',')}
            for idx in sorted(indices):
                if 0 <= idx < len(new_instruments): selected.append(new_instruments[idx])
                else: logger.warning(f"Invalid selection '{idx + 1}' ignored.")
            return selected
        except ValueError:
            logger.error("Invalid input. Please enter numbers (e.g., 1,3) or 'a'.")
            return []
    except FileNotFoundError:
        logger.error(f"The Silver chunks directory was not found at: {silver_dir}")
        return []

def main() -> None:
    """Main execution function."""
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    LOGS_DIR = os.path.join(PROJECT_ROOT, config.LOG_DIR)
    setup_logging(LOGS_DIR, config.CONSOLE_LOG_LEVEL, config.FILE_LOG_LEVEL)

    start_time = time.time()
    base_dirs = {
        'silver': os.path.join(PROJECT_ROOT, 'silver_data', 'chunked_outcomes'),
        'platinum_combo': os.path.join(PROJECT_ROOT, 'platinum_data', 'combinations'),
        'platinum_temp': os.path.join(PROJECT_ROOT, 'platinum_data', 'temp_targets'),
        'platinum_final': os.path.join(PROJECT_ROOT, 'platinum_data', 'targets')
    }
    # Create directories, handling potential permission issues on some systems
    for d in base_dirs.values(): os.makedirs(d, exist_ok=True)

    logger.info("--- Platinum Pre-Processor: Unified Blueprint Discovery ---")

    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        logger.info(f"Targeted Mode: Processing instrument '{target_arg}'")
        if os.path.isdir(os.path.join(base_dirs['silver'], target_arg)):
            instruments_to_process = [target_arg]
        else:
            logger.error(f"Silver chunk directory not found for: {target_arg}")
            instruments_to_process = []
    else:
        instruments_to_process = _select_instruments_interactively(base_dirs['silver'], base_dirs['platinum_combo'])
    
    if not instruments_to_process:
        logger.info("No instruments selected for processing. Exiting.")
    else:
        logger.info(f"Queued {len(instruments_to_process)} instrument(s): {', '.join(instruments_to_process)}")
        for instrument in instruments_to_process:
            logger.info(f"--- Processing Instrument: {instrument} ---")
            run_preprocessor_for_instrument(instrument, base_dirs)
    
    end_time = time.time()
    logger.info(f"Platinum pre-processing finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()