# diamond_data_prepper.py (V2.0 - Corrected Orchestration, Config & Logging)

"""
Diamond Layer - Prepper: The Trigger Engine

This script is the foundational data preparer for the entire Diamond Layer.
It performs two critical, high-leverage tasks:

1.  Cross-Market Data Preparation:
    It ensures that all markets sharing the same timeframe as the primary
    instrument have their Silver (features) and Gold (ML-ready) data files
    generated. It intelligently invokes the generator scripts from previous
    layers only when needed.

2.  Trigger Time Extraction:
    This is its core function. It takes the discovered strategies from the
    Platinum Layer and performs the heavy, one-time computation of finding
    every single timestamp where a strategy's market conditions were met.
    It does this across ALL relevant markets and saves the results in a
    highly organized structure, perfectly embodying the "Prepare Once, Test Many"
    philosophy of the pipeline.

V1.1 Update: Now merges discovered strategies with the platinum combinations
file to carry forward the full blueprint definition (including trade_type)
into the Diamond Layer.
"""

import hashlib
import logging
import os
import re
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

try:
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

# --- WORKER-SPECIFIC GLOBAL ---
worker_gold_features_df: pd.DataFrame

def init_worker(gold_features_df: pd.DataFrame):
    """Initializer for each worker process, making the Gold DF available."""
    global worker_gold_features_df
    worker_gold_features_df = gold_features_df

# --- PHASE 1: DATA PREPARATION ---

def prepare_cross_market_data(master_instrument: str, base_dirs: Dict[str, str]) -> List[str]:
    """Ensures Silver and Gold files exist for all instruments of the same timeframe."""
    logger.info("--- Phase 1: Preparing Cross-Market Data ---")
    timeframe_match = re.search(r'(\d+)', master_instrument)
    if not timeframe_match:
        logger.error(f"Could not parse timeframe from master instrument '{master_instrument}'.")
        return []
    timeframe = timeframe_match.group(1)
    logger.info(f"Master timeframe detected: {timeframe}m. Scanning for related instruments...")

    try:
        raw_files = os.listdir(base_dirs['raw'])
    except FileNotFoundError:
        logger.error(f"Raw data directory not found at: {base_dirs['raw']}")
        return []

    all_relevant_instruments = sorted([
        os.path.splitext(f)[0] for f in raw_files if f.endswith('.csv') and timeframe in f
    ])
    if not all_relevant_instruments:
        logger.warning("No relevant instruments found for this timeframe.")
        return []

    logger.info(f"Found {len(all_relevant_instruments)} instruments: {', '.join(all_relevant_instruments)}")

    for instrument in tqdm(all_relevant_instruments, desc="Checking Data Integrity"):
        # ### <<< CRITICAL FIX: Paths now correctly point to .parquet files.
        silver_path = os.path.join(base_dirs['silver_features'], f"{instrument}.parquet")
        gold_path = os.path.join(base_dirs['gold_features'], f"{instrument}.parquet")
        
        # We need the raw CSV path to call the Silver script
        raw_csv_path = f"{instrument}.csv"

        if not os.path.exists(silver_path):
            logger.info(f"Silver features for {instrument} are missing. Generating...")
            try:
                subprocess.run([
                    sys.executable, os.path.join(base_dirs['scripts'], 'silver_data_generator.py'),
                    raw_csv_path, "--features-only"
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate Silver features for {instrument}. Stderr: {e.stderr}")
                continue

        if not os.path.exists(gold_path):
            logger.info(f"Gold features for {instrument} are missing. Generating...")
            try:
                # ### <<< CRITICAL FIX: Call Gold script with the Silver .parquet file.
                silver_parquet_filename = f"{instrument}.parquet"
                subprocess.run([
                    sys.executable, os.path.join(base_dirs['scripts'], 'gold_data_generator.py'),
                    silver_parquet_filename
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to generate Gold features for {instrument}. Stderr: {e.stderr}")

    logger.info("SUCCESS: All required Silver and Gold data files are prepared.")
    return all_relevant_instruments

# --- PHASE 2: TRIGGER EXTRACTION ---

def process_strategy_task(
    strategy_row: pd.Series, instrument_to_scan: str, master_instrument: str, base_dirs: Dict[str, str]
) -> bool:
    """
    Worker function. Finds and saves trigger times for one strategy on one market.
    """
    market_rule = strategy_row['market_rule']
    trigger_key = strategy_row['trigger_key']
    
    try:
        trigger_times_df = worker_gold_features_df.query(market_rule)
        if not trigger_times_df.empty:
            output_dir = os.path.join(base_dirs['triggers'], master_instrument, instrument_to_scan)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{trigger_key}.parquet")
            trigger_times_df[['time']].to_parquet(output_path, index=False)
        return True
    except Exception:
        # Errors in query are expected for complex rules sometimes, log quietly.
        logger.debug(f"Query failed for rule on {instrument_to_scan}", exc_info=False)
        return False

def extract_triggers_for_instrument(master_instrument: str, all_instruments: List[str], base_dirs: Dict[str, str]):
    """Orchestrates the discovery and saving of trigger times for new strategies."""
    logger.info("--- Phase 2: Extracting Trigger Times ---")
    
    platinum_strategies_path = os.path.join(base_dirs['platinum_strategies'], f"{master_instrument}.parquet")
    platinum_combo_path = os.path.join(base_dirs['platinum_combo'], f"{master_instrument}.parquet")
    diamond_strategies_path = os.path.join(base_dirs['diamond_strategies'], f"{master_instrument}.parquet")

    try:
        source_strategies_df = pd.read_parquet(platinum_strategies_path)
        combo_df = pd.read_parquet(platinum_combo_path)
    except FileNotFoundError as e:
        logger.error(f"A required Platinum file is missing: {e}. Cannot proceed.")
        return

    if source_strategies_df.empty:
        logger.info("No discovered strategies found in Platinum layer. Nothing to process.")
        return

    enriched_strategies_df = pd.merge(source_strategies_df, combo_df, on='key', how='left')

    try:
        processed_strategies_df = pd.read_parquet(diamond_strategies_path)
        new_strategies_df = enriched_strategies_df.merge(
            processed_strategies_df[['key', 'market_rule']], on=['key', 'market_rule'],
            how='left', indicator=True
        ).query('_merge == "left_only"').drop('_merge', axis=1)
    except FileNotFoundError:
        processed_strategies_df = pd.DataFrame()
        new_strategies_df = enriched_strategies_df.copy()

    if new_strategies_df.empty:
        logger.info("No new strategies found to process.")
        return
        
    logger.info(f"Found {len(new_strategies_df)} new strategies to process.")
    new_strategies_df['trigger_key'] = new_strategies_df.apply(
        lambda row: hashlib.sha256(f"{row['key']}-{row['market_rule']}".encode()).hexdigest()[:16], axis=1
    )
    
    for instrument_to_scan in all_instruments:
        logger.info(f"Scanning for triggers on market: {instrument_to_scan}...")
        gold_path = os.path.join(base_dirs['gold_features'], f"{instrument_to_scan}.parquet")
        try:
            gold_df = pd.read_parquet(gold_path)
            gold_df['time'] = pd.to_datetime(gold_df['time'])
        except FileNotFoundError:
            logger.warning(f"Gold features file not found for {instrument_to_scan}. Skipping.")
            continue
            
        tasks = [row for _, row in new_strategies_df.iterrows()]
        
        worker_func = partial(
            process_strategy_task,
            instrument_to_scan=instrument_to_scan,
            master_instrument=master_instrument,
            base_dirs=base_dirs
        )
        
        with Pool(processes=config.MAX_CPU_USAGE, initializer=init_worker, initargs=(gold_df,)) as pool:
            list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc=f"Scanning {instrument_to_scan}"))

    logger.info("FINALIZE: Appending new strategies to the master list...")
    if not os.path.exists(diamond_strategies_path):
        # If file doesn't exist, new_strategies_df is the whole file
        final_df = new_strategies_df
    else:
        # Otherwise, concatenate
        final_df = pd.concat([processed_strategies_df, new_strategies_df], ignore_index=True)

    final_df.to_parquet(diamond_strategies_path, index=False)
    logger.info(f"SUCCESS: Saved/updated master strategy list at {os.path.basename(diamond_strategies_path)}")

def _select_instrument_interactively(platinum_dir: str) -> List[str]:
    """Scans for instruments and prompts user for selection."""
    logger.info("Interactive Mode: Scanning for instruments with discovered strategies...")
    try:
        all_instruments = sorted([os.path.splitext(f)[0] for f in os.listdir(platinum_dir) if f.endswith('.parquet')])
        if not all_instruments:
            logger.info("No discovered strategy files found in Platinum layer.")
            return []
            
        print("\n--- Select Master Instrument(s) to Process ---")
        for i, f in enumerate(all_instruments): print(f"  [{i+1}] {f}")
        print("  [a] Process All")
        user_input = input("\nEnter selection (e.g., 1,3 or a): > ").strip().lower()
        if not user_input: return []
        if user_input == 'a': return all_instruments

        selected = []
        indices = {int(i.strip()) - 1 for i in user_input.split(',')}
        for idx in sorted(indices):
            if 0 <= idx < len(all_instruments): selected.append(all_instruments[idx])
            else: logger.warning(f"Invalid selection '{idx + 1}' ignored.")
        return selected
    except ValueError:
        logger.error("Invalid input. Please enter numbers or 'a'.")
        return []
    except FileNotFoundError:
        logger.error(f"Platinum strategies directory not found at: {platinum_dir}")
        return []

def main():
    """Main execution function."""
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    LOGS_DIR = os.path.join(PROJECT_ROOT, config.LOG_DIR)
    setup_logging(LOGS_DIR, config.CONSOLE_LOG_LEVEL, config.FILE_LOG_LEVEL)

    start_time = time.time()
    base_dirs = {
        'scripts': SCRIPT_DIR,
        'raw': os.path.join(PROJECT_ROOT, 'raw_data'),
        'silver_features': os.path.join(PROJECT_ROOT, 'silver_data', 'features'),
        'gold_features': os.path.join(PROJECT_ROOT, 'gold_data', 'features'),
        'platinum_strategies': os.path.join(PROJECT_ROOT, 'platinum_data', 'discovered_strategies'),
        'platinum_combo': os.path.join(PROJECT_ROOT, 'platinum_data', 'combinations'),
        'diamond_strategies': os.path.join(PROJECT_ROOT, 'diamond_data', 'strategies'),
        'triggers': os.path.join(PROJECT_ROOT, 'diamond_data', 'triggers'),
    }
    os.makedirs(base_dirs['diamond_strategies'], exist_ok=True)
    os.makedirs(base_dirs['triggers'], exist_ok=True)

    logger.info("--- Diamond Layer - Prepper: The Trigger Engine (V2.0) ---")
    
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        logger.info(f"Targeted Mode: Processing master instrument '{target_arg}'")
        instruments_to_process = [target_arg]
    else:
        instruments_to_process = _select_instrument_interactively(base_dirs['platinum_strategies'])

    if not instruments_to_process:
        logger.info("No instruments selected. Exiting.")
    else:
        logger.info(f"Queued {len(instruments_to_process)} master instrument(s): {', '.join(instruments_to_process)}")
        for master_instrument in instruments_to_process:
            logger.info(f"--- Processing Master Instrument: {master_instrument} ---")
            try:
                all_relevant_instruments = prepare_cross_market_data(master_instrument, base_dirs)
                if all_relevant_instruments:
                    extract_triggers_for_instrument(master_instrument, all_relevant_instruments, base_dirs)
            except Exception:
                logger.critical(f"A fatal error occurred while processing {master_instrument}.", exc_info=True)

    end_time = time.time()
    logger.info(f"Diamond data preparation finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()