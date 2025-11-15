# diamond_data_prepper.py (V1.1 - Full Blueprint Integration)

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

import os
import re
import sys
import hashlib
import subprocess
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
import time

import pandas as pd
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[FATAL] 'pyarrow' library not found. Please run 'pip install pyarrow'.")
    sys.exit(1)

# --- CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)

# --- WORKER-SPECIFIC GLOBAL ---
worker_gold_features_df: pd.DataFrame

def init_worker(gold_features_df: pd.DataFrame):
    """Initializer for each worker process, making the Gold DF available."""
    global worker_gold_features_df
    worker_gold_features_df = gold_features_df

# --- PHASE 1: DATA PREPARATION ---

def prepare_cross_market_data(master_instrument: str, base_dirs: Dict[str, str]) -> List[str]:
    """
    Ensures Silver and Gold feature files exist for all instruments of the same
    timeframe as the master instrument.
    """
    print("\n--- Phase 1: Preparing Cross-Market Data ---")
    
    timeframe_match = re.search(r'(\d+)', master_instrument)
    if not timeframe_match:
        print(f"[ERROR] Could not parse timeframe from master instrument '{master_instrument}'.")
        return []
    timeframe = timeframe_match.group(1)
    
    print(f"[INFO] Master timeframe detected: {timeframe}m. Scanning for related instruments...")
    
    try:
        raw_files = os.listdir(base_dirs['raw'])
    except FileNotFoundError:
        print(f"[ERROR] Raw data directory not found at: {base_dirs['raw']}")
        return []

    all_relevant_instruments = sorted([
        os.path.splitext(f)[0] for f in raw_files if f.endswith('.csv') and timeframe in f
    ])
    
    if not all_relevant_instruments:
        print("[WARN] No relevant instruments found for this timeframe.")
        return []

    print(f"Found {len(all_relevant_instruments)} instruments: {', '.join(all_relevant_instruments)}")

    for instrument in tqdm(all_relevant_instruments, desc="Checking Data Integrity"):
        silver_path = os.path.join(base_dirs['silver_features'], f"{instrument}.csv")
        gold_path = os.path.join(base_dirs['gold_features'], f"{instrument}.parquet")
        
        if not os.path.exists(silver_path):
            print(f"\n[INFO] Silver features for {instrument} are missing. Generating...")
            try:
                subprocess.run([
                    sys.executable, os.path.join(base_dirs['scripts'], 'silver_data_generator.py'),
                    f"{instrument}.csv", "--features-only"
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to generate Silver features for {instrument}.")
                print(f"Stderr: {e.stderr}")
                continue

        if not os.path.exists(gold_path):
            print(f"\n[INFO] Gold features for {instrument} are missing. Generating...")
            try:
                subprocess.run([
                    sys.executable, os.path.join(base_dirs['scripts'], 'gold_data_generator.py'),
                    f"{instrument}.csv"
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to generate Gold features for {instrument}.")
                print(f"Stderr: {e.stderr}")

    print("[SUCCESS] All required Silver and Gold data files are prepared.")
    return all_relevant_instruments

# --- PHASE 2: TRIGGER EXTRACTION ---

def process_strategy_task(
    strategy_row: pd.Series, 
    instrument_to_scan: str, 
    master_instrument: str,
    base_dirs: Dict[str, str]
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
        return False


def extract_triggers_for_instrument(master_instrument: str, all_instruments: List[str], base_dirs: Dict[str, str]):
    """
    Orchestrates the discovery and saving of trigger times for new strategies.
    """
    print("\n--- Phase 2: Extracting Trigger Times ---")
    
    # ### <<< MODIFIED BLOCK: This entire section is updated to handle the merge.
    
    # Step 1: Load all three necessary data sources.
    platinum_strategies_path = os.path.join(base_dirs['platinum_strategies'], f"{master_instrument}.parquet")
    platinum_combo_path = os.path.join(base_dirs['platinum_combo'], f"{master_instrument}.parquet")
    diamond_strategies_path = os.path.join(base_dirs['diamond_strategies'], f"{master_instrument}.parquet")

    try:
        source_strategies_df = pd.read_parquet(platinum_strategies_path)
        combo_df = pd.read_parquet(platinum_combo_path)
    except FileNotFoundError as e:
        print(f"[ERROR] A required Platinum file is missing: {e}. Cannot proceed.")
        return

    if source_strategies_df.empty:
        print("[INFO] No discovered strategies found in Platinum layer. Nothing to process.")
        return

    # Step 2: Enrich the discovered strategies with their full blueprint definitions.
    # This merge adds 'type', 'sl_def', 'tp_def', 'trade_type', etc., to each strategy.
    enriched_strategies_df = pd.merge(source_strategies_df, combo_df, on='key', how='left')

    # Step 3: Identify what is new by comparing against the existing Diamond file.
    # The comparison now uses 'key' and 'market_rule' as a composite unique identifier.
    try:
        processed_strategies_df = pd.read_parquet(diamond_strategies_path)
        new_strategies_df = enriched_strategies_df.merge(
            processed_strategies_df[['key', 'market_rule']],
            on=['key', 'market_rule'],
            how='left',
            indicator=True
        ).query('_merge == "left_only"').drop('_merge', axis=1)
    except FileNotFoundError:
        processed_strategies_df = pd.DataFrame() # Create empty DF if it doesn't exist
        new_strategies_df = enriched_strategies_df.copy()

    # ### <<< END OF MODIFIED BLOCK

    if new_strategies_df.empty:
        print("[INFO] No new strategies found to process.")
        return
        
    print(f"Found {len(new_strategies_df)} new strategies to process.")

    # Generate unique trigger_key for each new strategy
    new_strategies_df['trigger_key'] = new_strategies_df.apply(
        lambda row: hashlib.sha256(f"{row['key']}-{row['market_rule']}".encode()).hexdigest()[:16],
        axis=1
    )
    
    for instrument_to_scan in all_instruments:
        print(f"\nScanning for triggers on market: {instrument_to_scan}...")
        gold_path = os.path.join(base_dirs['gold_features'], f"{instrument_to_scan}.parquet")
        try:
            gold_df = pd.read_parquet(gold_path)
            gold_df['time'] = pd.to_datetime(gold_df['time']).dt.tz_localize(None)
        except FileNotFoundError:
            print(f"[WARN] Gold features file not found for {instrument_to_scan}. Skipping.")
            continue
            
        tasks = [row for _, row in new_strategies_df.iterrows()]
        
        worker_func = partial(
            process_strategy_task,
            instrument_to_scan=instrument_to_scan,
            master_instrument=master_instrument,
            base_dirs=base_dirs
        )
        
        with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=(gold_df,)) as pool:
            list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc=f"Processing on {instrument_to_scan}"))

    # Finalization: Append new, fully-defined strategies to the diamond strategies file
    print("\n[FINALIZE] Appending new strategies to the master list...")
    # Use concat which is safer for this append operation
    final_df = pd.concat([processed_strategies_df, new_strategies_df], ignore_index=True)
    
    final_df.to_parquet(diamond_strategies_path, index=False)
    print(f"[SUCCESS] Saved/updated master strategy list at {diamond_strategies_path}")


def _select_instrument_interactively(platinum_dir: str) -> List[str]:
    """Scans for instruments in the Platinum layer and prompts user for selection."""
    print("[INFO] Interactive Mode: Scanning for instruments with discovered strategies...")
    try:
        all_instruments = sorted([
            os.path.splitext(f)[0] for f in os.listdir(platinum_dir) if f.endswith('.parquet')
        ])
        if not all_instruments:
            print("[INFO] No discovered strategy files found in Platinum layer.")
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
            if 0 <= idx < len(all_instruments):
                selected.append(all_instruments[idx])
            else:
                print(f"[WARN] Invalid selection '{idx + 1}' ignored.")
        return selected
    except ValueError:
        print("[ERROR] Invalid input. Please enter numbers or 'a'.")
        return []
    except FileNotFoundError:
        print(f"[ERROR] Platinum strategies directory not found at: {platinum_dir}")
        return []


def main():
    """Main execution function."""
    start_time = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    base_dirs = {
        'scripts': SCRIPT_DIR,
        'raw': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'raw_data')),
        'silver_features': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'silver_data', 'features')),
        'gold_features': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'gold_data', 'features')),
        'platinum_strategies': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'platinum_data', 'discovered_strategies')),
        'platinum_combo': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'platinum_data', 'combinations')), # <-- Added combo path
        'diamond_strategies': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'strategies')),
        'triggers': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'triggers')),
    }
    
    os.makedirs(base_dirs['diamond_strategies'], exist_ok=True)
    os.makedirs(base_dirs['triggers'], exist_ok=True)

    print("--- Diamond Layer - Prepper: The Trigger Engine (V1.1) ---")
    
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        print(f"\n[INFO] Targeted Mode: Processing master instrument '{target_arg}'")
        instruments_to_process = [target_arg]
    else:
        # ### <<< MODIFIED: The selection should be based on discovered strategies, not combos
        instruments_to_process = _select_instrument_interactively(base_dirs['platinum_strategies'])

    if not instruments_to_process:
        print("\n[INFO] No instruments selected. Exiting.")
    else:
        print(f"\n[QUEUE] Queued {len(instruments_to_process)} master instrument(s): {', '.join(instruments_to_process)}")
        for master_instrument in instruments_to_process:
            print(f"\n{'='*70}\nProcessing Master Instrument: {master_instrument}\n{'='*70}")
            try:
                all_relevant_instruments = prepare_cross_market_data(master_instrument, base_dirs)
                
                if all_relevant_instruments:
                    extract_triggers_for_instrument(master_instrument, all_relevant_instruments, base_dirs)
            except Exception:
                print(f"\n[FATAL ERROR] A critical error occurred while processing {master_instrument}.")
                traceback.print_exc()

    end_time = time.time()
    print(f"\nDiamond data preparation finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()