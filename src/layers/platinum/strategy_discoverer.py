# platinum_strategy_discoverer.py (V4.0 - Central Config & Logging)

"""
Platinum Layer - Strategy Discoverer: The Rule Miner

This is the intelligent heart of the discovery pipeline. It uses a machine
learning model (DecisionTreeRegressor) to mine the vast, preprocessed Gold
dataset for explicit, human-readable trading rules associated with profitable
strategy blueprints.

This script embodies a "Two-Phase Learning" architecture for maximum efficiency
and continuous improvement:

Phase 1: Discovery
- A comprehensive run to find initial rules for all new and unprocessed
  strategy blueprints. This phase is fully resumable and is designed to find
  high-quality patterns in blueprints that have never been seen before.

Phase 2: Iterative Improvement
- A fast, targeted run that ONLY re-evaluates blueprints for which the
  backtester has provided new negative feedback (i.e., new blacklisted rules).
  This is the core of the iterative learning loop. It uses a "data pruning"
  technique to force the Decision Tree to find novel, alternative rules,
  avoiding patterns that are known to be unprofitable.

The entire process is highly parallelized, resumable, and designed to become
smarter over time by learning from backtesting results.
"""

import logging
import os
import sys
import traceback
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.tree import DecisionTreeRegressor
except ImportError:
    print("CRITICAL: 'scikit-learn' library not found. Please run 'pip install scikit-learn'.")
    sys.exit(1)
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
except ImportError as e:
    # Fallback logging if setup hasn't run
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules. Ensure config.py and utils are accessible: {e}")
    sys.exit(1)

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- WORKER-SPECIFIC GLOBALS ---
worker_gold_features_df: pd.DataFrame

def init_worker(gold_features_df: pd.DataFrame):
    """Initializer for each worker process in the Pool."""
    global worker_gold_features_df
    worker_gold_features_df = gold_features_df

# --- HELPER & CORE ML LOGIC ---

def _ensure_paths_exist(paths: Dict[str, str]):
    """Ensures all necessary output files and directories exist before processing."""
    for key, path in paths.items():
        if key in ['strategies', 'blacklists', 'exhausted', 'processed_log']:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                if path.endswith('.log'):
                    open(path, 'w').close()
                elif key == 'blacklists':
                    pq.write_table(pa.Table.from_pydict({'key': [], 'market_rule': []}), path)
                elif key == 'strategies':
                    pq.write_table(pa.Table.from_pydict({'key': [], 'market_rule': [], 'n_candles': [], 'avg_density': []}), path)
                elif key == 'exhausted':
                    pq.write_table(pa.Table.from_pydict({'key': []}), path)

def _load_keys_from_file(filepath: str) -> Set[str]:
    """Loads keys from a simple text file, one key per line."""
    try:
        with open(filepath, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()
    except Exception as e:
        logger.warning(f"Could not read keys from {filepath}. Error: {e}")
        return set()

def _load_keys_from_parquet(filepath: str, column_name: str = 'key') -> Set[str]:
    """Efficiently loads a single column from a Parquet file into a set."""
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            df = pd.read_parquet(filepath, columns=[column_name])
            return set(df[column_name].dropna().unique())
    except Exception as e:
        logger.warning(f"Could not read keys from Parquet file {filepath}. Error: {e}")
    return set()

def get_rule_from_tree(tree, feature_names: List[str]) -> List[Dict]:
    """Recursively traverses a trained Decision Tree to extract human-readable rules."""
    rules = []
    def recurse(node_id: int, path: List[str]):
        if tree.feature[node_id] != -2:
            name = feature_names[tree.feature[node_id]]
            threshold = round(tree.threshold[node_id], 5)
            recurse(tree.children_left[node_id], path + [f"`{name}` <= {threshold}"])
            recurse(tree.children_right[node_id], path + [f"`{name}` > {threshold}"])
        else:
            rules.append({
                'rule': " and ".join(path),
                'n_candles': tree.n_node_samples[node_id],
                'avg_density': tree.value[node_id][0][0]
            })
    recurse(0, [])
    return rules

def find_rules_with_decision_tree(training_df: pd.DataFrame, exclusion_rules: Set[str]) -> Dict:
    """Trains a Decision Tree on a blueprint's data and extracts high-quality rules."""
    pruned_df = training_df.copy()
    baseline_avg_density = pruned_df['trade_count'].mean()
    if pd.isna(baseline_avg_density) or baseline_avg_density == 0:
        return {'status': 'exhausted'}

    for rule_str in exclusion_rules:
        try:
            indices = pruned_df.query(rule_str).index
            if not indices.empty:
                pruned_df.loc[indices, 'trade_count'] = 0
        except Exception:
            pass

    X = pruned_df.drop(columns=['time', 'trade_count'])
    y = pruned_df['trade_count']
    if y.sum() == 0:
        return {'status': 'exhausted'}

    model = DecisionTreeRegressor(
        max_depth=c.PLATINUM_DT_MAX_DEPTH,
        min_samples_leaf=c.PLATINUM_MIN_CANDLES_PER_RULE,
        random_state=42
    )
    model.fit(X, y)
    
    all_rules = get_rule_from_tree(model.tree_, X.columns)
    valid_new_rules = [
        rule_info for rule_info in all_rules
        if rule_info['avg_density'] >= (baseline_avg_density * c.PLATINUM_DENSITY_LIFT_THRESHOLD)
        and rule_info['rule'] and rule_info['rule'] not in exclusion_rules
    ]
    return {'status': 'success', 'rules': valid_new_rules} if valid_new_rules else {'status': 'exhausted'}

# --- PARALLEL WORKER FUNCTION ---
def process_key_batch(task_tuple: Tuple[List[Tuple[str, str]], Dict[str, Set[str]]]) -> Dict:
    """Processes a batch of blueprints (keys) to discover new trading rules."""
    key_paths_batch, exclusion_rules_by_key = task_tuple
    discovered_in_batch, exhausted_in_batch = [], []
    for key, target_path in key_paths_batch:
        try:
            target_df = pd.read_parquet(target_path)
            target_df['time'] = pd.to_datetime(target_df['entry_time'])
            
            training_df = pd.merge(worker_gold_features_df, target_df[['time', 'trade_count']], on='time', how='inner')
            if training_df.empty: continue

            rules_to_exclude = exclusion_rules_by_key.get(key, set())
            result = find_rules_with_decision_tree(training_df, rules_to_exclude)
            
            if result['status'] == 'success':
                for rule_info in result['rules']:
                    discovered_in_batch.append({'key': key, 'market_rule': rule_info['rule'], 'n_candles': rule_info['n_candles'], 'avg_density': rule_info['avg_density']})
            elif result['status'] == 'exhausted':
                exhausted_in_batch.append(key)
        except Exception:
            logger.error(f"Worker failed to process key {key}.", exc_info=True)
    return {'strategies': discovered_in_batch, 'exhausted_keys': exhausted_in_batch}

# --- ORCHESTRATION FUNCTIONS ---

def _load_all_data_sources(paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Set]:
    """Loads all necessary data files for an instrument."""
    
    gold_df = pd.read_parquet(paths['gold'])
    gold_df['time'] = pd.to_datetime(gold_df['time']) 
    
    all_blueprints_df = pd.read_parquet(paths['combo'])
    
    exclusion_rules_by_key = defaultdict(set)
    df_discovered = pd.read_parquet(paths['strategies'], columns=['key', 'market_rule'])
    df_blacklist = pd.read_parquet(paths['blacklists'], columns=['key', 'market_rule'])
    for _, row in pd.concat([df_discovered, df_blacklist]).iterrows():
        exclusion_rules_by_key[row['key']].add(row['market_rule'])
        
    exhausted_keys = _load_keys_from_parquet(paths['exhausted'])
    return gold_df, all_blueprints_df, exclusion_rules_by_key, exhausted_keys

def execute_parallel_processing(
    tasks: List[Tuple], gold_df: pd.DataFrame, desc: str
) -> Tuple[List[Dict], List[str]]:
    """Manages the multiprocessing pool and executes a list of tasks."""
    if not tasks: return [], []
    all_new_strategies, all_exhausted_keys = [], []
    with Pool(processes=c.MAX_CPU_USAGE, initializer=init_worker, initargs=(gold_df,)) as pool:
        for result in tqdm(pool.imap_unordered(process_key_batch, tasks), total=len(tasks), desc=desc):
            if result['strategies']: all_new_strategies.extend(result['strategies'])
            if result['exhausted_keys']: all_exhausted_keys.extend(result['exhausted_keys'])
    return all_new_strategies, all_exhausted_keys

def run_discovery_for_instrument(instrument_name: str, base_dirs: Dict[str, str]):
    """Main orchestration logic for a single instrument."""
    logger.info("SETUP: Loading all required data sources...")
    try:
        paths = {
        'gold': os.path.join(base_dirs['gold'], instrument_name),
        'combo': os.path.join(base_dirs['platinum_combo'], instrument_name),
        'targets_dir': os.path.join(base_dirs['platinum_targets'], instrument_name.replace(".parquet", '')),
        'strategies': os.path.join(base_dirs['platinum_strategies'], instrument_name),
        'blacklists': os.path.join(base_dirs['platinum_blacklists'], instrument_name),
        'exhausted': os.path.join(base_dirs['platinum_exhausted'], instrument_name),
        'processed_log': os.path.join(base_dirs['platinum_logs'], instrument_name.replace(".parquet", ".processed.log"))
        }
        _ensure_paths_exist(paths)
        gold_df, all_blueprints_df, exclusion_rules, exhausted_keys = _load_all_data_sources(paths)
    except FileNotFoundError as e:
        logger.error(f"A required input file is missing: {e}. Aborting.")
        return

    # --- PHASE 1: Discovery ---
    logger.info("--- Phase 1: Discovery of New Blueprints ---")
    processed_keys = _load_keys_from_file(paths['processed_log'])
    df_filtered = all_blueprints_df[
        (all_blueprints_df['num_candles'] >= c.PLATINUM_MIN_CANDLE_LIMIT) &
        (~all_blueprints_df['key'].isin(exhausted_keys)) &
        (~all_blueprints_df['key'].isin(processed_keys))
    ]
    discovery_keys = df_filtered['key'].tolist()
    logger.info(f"Found {len(discovery_keys)} new blueprints to process.")
    
    discovery_paths = [(key, os.path.join(paths['targets_dir'], f"{key}.parquet")) for key in discovery_keys]
    discovery_tasks = [
        (batch, exclusion_rules) for batch in np.array_split(discovery_paths, max(1, len(discovery_paths) // c.PLATINUM_DISCOVERY_BATCH_SIZE))
    ]
    new_strategies, exhausted_discovery = execute_parallel_processing(discovery_tasks, gold_df, "Phase 1: Discovery")
    
    # --- PHASE 2: Iterative Improvement ---
    logger.info("--- Phase 2: Iterative Improvement from Blacklist ---")
    feedback_keys = _load_keys_from_parquet(paths['blacklists'])
    df_filtered = all_blueprints_df[
        (all_blueprints_df['key'].isin(feedback_keys)) &
        (all_blueprints_df['num_candles'] >= c.PLATINUM_MIN_CANDLE_LIMIT) &
        (~all_blueprints_df['key'].isin(exhausted_keys))
    ]
    improvement_keys = df_filtered['key'].tolist()
    logger.info(f"Found {len(improvement_keys)} blueprints with new feedback to re-process.")

    improvement_paths = [(key, os.path.join(paths['targets_dir'], f"{key}.parquet")) for key in improvement_keys]
    improvement_tasks = [
        (batch, exclusion_rules) for batch in np.array_split(improvement_paths, max(1, len(improvement_paths) // c.PLATINUM_DISCOVERY_BATCH_SIZE))
    ]
    relearned_strategies, exhausted_improvement = execute_parallel_processing(improvement_tasks, gold_df, "Phase 2: Improvement")
    
    # --- Final Write to Disk ---
    logger.info("FINALIZE: Saving all results to disk...")
    all_new_strategies = new_strategies + relearned_strategies
    if all_new_strategies:
        new_strategies_df = pd.DataFrame(all_new_strategies)
        existing_strategies_df = pd.read_parquet(paths['strategies'])
        combined_df = pd.concat([existing_strategies_df, new_strategies_df]).drop_duplicates(subset=['key', 'market_rule'], keep='last')
        combined_df.to_parquet(paths['strategies'], index=False)
        logger.info(f"  - Saved {len(all_new_strategies)} new/updated strategies.")

    all_exhausted = set(exhausted_discovery + exhausted_improvement)
    if all_exhausted:
        exhausted_df = pd.DataFrame(list(all_exhausted), columns=['key'])
        existing_exhausted_df = pd.read_parquet(paths['exhausted'])
        combined_df = pd.concat([existing_exhausted_df, exhausted_df]).drop_duplicates(subset=['key'])
        combined_df.to_parquet(paths['exhausted'], index=False)
        logger.info(f"  - Marked {len(all_exhausted)} blueprints as exhausted.")
        
    if discovery_keys:
        # ### <<< CHANGE: Improved log writing.
        with open(paths['processed_log'], 'a') as f:
            for key in discovery_keys:
                f.write(f"{key}\n")
        logger.info(f"  - Logged {len(discovery_keys)} blueprints as processed.")

def main():
    """Main execution function."""
    # Setup Logging using Config levels
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "platinum_discoverer_layer")

    start_time = time.time()
    base_dirs = {
        'gold': p.GOLD_DATA_FEATURES_DIR,
        'platinum_combo': p.PLATINUM_DATA_COMBINATIONS_DIR,
        'platinum_targets': p.PLATINUM_DATA_TARGETS_DIR,
        'platinum_strategies': p.PLATINUM_DATA_STRATEGIES_DIR,
        'platinum_blacklists': p.PLATINUM_DATA_BLACKLISTS_DIR,
        'platinum_exhausted': p.PLATINUM_DATA_EXHAUSTED_KEYS_DIR,
        'platinum_logs': p.PLATINUM_DATA_DISCOVERY_LOG_DIR,
    }

    # File Selection Logic
    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # User passed a specific file
        target_path = os.path.join(base_dirs['platinum_combo'], f"{target_file_arg}.parquet")
        if os.path.exists(target_path):
            files_to_process = [f"{target_file_arg}.parquet"]
            logger.info(f"Targeted Mode: Processing '{target_file_arg}'")
        else:
            logger.error(f"Target file not found: {target_path}")
    else:
        # Standard Mode: Scan for new files
        new_files = scan_new_files(p.PLATINUM_DATA_COMBINATIONS_DIR, p.PLATINUM_DATA_STRATEGIES_DIR)
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    logger.info(f"Queued {len(files_to_process)} instrument(s): {', '.join(files_to_process)}")
    for instrument in files_to_process:
        logger.info(f"--- Processing Instrument: {instrument} ---")
        run_discovery_for_instrument(instrument, base_dirs)
    
    end_time = time.time()
    logger.info(f"Strategy discovery finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()