# platinum_strategy_discoverer.py (V6.1 - Initialization Fix)

"""
Platinum Layer - Strategy Discoverer: The Rule Miner

This is the intelligent heart of the discovery pipeline. It uses a machine
learning model (DecisionTreeRegressor) to mine the vast, preprocessed Gold
dataset for explicit, human-readable trading rules.

Updates in V6.1:
- Fixed PyArrow initialization error (explicit empty dictionary structure).
- Config Aligned & Rule Simplification included.
"""

import logging
import os
import re
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

# Initialize logger
logger = logging.getLogger(__name__)

# --- WORKER-SPECIFIC GLOBALS ---
worker_gold_features_df: pd.DataFrame = None

def init_worker(gold_features_df: pd.DataFrame):
    """Initializer for each worker process in the Pool."""
    global worker_gold_features_df
    worker_gold_features_df = gold_features_df

# --- HELPER FUNCTIONS ---

def simplify_rule_string(rule_str: str) -> str:
    """
    Parses a raw Decision Tree rule string and removes redundant conditions.
    Example: "A <= 5 and A <= 3" becomes "A <= 3".
    """
    if not rule_str: return ""
    
    # Dictionary to store bounds: {feature: {'min': val, 'max': val}}
    conditions = rule_str.split(" and ")
    parsed = defaultdict(lambda: {'min': -float('inf'), 'max': float('inf')})
    
    # Regex to parse "`Feature` operator Value"
    pattern = re.compile(r"`(.+?)`\s*(<=|>)\s*([-\d\.]+)")
    
    for cond in conditions:
        match = pattern.search(cond)
        if match:
            feature, op, val_str = match.groups()
            val = float(val_str)
            
            if op == '<=':
                # We want the tightest (lowest) upper bound
                parsed[feature]['max'] = min(parsed[feature]['max'], val)
            elif op == '>':
                # We want the tightest (highest) lower bound
                parsed[feature]['min'] = max(parsed[feature]['min'], val)
                
    # Reconstruct the string
    cleaned_parts = []
    for feature, bounds in parsed.items():
        if bounds['min'] != -float('inf'):
            cleaned_parts.append(f"`{feature}` > {bounds['min']}")
        if bounds['max'] != float('inf'):
            cleaned_parts.append(f"`{feature}` <= {bounds['max']}")
            
    return " and ".join(sorted(cleaned_parts))

def _ensure_paths_exist(paths: Dict[str, str]):
    """Ensures all necessary output files and directories exist."""
    for key, path in paths.items():
        if key in ['strategies', 'blacklists', 'exhausted']:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                # FIX: Must provide empty lists for all columns in schema
                if key == 'blacklists':
                    schema = pa.schema([('key', pa.string()), ('market_rule', pa.string())])
                    data = {'key': [], 'market_rule': []}
                    pq.write_table(pa.Table.from_pydict(data, schema=schema), path)
                elif key == 'strategies':
                    # Using int64/float64 to match Pandas defaults
                    schema = pa.schema([('key', pa.string()), ('market_rule', pa.string()), 
                                      ('n_candles', pa.int64()), ('avg_density', pa.float64())])
                    data = {'key': [], 'market_rule': [], 'n_candles': [], 'avg_density': []}
                    pq.write_table(pa.Table.from_pydict(data, schema=schema), path)
                elif key == 'exhausted':
                    schema = pa.schema([('key', pa.string())])
                    data = {'key': []}
                    pq.write_table(pa.Table.from_pydict(data, schema=schema), path)
        elif key == 'processed_log':
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                open(path, 'w').close()

def _load_keys_from_file(filepath: str) -> Set[str]:
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return {line.strip() for line in f if line.strip()}
    except Exception: pass
    return set()

def _load_keys_from_parquet(filepath: str, column_name: str = 'key') -> Set[str]:
    try:
        if os.path.exists(filepath):
            pf = pq.ParquetFile(filepath)
            if pf.metadata.num_rows > 0:
                df = pd.read_parquet(filepath, columns=[column_name])
                return set(df[column_name].dropna().unique())
    except Exception: pass
    return set()

# --- ML CORE LOGIC ---

def get_rule_from_tree(tree, feature_names: List[str]) -> List[Dict]:
    """Recursively traverses a trained Decision Tree to extract raw rules."""
    rules = []
    def recurse(node_id: int, path: List[str]):
        if tree.feature[node_id] != -2: # Not a leaf
            name = feature_names[tree.feature[node_id]]
            threshold = round(tree.threshold[node_id], 5)
            recurse(tree.children_left[node_id], path + [f"`{name}` <= {threshold}"])
            recurse(tree.children_right[node_id], path + [f"`{name}` > {threshold}"])
        else: # Leaf
            # Only keep leaves that actually have samples
            if tree.n_node_samples[node_id] > 0:
                rules.append({
                    'rule_raw': " and ".join(path), # Store raw path
                    'n_candles': int(tree.n_node_samples[node_id]),
                    'avg_density': float(tree.value[node_id][0][0])
                })
    recurse(0, [])
    return rules

def find_rules_with_decision_tree(training_df: pd.DataFrame, exclusion_rules: Set[str]) -> Dict:
    """Trains a Decision Tree and extracts simplified, high-quality rules."""
    pruned_df = training_df.copy()
    baseline_avg_density = pruned_df['trade_count'].mean()
    
    if pd.isna(baseline_avg_density) or baseline_avg_density == 0:
        return {'status': 'exhausted'}

    # Feedback Loop: Zero out bad data points
    for rule_str in exclusion_rules:
        try:
            indices = pruned_df.query(rule_str).index
            if not indices.empty:
                pruned_df.loc[indices, 'trade_count'] = 0
        except Exception: pass

    X = pruned_df.drop(columns=['time', 'trade_count'])
    y = pruned_df['trade_count']
    
    if y.sum() == 0: return {'status': 'exhausted'}

    model = DecisionTreeRegressor(
        max_depth=c.PLATINUM_DT_MAX_DEPTH,
        min_samples_leaf=c.PLATINUM_MIN_CANDLES_PER_RULE,
        random_state=42
    )
    model.fit(X, y)
    
    all_rules = get_rule_from_tree(model.tree_, X.columns)
    
    valid_new_rules = []
    for rule_info in all_rules:
        # Check density lift
        if rule_info['avg_density'] >= (baseline_avg_density * c.PLATINUM_DENSITY_LIFT_THRESHOLD):
            
            # SIMPLIFY THE RULE STRING HERE
            simplified_rule = simplify_rule_string(rule_info['rule_raw'])
            
            if simplified_rule and simplified_rule not in exclusion_rules:
                rule_info['rule'] = simplified_rule # Update to clean rule
                valid_new_rules.append(rule_info)
    
    return {'status': 'success', 'rules': valid_new_rules} if valid_new_rules else {'status': 'exhausted'}

# --- PARALLEL WORKER ---
def process_key_batch(task_tuple: Tuple[List[Tuple[str, str]], Dict[str, Set[str]]]) -> Dict:
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
                    discovered_in_batch.append({
                        'key': key, 
                        'market_rule': rule_info['rule'], # Uses the Simplified Rule
                        'n_candles': rule_info['n_candles'], 
                        'avg_density': rule_info['avg_density']
                    })
            elif result['status'] == 'exhausted':
                exhausted_in_batch.append(key)
        except Exception:
            logger.error(f"Worker failed on {key}", exc_info=True)
            
    return {'strategies': discovered_in_batch, 'exhausted_keys': exhausted_in_batch}

# --- ORCHESTRATION ---

def _load_all_data_sources(paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Set]:
    logger.info("  - Loading Gold Features...")
    gold_df = pd.read_parquet(paths['gold'])
    gold_df['time'] = pd.to_datetime(gold_df['time']) 
    
    logger.info("  - Loading Combinations...")
    all_blueprints_df = pd.read_parquet(paths['combo'])
    
    logger.info("  - Loading History...")
    exclusion_rules = defaultdict(set)
    
    for fname in ['strategies', 'blacklists']:
        if os.path.exists(paths[fname]) and os.path.getsize(paths[fname]) > 0:
            try:
                # Check for metadata to avoid reading empty files
                pf = pq.ParquetFile(paths[fname])
                if pf.metadata.num_rows > 0:
                    df = pd.read_parquet(paths[fname], columns=['key', 'market_rule'])
                    for _, row in df.iterrows():
                        exclusion_rules[row['key']].add(row['market_rule'])
            except Exception: pass
            
    exhausted = _load_keys_from_parquet(paths['exhausted'])
    return gold_df, all_blueprints_df, exclusion_rules, exhausted

def execute_parallel_processing(tasks, gold_df, desc):
    if not tasks: return [], []
    strategies, exhausted = [], []
    with Pool(processes=c.MAX_CPU_USAGE, initializer=init_worker, initargs=(gold_df,)) as pool:
        for res in tqdm(pool.imap_unordered(process_key_batch, tasks), total=len(tasks), desc=desc):
            if res['strategies']: strategies.extend(res['strategies'])
            if res['exhausted_keys']: exhausted.extend(res['exhausted_keys'])
    return strategies, exhausted

def run_discovery_for_instrument(instrument_name: str, base_dirs: Dict[str, str]):
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
        logger.error(f"Input missing: {e}")
        return

    # --- PHASE 1: DISCOVERY ---
    logger.info("--- Phase 1: Discovery ---")
    processed_keys = _load_keys_from_file(paths['processed_log'])
    
    df_filt = all_blueprints_df[
        (all_blueprints_df['num_candles'] >= c.PLATINUM_MIN_CANDLE_LIMIT) &
        (~all_blueprints_df['key'].isin(exhausted_keys)) &
        (~all_blueprints_df['key'].isin(processed_keys))
    ]
    discovery_keys = df_filt['key'].tolist()
    logger.info(f"Blueprints to process: {len(discovery_keys)}")
    
    if discovery_keys:
        paths_list = [(k, os.path.join(paths['targets_dir'], f"{k}.parquet")) for k in discovery_keys]
        tasks = [(batch, exclusion_rules) for batch in np.array_split(paths_list, max(1, len(paths_list) // c.PLATINUM_DISCOVERY_BATCH_SIZE))]
        new_strats, new_exhausted = execute_parallel_processing(tasks, gold_df, "Discovery")
    else:
        new_strats, new_exhausted = [], []

    # --- PHASE 2: IMPROVEMENT ---
    logger.info("--- Phase 2: Improvement ---")
    feedback_keys = _load_keys_from_parquet(paths['blacklists'])
    df_imp = all_blueprints_df[
        (all_blueprints_df['key'].isin(feedback_keys)) &
        (all_blueprints_df['num_candles'] >= c.PLATINUM_MIN_CANDLE_LIMIT) &
        (~all_blueprints_df['key'].isin(exhausted_keys))
    ]
    improve_keys = df_imp['key'].tolist()
    logger.info(f"Blueprints to improve: {len(improve_keys)}")
    
    if improve_keys:
        paths_list = [(k, os.path.join(paths['targets_dir'], f"{k}.parquet")) for k in improve_keys]
        tasks = [(batch, exclusion_rules) for batch in np.array_split(paths_list, max(1, len(paths_list) // c.PLATINUM_DISCOVERY_BATCH_SIZE))]
        imp_strats, imp_exhausted = execute_parallel_processing(tasks, gold_df, "Improvement")
    else:
        imp_strats, imp_exhausted = [], []

    # --- SAVE RESULTS ---
    logger.info("Saving results...")
    
    # Save Strategies
    total_strats = new_strats + imp_strats
    if total_strats:
        new_df = pd.DataFrame(total_strats)
        if os.path.exists(paths['strategies']) and os.path.getsize(paths['strategies']) > 0:
            existing_strategies_df = pd.read_parquet(paths['strategies'])
            final_df = pd.concat([existing_strategies_df, new_df]).drop_duplicates(subset=['key', 'market_rule'], keep='last')
        else:
            final_df = new_df
        final_df.to_parquet(paths['strategies'], index=False)
        logger.info(f"Saved {len(total_strats)} strategies.")

    # Save Exhausted
    total_exhausted = set(new_exhausted + imp_exhausted)
    if total_exhausted:
        ex_df = pd.DataFrame(list(total_exhausted), columns=['key'])
        if os.path.exists(paths['exhausted']) and os.path.getsize(paths['exhausted']) > 0:
            old_df = pd.read_parquet(paths['exhausted'])
            final_df = pd.concat([old_df, ex_df]).drop_duplicates(subset=['key'])
        else:
            final_df = ex_df
        final_df.to_parquet(paths['exhausted'], index=False)
        logger.info(f"Marked {len(total_exhausted)} exhausted.")

    # Update Log
    if discovery_keys:
        with open(paths['processed_log'], 'a') as f:
            for k in discovery_keys: f.write(f"{k}\n")

def main():
    """Main execution function."""
    # Setup Logging using Config levels
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "platinum_discoverer_layer")
    p.ensure_directories()

    logger.info("--- Platinum Strategy Discoverer (V6.1) ---")
    start_time = time.time()
    base_dirs = {
        'gold': p.GOLD_FEATURES_DIR,
        'platinum_combo': p.PLATINUM_COMBINATIONS,
        'platinum_targets': p.PLATINUM_TARGETS,
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
        new_files = scan_new_files(p.PLATINUM_COMBINATIONS, p.PLATINUM_DATA_STRATEGIES_DIR)
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