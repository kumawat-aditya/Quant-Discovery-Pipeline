# src/layers/diamond/trainer.py (V2.2 - Hybrid Memory Mode)

"""
Diamond Layer: The Model Trainer

This script acts as the 'Strategy Factory'. It consumes the massive, unified 
Platinum Dataset and trains an XGBoost model.

Updates in V2.2:
- Added support for `DIAMOND_LOAD_FULL_DATASET_IN_MEMORY` config.
- If True: Loads full dataset into RAM (Fastest for Servers/GPUs).
- If False: Uses BatchIterator (Memory Safe for Laptops).
"""

import os
import sys
import gc
import re
import time
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
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

# --- 1. MEMORY SAFE ITERATOR (Low RAM Mode) ---

class ParquetBatchIterator(xgb.DataIter):
    """Custom iterator for reading Parquet shards one by one."""
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self._it = 0
        super().__init__()

    def next(self, input_data):
        if self._it == len(self.file_paths): return 0
        try:
            path = self.file_paths[self._it]
            df = pd.read_parquet(path)
            if 'target' not in df.columns:
                self._it += 1; return 1 
            y = df['target']
            drop_cols = ['target', 'entry_time', 'exit_time']
            X = df.drop(columns=[col for col in drop_cols if col in df.columns])
            input_data(data=X, label=y)
            del df, X, y; gc.collect()
            self._it += 1
            return 1
        except Exception as e:
            logger.error(f"Error reading shard {self.file_paths[self._it]}: {e}")
            return 0

    def reset(self): self._it = 0

# --- 2. FULL MEMORY LOADER (Server/GPU Mode) ---

def load_full_dataset_in_memory(train_files: list, val_files: list) -> tuple:
    """
    Loads ALL data into RAM. much faster for training, but requires huge RAM.
    """
    logger.info(f"  - [High Performance Mode] Loading {len(train_files) + len(val_files)} shards into RAM...")
    
    # Load Train
    train_dfs = [pd.read_parquet(f) for f in tqdm(train_files, desc="Loading Train RAM")]
    df_train = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    
    # Load Val
    val_dfs = [pd.read_parquet(f) for f in tqdm(val_files, desc="Loading Val RAM")]
    df_val = pd.concat(val_dfs, ignore_index=True)
    del val_dfs
    
    # Prepare X, y
    drop_cols = ['target', 'entry_time', 'exit_time']
    
    y_train = df_train['target']
    X_train = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns])
    
    y_val = df_val['target']
    X_val = df_val.drop(columns=[col for col in drop_cols if col in df_val.columns])
    
    logger.info(f"  - Loaded. Train Shape: {X_train.shape} | Val Shape: {X_val.shape}")
    
    # Create Standard DMatrices (Faster for GPU than QuantileDMatrix iterators)
    logger.info("  - Constructing DMatrices in Memory...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Clean up raw pandas frames immediately
    del df_train, df_val, X_train, X_val, y_train, y_val
    gc.collect()
    
    return dtrain, dval

# --- 3. SHARED UTILS ---

def get_sorted_shards(instrument_name: str) -> tuple:
    """Finds and sorts Parquet shards chronologically."""
    dataset_dir = p.PLATINUM_TARGETS / instrument_name
    if not dataset_dir.exists(): return [], []
    files = list(dataset_dir.glob("*.parquet"))
    if not files: return [], []

    def extract_index(f_path):
        match = re.search(r'part_(\d+)', f_path.name)
        return int(match.group(1)) if match else -1
        
    sorted_files = sorted(files, key=extract_index)
    split_idx = int(len(sorted_files) * (1 - c.DIAMOND_TEST_SIZE))
    split_idx = min(split_idx, len(sorted_files) - 1)
    
    return sorted_files[:split_idx], sorted_files[split_idx:]

def evaluate_model(model, val_files, dval=None):
    """Calculates metrics. Handles both Iterator and In-Memory cases."""
    logger.info("  - Evaluating Performance...")
    
    y_true = []
    y_pred_prob = []
    
    if c.DIAMOND_LOAD_FULL_DATASET_IN_MEMORY and dval is not None:
        # Fast Path: In-Memory
        y_true = dval.get_label()
        y_pred_prob = model.predict(dval)
    else:
        # Slow Path: Batch Load for evaluation
        drop_cols = ['target', 'entry_time', 'exit_time']
        for f in tqdm(val_files, desc="Running Validation"):
            df = pd.read_parquet(f)
            if 'target' not in df.columns: continue
            y_chunk = df['target'].values
            X_chunk = df.drop(columns=[col for col in drop_cols if col in df.columns])
            d_batch = xgb.DMatrix(X_chunk)
            preds_chunk = model.predict(d_batch)
            y_true.extend(y_chunk)
            y_pred_prob.extend(preds_chunk)
            del df, X_chunk, d_batch
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_pred_prob]
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred_bin),
        "Precision": precision_score(y_true, y_pred_bin),
        "Recall": recall_score(y_true, y_pred_bin),
        "AUC": roc_auc_score(y_true, y_pred_prob)
    }
    
    logger.info("--- Model Metrics (Validation Set) ---")
    for k, v in metrics.items():
        logger.info(f"  {k:<10}: {v:.4f}")

def train_xgb_model(instrument_name: str):
    train_files, val_files = get_sorted_shards(instrument_name)
    if not train_files: return

    logger.info(f"  - Found {len(train_files) + len(val_files)} shards.")
    
    # --- BRANCHING LOGIC ---
    if c.DIAMOND_LOAD_FULL_DATASET_IN_MEMORY:
        dtrain, dval = load_full_dataset_in_memory(train_files, val_files)
    else:
        logger.info("  - Initializing Iterators (Memory Safe Mode)...")
        train_iter = ParquetBatchIterator(train_files)
        val_iter = ParquetBatchIterator(val_files)
        logger.info("  - Building Quantile Matrices...")
        dtrain = xgb.QuantileDMatrix(train_iter)
        dval = xgb.QuantileDMatrix(val_iter, ref=dtrain)
        
        # Metadata needed for feature names
        sample = pd.read_parquet(train_files[0])
        cols = [c for c in sample.columns if c not in ['target', 'entry_time', 'exit_time']]
        dtrain.feature_names = cols
        dval.feature_names = cols
        del sample

    # --- TRAINING ---
    logger.info(f"  - Starting Training ({c.DIAMOND_BOOST_ROUNDS} rounds)...")
    start_ts = time.time()
    
    params = c.DIAMOND_XGB_PARAMS.copy()
    params['nthread'] = c.MAX_CPU_USAGE
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=c.DIAMOND_BOOST_ROUNDS,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=c.DIAMOND_EARLY_STOPPING,
        verbose_eval=50
    )
    
    logger.info(f"  - Training complete: {time.time() - start_ts:.2f}s")

    # --- EVALUATION ---
    # Pass dval only if in-memory, otherwise pass file list for batch processing
    dval_arg = dval if c.DIAMOND_LOAD_FULL_DATASET_IN_MEMORY else None
    evaluate_model(model, val_files, dval_arg)

    # --- SAVE ARTIFACTS ---
    if not p.DIAMOND_STRATEGIES.exists(): p.DIAMOND_STRATEGIES.mkdir(parents=True, exist_ok=True)

    strategy_path = p.DIAMOND_STRATEGIES / f"{instrument_name}_xgb.json"
    model.save_model(str(strategy_path))
    logger.info(f"  - Model saved to: {strategy_path}")
    
    importance_path = p.DIAMOND_STRATEGIES / f"{instrument_name}_importance.csv"
    importance = model.get_score(importance_type='gain')
    imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain']).sort_values('Gain', ascending=False)
    imp_df.to_csv(importance_path, index=False)
    logger.info(f"  - Importance saved to: {importance_path}")

def main():
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "diamond_trainer")
    p.ensure_directories()
    
    start_time = time.time()
    logger.info("--- Diamond Layer: Strategy Trainer (V2.2 - Hybrid) ---")
    
    mode = "FULL MEMORY (SERVER)" if c.DIAMOND_LOAD_FULL_DATASET_IN_MEMORY else "ITERATIVE (LOW RAM)"
    logger.info(f"Loading Mode: {mode}")
    
    if not p.PLATINUM_TARGETS.exists():
        logger.error("Platinum targets directory does not exist. Run the Platinum layer first.")
        return

    instruments = sorted([d.name for d in p.PLATINUM_TARGETS.iterdir() if d.is_dir()])
    if not instruments:
        logger.info("No datasets found in Platinum targets.")
        return

    target_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_arg:
        # Non-interactive: orchestrator passed the instrument name directly
        if target_arg not in instruments:
            logger.error(f"Instrument '{target_arg}' not found in {p.PLATINUM_TARGETS}.")
            return
        logger.info(f"Targeted Mode: Training '{target_arg}'")
        selected = [target_arg]
    else:
        # Interactive selection
        print("\nAvailable Instruments:")
        for i, inst in enumerate(instruments):
            print(f"[{i+1}] {inst}")
        choice = input("\nSelect instrument (e.g., 1) or 'a': ").strip().lower()
        selected = instruments if choice == 'a' else [instruments[int(choice)-1]] if choice.isdigit() else []

    for instr in selected:
        logger.info(f"--- Starting Training for {instr} ---")
        train_xgb_model(instr)

    end_time = time.time()
    logger.info(f"Diamond layer training complete. Total Runtime: {end_time - start_time:.2f}s")
    
if __name__ == "__main__":
    main()