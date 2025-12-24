# src/layers/diamond/trainer.py (V2.1 - Quantile Fix)

"""
Diamond Layer: The Model Trainer

This script acts as the 'Strategy Factory'. It consumes the massive, unified 
Platinum Dataset (Context + Trade Params + Outcome) and trains a gradient 
boosting model to predict the probability of a win.

Updates in V2.1:
- Fixed ValueError by passing 'ref=dtrain' to validation QuantileDMatrix.
- Optimized logging to show progress during DMatrix construction.
"""

import os
import sys
import gc
import re
import time
import logging
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    import config.config as c
    from src.utils import paths as p
    from src.utils.logger import setup_logging 
    from src.utils.file_selector import scan_new_files, select_files_interactively
except ImportError as e:
    print(f"CRITICAL: Import failed: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- MEMORY OPTIMIZED ITERATOR ---

class ParquetBatchIterator(xgb.DataIter):
    """
    A custom iterator that reads Parquet shards one by one.
    This prevents loading the entire 55M+ row dataset into RAM.
    """
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        self._it = 0
        super().__init__()

    def next(self, input_data):
        """Called by XGBoost to get the next batch of data."""
        if self._it == len(self.file_paths):
            return 0 # Stop iteration
        
        # Load one shard
        try:
            path = self.file_paths[self._it]
            df = pd.read_parquet(path)
            
            # Separate Target and Features
            if 'target' not in df.columns:
                self._it += 1
                return 1 
                
            y = df['target']
            
            # Drop non-feature columns
            drop_cols = ['target', 'entry_time', 'exit_time']
            X = df.drop(columns=[col for col in drop_cols if col in df.columns])
            
            # Pass to XGBoost
            input_data(data=X, label=y)
            
            # Cleanup
            del df, X, y
            gc.collect()
            
            self._it += 1
            return 1 # Continue iteration
            
        except Exception as e:
            logger.error(f"Error reading shard {self.file_paths[self._it]}: {e}")
            return 0 # Stop on error

    def reset(self):
        """Resets the iterator for the next training round."""
        self._it = 0

# --- CORE LOGIC ---

def get_sorted_shards(instrument_name: str) -> tuple:
    """
    Finds and sorts Parquet shards chronologically.
    Returns (train_files, val_files).
    """
    dataset_dir = p.PLATINUM_TARGETS / instrument_name
    
    if not dataset_dir.exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        return [], []

    files = list(dataset_dir.glob("*.parquet"))
    if not files:
        logger.error("No data shards found.")
        return [], []

    # Sort by integer index
    def extract_index(f_path):
        match = re.search(r'part_(\d+)', f_path.name)
        return int(match.group(1)) if match else -1
        
    sorted_files = sorted(files, key=extract_index)
    
    # Time-Series Split (80/20 by file count)
    split_idx = int(len(sorted_files) * (1 - c.DIAMOND_TEST_SIZE))
    # Ensure at least one val file
    split_idx = min(split_idx, len(sorted_files) - 1)
    
    train_files = sorted_files[:split_idx]
    val_files = sorted_files[split_idx:]
    
    logger.info(f"  - Found {len(sorted_files)} shards.")
    logger.info(f"  - Split: {len(train_files)} Train files | {len(val_files)} Val files.")
    
    return train_files, val_files

def train_xgb_model(instrument_name: str):
    """Orchestrates training using Iterative Loading."""
    
    # 1. Prepare File Lists
    train_files, val_files = get_sorted_shards(instrument_name)
    if not train_files: return

    # 2. Initialize Iterators
    logger.info("  - Initializing Data Iterators (Low RAM Mode)...")
    train_iter = ParquetBatchIterator(train_files)
    val_iter = ParquetBatchIterator(val_files)

    # 3. Create QuantileDMatrix
    # QuantileDMatrix reads data immediately to create histogram sketches.
    logger.info("  - Building Training Matrix (Reading Data)...")
    dtrain = xgb.QuantileDMatrix(train_iter)
    
    logger.info("  - Building Validation Matrix (Aligning with Train)...")
    # CRITICAL FIX: Must pass ref=dtrain to ensure cuts match!
    dval = xgb.QuantileDMatrix(val_iter, ref=dtrain)
    
    # Get feature names for importance plot
    sample_df = pd.read_parquet(train_files[0])
    drop_cols = ['target', 'entry_time', 'exit_time']
    feature_names = [col for col in sample_df.columns if col not in drop_cols]
    dtrain.feature_names = feature_names
    dval.feature_names = feature_names
    del sample_df

    # 4. Train
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

    # 5. Evaluation
    logger.info("  - Evaluating Performance...")
    
    y_true = []
    y_pred_prob = []
    
    # Predict in batches to save RAM
    # Note: We re-read raw files here because DMatrix doesn't expose labels easily for massive datasets
    for f in tqdm(val_files, desc="Running Validation"):
        df = pd.read_parquet(f)
        if 'target' not in df.columns: continue
        
        y_chunk = df['target'].values
        X_chunk = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Use simple DMatrix for inference batch
        d_batch = xgb.DMatrix(X_chunk)
        preds_chunk = model.predict(d_batch)
        
        y_true.extend(y_chunk)
        y_pred_prob.extend(preds_chunk)
        
        del df, X_chunk, d_batch
    
    # Calculate Metrics
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

    # 6. Save Artifacts
    if not p.DIAMOND_STRATEGIES.exists():
        p.DIAMOND_STRATEGIES.mkdir(parents=True, exist_ok=True)

    strategy_path = p.DIAMOND_STRATEGIES / f"{instrument_name}_xgb.json"
    model.save_model(str(strategy_path))
    logger.info(f"  - Model saved to: {strategy_path}")
    
    # Feature Importance
    importance_path = p.DIAMOND_STRATEGIES / f"{instrument_name}_importance.csv"
    importance = model.get_score(importance_type='gain')
    imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain']).sort_values('Gain', ascending=False)
    imp_df.to_csv(importance_path, index=False)
    logger.info(f"  - Importance saved to: {importance_path}")

def main():
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "diamond_trainer")
    p.ensure_directories()
    
    start_time = time.time()
    logger.info("--- Diamond Layer: Strategy Trainer (V2.1 - Quantile Fix) ---")

    files_to_process = []
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_arg:
        path = os.path.join(p.PLATINUM_TARGETS, target_arg)
        if os.path.exists(path):
            files_to_process = [target_arg]
            logger.info(f"Targeted Mode: Processing '{target_arg}'")
        else:
            logger.error(f"Target file not found: {path}")
    else:
        # Standard Mode: Scan for new files
        new_files = scan_new_files(p.PLATINUM_TARGETS, p.BRONZE_DATA_DIR)
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    logger.info(f"Processing {len(files_to_process)} file(s)...")
    for instrument in files_to_process:
        logger.info(f"--- Starting Training for {instrument} ---")
        train_xgb_model(instrument)
    
    end_time = time.time()
    logger.info(f"Diamond layer training complete. Total Runtime: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()