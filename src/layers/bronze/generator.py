# bronze_data_generator.py (V20.1 - Stats Reporting)

"""
Bronze Layer: The Possibility Engine

This script generates the raw trade simulation data.
Features:
- Configurable Generation Mode: WINS_ONLY, BALANCED, or ALL.
- Strict SL Priority: Always checks if Stop Loss was hit before Take Profit.
- Dynamic Pip Sizing: Uses centralized config for instrument precision.
"""

import os
import re
import sys
import time
import logging
import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

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
    from src.utils.raw_data_loader import load_and_clean_raw_ohlc_csv # type: ignore
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- WORKER-SPECIFIC GLOBALS ---
worker_df: Optional[pd.DataFrame] = None
worker_config: Optional[Dict[str, Any]] = None
worker_spread_cost: Optional[float] = None
worker_max_lookforward: Optional[int] = None
worker_gen_mode: Optional[str] = None

def init_worker(df: pd.DataFrame, config_dict: Dict, spread_cost: float, max_lookforward: int, gen_mode: str) -> None:
    """Initializer for each worker process."""
    global worker_df, worker_config, worker_spread_cost, worker_max_lookforward, worker_gen_mode
    worker_df = df
    worker_config = config_dict
    worker_spread_cost = spread_cost
    worker_max_lookforward = max_lookforward
    worker_gen_mode = gen_mode

# --- NUMBA CORE LOGIC ---
@njit
def find_trades_numba(
    close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, timestamps: np.ndarray,
    sl_ratios: np.ndarray, tp_ratios: np.ndarray, max_lookforward: int, spread_cost: float,
    processing_limit: int, capture_losses: bool
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Simulates trades with STRICT SL priority.
    Returns: (winning_trades, losing_trades)
    """
    winning_trades = []
    losing_trades = []
    
    n_sl = len(sl_ratios)
    n_tp = len(tp_ratios)
    limit_prices = len(close_prices)

    for i in range(processing_limit):
        entry_price = close_prices[i]
        entry_time = timestamps[i]
        search_limit = min(i + 1 + max_lookforward, limit_prices)

        # --- BUY Simulation ---
        for sl_idx in range(n_sl):
            sl_r = sl_ratios[sl_idx]
            sl_price = entry_price * (1 - sl_r)
            
            for tp_idx in range(n_tp):
                tp_r = tp_ratios[tp_idx]
                tp_price = entry_price * (1 + tp_r)
                
                trade_result = 0 # 0: Pending, 1: Win, -1: Loss
                exit_time = 0
                
                for j in range(i + 1, search_limit):
                    # STRICT RULE: Check SL First
                    if low_prices[j] <= sl_price:
                        trade_result = -1
                        exit_time = timestamps[j]
                        break 
                    
                    # If SL not hit, Check TP
                    if high_prices[j] >= (tp_price + spread_cost):
                        trade_result = 1
                        exit_time = timestamps[j]
                        break
                
                if trade_result == 1:
                    winning_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, exit_time, 1))
                elif trade_result == -1 and capture_losses:
                    losing_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, exit_time, 0))

        # --- SELL Simulation ---
        for sl_idx in range(n_sl):
            sl_r = sl_ratios[sl_idx]
            sl_price = entry_price * (1 + sl_r)
            
            for tp_idx in range(n_tp):
                tp_r = tp_ratios[tp_idx]
                tp_price = entry_price * (1 - tp_r)
                
                trade_result = 0
                exit_time = 0
                
                for j in range(i + 1, search_limit):
                    # STRICT RULE: Check SL First
                    if high_prices[j] >= sl_price:
                        trade_result = -1
                        exit_time = timestamps[j]
                        break
                    
                    # Check TP
                    if low_prices[j] <= (tp_price - spread_cost):
                        trade_result = 1
                        exit_time = timestamps[j]
                        break
                
                if trade_result == 1:
                    winning_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, exit_time, 1))
                elif trade_result == -1 and capture_losses:
                    losing_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, exit_time, 0))

    return winning_trades, losing_trades

# --- WORKER & HELPER FUNCTIONS ---

def get_pip_size(instrument: str) -> float:
    """Determines pip size using config map."""
    pip_map = getattr(c, "PIP_SIZE_MAP", {})
    if instrument in pip_map: return pip_map[instrument]
    for key, val in pip_map.items():
        if key in instrument: return val
    return pip_map.get("DEFAULT", 0.0001)

def get_config_from_filename(filename: str) -> Tuple[Optional[Dict], Optional[float]]:
    """Parses filename and gets instrument config."""
    match = re.search(r"([A-Z0-9_]+?)(\d+)\.csv$", filename, re.IGNORECASE)
    if not match:
        logger.warning(f"Could not parse '{filename}'. Skipping.")
        return None, None

    instrument_raw, timeframe_num = match.group(1), match.group(2)
    instrument = instrument_raw.replace("_", "").upper()
    timeframe_key = f"{timeframe_num}m"

    if timeframe_key not in c.TIMEFRAME_PRESETS:
        logger.warning(f"No preset for '{timeframe_key}'.")
        return None, None

    config_preset = c.TIMEFRAME_PRESETS[timeframe_key]
    pip_size = get_pip_size(instrument)
    spread_in_pips = c.SIMULATION_SPREAD_PIPS.get(instrument, c.SIMULATION_SPREAD_PIPS.get("DEFAULT", 3.0))
    spread_cost = spread_in_pips * pip_size

    logger.info(f"Config: {instrument} ({timeframe_key}) | Pip: {pip_size} | Spread: {spread_in_pips} ({spread_cost:.5f})")
    return config_preset, spread_cost

def process_chunk_task(task_indices: Tuple[int, int]) -> List[Tuple]:
    """Worker function: Simulates trades based on Mode."""
    start_index, end_index = task_indices
    df_slice = worker_df.iloc[start_index:end_index]
    
    processing_limit = min(c.BRONZE_INPUT_CHUNK_SIZE, len(worker_df) - start_index)
    processing_limit = min(processing_limit, len(df_slice) - worker_max_lookforward)
    
    if processing_limit <= 0:
        return []

    # Prepare numpy arrays for Numba
    close = df_slice["close"].values
    high = df_slice["high"].values
    low = df_slice["low"].values
    timestamps = df_slice["time"].values.astype("datetime64[ns]").astype(np.int64)
    
    sl_ratios = worker_config["SL_RATIOS"]
    tp_ratios = worker_config["TP_RATIOS"]
    
    # Logic: Do we need to capture losses?
    capture_losses = (worker_gen_mode in ['BALANCED', 'ALL'])

    wins, losses = find_trades_numba(
        close, high, low, timestamps, 
        sl_ratios, tp_ratios,
        worker_max_lookforward, worker_spread_cost, processing_limit, capture_losses
    )
    
    if worker_gen_mode == 'WINS_ONLY':
        return wins
    elif worker_gen_mode == 'ALL':
        wins.extend(losses)
        return wins
    elif worker_gen_mode == 'BALANCED':
        if losses:
            num_wins = len(wins)
            if num_wins > 0:
                count_to_sample = min(len(losses), num_wins)
                sampled_losses = random.sample(losses, count_to_sample)
                wins.extend(sampled_losses)
        return wins
        
    return wins

def _create_df_from_results(data: List[Tuple]) -> pd.DataFrame:
    """Converts trade tuples into DataFrame."""
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "entry_time", "trade_type", "entry_price", "sl_price", "tp_price",
        "sl_ratio", "tp_ratio", "exit_time", "outcome"
    ])

    if df.empty:
        return df

    # Convert timestamps back to datetime
    df['entry_time'] = pd.to_datetime(df['entry_time'], unit='ns')
    df['exit_time'] = pd.to_datetime(df['exit_time'], unit='ns')
    
    # Map trade_type: 1 -> 'buy', -1 -> 'sell'
    df['trade_type'] = np.where(df['trade_type'] == 1, 'buy', 'sell')
    df['trade_type'] = df['trade_type'].astype('category')
    
    # Outcome: 1 -> 'win', 0 -> 'loss'
    df['outcome'] = np.where(df['outcome'] == 1, 'win', 'loss')
    df['outcome'] = df['outcome'].astype('category')
    
    # Optimize numeric columns to float32 to save RAM (Parquet is efficient with this)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    return df

# --- MAIN ORCHESTRATOR ---

def process_file_pipelined(
    input_file: str, output_file: str, config_dict: Dict, spread_cost: float
) -> str:
    filename = os.path.basename(input_file)
    logger.info(f"Starting simulation for {filename}...")

    # --- 1. Load and Clean Raw Data ---
    try:
        df = load_and_clean_raw_ohlc_csv(input_file)
    except Exception as e:
        return f"ERROR: Failed to load '{filename}': {e}"
    
    # --- 2. Validation ---
    max_lookforward = config_dict["MAX_LOOKFORWARD"]
    if len(df) <= max_lookforward:
        return f"ERROR: Not enough data ({len(df)} rows)."

    if os.path.exists(output_file):
        logger.warning(f"Output file {os.path.basename(output_file)} exists. Overwriting.")
        os.remove(output_file)

    tasks = [(i, i + c.BRONZE_INPUT_CHUNK_SIZE + max_lookforward) for i in range(0, len(df), c.BRONZE_INPUT_CHUNK_SIZE)]
    
    accumulator = []
    total_wins = 0
    total_losses = 0
    writer = None
    gen_mode = getattr(c, 'BRONZE_GENERATION_MODE', 'WINS_ONLY')

    # --- 4. Execution ---
    try:
        pool_init_args = (df, config_dict, spread_cost, max_lookforward, gen_mode)
        
        with Pool(processes=c.MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
            results_iter = pool.imap(process_chunk_task, tasks)
            pbar = tqdm(results_iter, total=len(tasks), desc=f"Scanning {filename} [{gen_mode}]", unit="chunk")
            
            for res in pbar:
                if res: 
                    accumulator.extend(res)
                
                # Memory Management: Flush to disk if buffer fills up
                if len(accumulator) >= c.BRONZE_OUTPUT_CHUNK_SIZE:
                    chunk_df = _create_df_from_results(accumulator)
                    if not chunk_df.empty:
                        # Count stats before writing
                        counts = chunk_df['outcome'].value_counts()
                        total_wins += counts.get('win', 0)
                        total_losses += counts.get('loss', 0)
                        
                        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(output_file, table.schema)
                        writer.write_table(table)
                    accumulator.clear()
        
        if accumulator:
            chunk_df = _create_df_from_results(accumulator)
            if not chunk_df.empty:
                counts = chunk_df['outcome'].value_counts()
                total_wins += counts.get('win', 0)
                total_losses += counts.get('loss', 0)
                
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema)
                writer.write_table(table)

    except Exception as e:
        logger.error(f"Processing error: {e}")
        return f"ERROR: {e}"
    finally:
        if writer:
            writer.close()

    total_trades = total_wins + total_losses
    if total_trades == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return "No trades found matching criteria."
    
    return f"SUCCESS: Generated {total_trades:,} trades (W: {total_wins:,} | L: {total_losses:,})."

def main() -> None:
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "bronze_layer")
    p.ensure_directories()

    start_time = time.time()
    logger.info("--- Bronze Layer: The Possibility Engine (V20.1 - Stats) ---")
    
    gen_mode = getattr(c, 'BRONZE_GENERATION_MODE', 'WINS_ONLY')
    logger.info(f"Generation Mode: {gen_mode}")

    # Warmup
    find_trades_numba(np.random.rand(10), np.random.rand(10), np.random.rand(10), 
                      np.arange(10, dtype=np.int64), np.array([0.01]), np.array([0.02]), 1, 0.0001, 5, True)

    files_to_process = []
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_arg:
        path = os.path.join(p.RAW_DATA_DIR, f"{target_arg}.csv")
        if os.path.exists(path):
            files_to_process = [f"{target_arg}.csv"]
            logger.info(f"Targeted Mode: Processing '{target_arg}'")
        else:
            logger.error(f"Target file not found: {path}")
    else:
        # Standard Mode: Scan for new files
        new_files = scan_new_files(p.RAW_DATA_DIR, p.BRONZE_DATA_DIR)
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    # Processing Loop
    logger.info(f"Processing {len(files_to_process)} file(s)...")
    for fname in files_to_process:
        input_path = os.path.join(p.RAW_DATA_DIR, fname)
        output_path = os.path.join(p.BRONZE_DATA_DIR, fname.replace('.csv', '.parquet'))
        
        # 1. Get Dynamic Config for this specific file/timeframe
        config_dict, spread_cost = get_config_from_filename(fname)
        if config_dict:
            res = process_file_pipelined(input_path, output_path, config_dict, spread_cost)
            logger.info(f"RESULT [{fname}]: {res}")
        else:
            logger.error(f"Skipping {fname}: Configuration mismatch (Instrument/Timeframe not in config.py).")

    end_time = time.time()
    logger.info(f"Bronze layer generation complete. Total Runtime: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()