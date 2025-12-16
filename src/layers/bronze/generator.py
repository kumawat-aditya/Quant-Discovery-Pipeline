# bronze_data_generator.py

"""
Bronze Layer: The Possibility Engine

This script serves as the foundational data generation layer for the quantitative
trading strategy discovery pipeline. Its primary purpose is to systematically
scan historical price data and generate a comprehensive dataset of every
conceivable winning trade based on a predefined grid of Stop-Loss (SL) and
Take-Profit (TP) ratios defined in the central configuration.

Architectural Highlights:
- Parquet Output: Saves data in the highly efficient, columnar Parquet format.
- Numba JIT Compilation: Core simulation logic is hardware-accelerated.
- Producer-Consumer Model: Uses multiprocessing to simulate trades in parallel.
- Dynamic Configuration: Adapts to instrument-specific spreads and timeframe-specific
  SL/TP grids defined in config.py.
"""

import os
import re
import sys
import time
import logging
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
    logging.critical("'pyarrow' library not found. Please run 'pip install pyarrow' to continue.")
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


# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
# These are initialized in each worker process to avoid serialization overhead
worker_df: Optional[pd.DataFrame] = None
worker_config: Optional[Dict[str, Any]] = None
worker_spread_cost: Optional[float] = None
worker_max_lookforward: Optional[int] = None


def init_worker(df: pd.DataFrame, config_dict: Dict, spread_cost: float, max_lookforward: int) -> None:
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_df, worker_config, worker_spread_cost, worker_max_lookforward
    worker_df = df
    worker_config = config_dict
    worker_spread_cost = spread_cost
    worker_max_lookforward = max_lookforward


# --- NUMBA-ACCELERATED CORE LOGIC (UNCHANGED) ---
@njit
def find_winning_trades_numba(
    close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, timestamps: np.ndarray,
    sl_ratios: np.ndarray, tp_ratios: np.ndarray, max_lookforward: int, spread_cost: float,
    processing_limit: int
) -> List[Tuple]:
    """Executes the core trade simulation logic at high speed using Numba."""
    # This core logic is identical to the previous version.
    all_profitable_trades = []
    for i in range(processing_limit):
        entry_price = close_prices[i]
        entry_time = timestamps[i]
        # --- Simulate BUY trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 - sl_r)
                tp_price = entry_price * (1 + tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    if high_prices[j] >= (tp_price + spread_cost):
                        all_profitable_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    if low_prices[j] <= sl_price:
                        break
        # --- Simulate SELL trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 + sl_r)
                tp_price = entry_price * (1 - tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    if low_prices[j] <= (tp_price - spread_cost):
                        all_profitable_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    if high_prices[j] >= sl_price:
                        break
    return all_profitable_trades


# --- HELPER & WORKER FUNCTIONS ---

def get_config_from_filename(filename: str) -> Tuple[Optional[Dict], Optional[float]]:
    """
    Parses a filename to extract instrument and timeframe, returning the 
    appropriate simulation configuration and spread cost.
    """
    # Expected format: EURUSD60.csv or similar
    match = re.search(r"([A-Z0-9_]+?)(\d+)\.csv$", filename, re.IGNORECASE)
    if not match:
        logger.warning(f"Could not parse timeframe or instrument from '{filename}'. Skipping.")
        return None, None

    instrument_raw, timeframe_num = match.group(1), match.group(2)
    instrument = instrument_raw.replace("_", "").upper()
    timeframe_key = f"{timeframe_num}m"

    # Validate against presets in config.py
    if timeframe_key not in c.TIMEFRAME_PRESETS:
        logger.warning(f"No preset found in config for timeframe '{timeframe_key}' (File: {filename}). Skipping.")
        return None, None

    config_preset = c.TIMEFRAME_PRESETS[timeframe_key]
    
    # Calculate Spread Cost
    # JPY and Gold (XAU) usually have 2 decimal places (pip = 0.01), others 4 (pip = 0.0001)
    pip_size = 0.01 if "JPY" in instrument or "XAU" in instrument else 0.0001
    # TODO should make it dynamic from config file.
    
    # Retrieve spread in pips from config, defaulting if instrument not found
    spread_in_pips = c.SIMULATION_SPREAD_PIPS.get(instrument, c.SIMULATION_SPREAD_PIPS.get("DEFAULT", 3.0))
    spread_cost = spread_in_pips * pip_size

    logger.info(f"Config detected: {instrument} ({timeframe_key})")
    logger.info(f"   -> Spread: {spread_in_pips} pips ({spread_cost:.5f}) | Max Lookforward: {config_preset['MAX_LOOKFORWARD']}")
    
    return config_preset, spread_cost


def process_chunk_task(task_indices: Tuple[int, int]) -> List[Tuple]:
    """
    The "Producer" worker function, executed in parallel by the Pool.
    Extracts numpy arrays from global worker variables and calls Numba logic.
    """
    start_index, end_index = task_indices
    
    # Slice the dataframe
    df_slice = worker_df.iloc[start_index:end_index]
    
    # Determine valid processing range (cannot simulate beyond data end)
    processing_limit = min(c.BRONZE_INPUT_CHUNK_SIZE, len(worker_df) - start_index)
    processing_limit = min(processing_limit, len(df_slice) - worker_max_lookforward)
    
    if processing_limit <= 0:
        return []

    # Prepare numpy arrays for Numba
    close = df_slice["close"].values
    high = df_slice["high"].values
    low = df_slice["low"].values
    timestamps = df_slice["time"].values.astype("datetime64[ns]").astype(np.int64)
    
    # Ensure SL/TP ratios are numpy arrays (as defined in config.py)
    # The config now uses np.arange, so these are already arrays.
    sl_ratios = worker_config["SL_RATIOS"]
    tp_ratios = worker_config["TP_RATIOS"]

    return find_winning_trades_numba(
        close, high, low, timestamps, 
        sl_ratios, tp_ratios,
        worker_max_lookforward, worker_spread_cost, processing_limit
    )


def _create_df_from_results(data: List[Tuple]) -> pd.DataFrame:
    """
    Converts a list of raw trade tuples into a structured, optimized DataFrame.
    Applies categorical types and cleans up columns.
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "entry_time", "trade_type", "entry_price", "sl_price", "tp_price",
        "sl_ratio", "tp_ratio", "exit_time"
    ])

    if df.empty:
        return df

    # Convert timestamps back to datetime
    df['entry_time'] = pd.to_datetime(df['entry_time'], unit='ns')
    df['exit_time'] = pd.to_datetime(df['exit_time'], unit='ns')
    
    # Map trade_type 1/-1 to readable string and categorize
    df['trade_type'] = np.where(df['trade_type'] == 1, 'buy', 'sell')
    df['trade_type'] = df['trade_type'].astype('category')
    
    # Set outcome to 'win' (since this script only finds winning trades)
    df['outcome'] = 'win'
    df['outcome'] = df['outcome'].astype('category')
    
    # Optimize numeric columns to float32 to save RAM (Parquet is efficient with this)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
        
    return df


# --- MAIN PROCESSING ORCHESTRATOR ---

def process_file_pipelined(
    input_file: str, output_file: str, config_dict: Dict, spread_cost: float
) -> str:
    """Orchestrates data generation for a single file, writing to Parquet."""
    filename = os.path.basename(input_file)
    logger.info(f"Starting simulation for {filename}...")

    # --- 1. Load and Clean Raw Data ---
    try:
        # Load using columns defined in config
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        
        # Validate column count
        required_cols = len(c.RAW_DATA_COLUMNS)
        if df.shape[1] < required_cols:
            # Fallback: Try to use as many as available if it matches minimal set
            if df.shape[1] >= 5: 
                 df.columns = c.RAW_DATA_COLUMNS[:df.shape[1]]
            else:
                return f"ERROR: File has {df.shape[1]} columns, expected at least 5 (OHLCV)."
        else:
            df.columns = c.RAW_DATA_COLUMNS[:df.shape[1]]

        # Standardize Time
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        # If timezone information exists, remove it.
        if df['time'].dt.tz is not None:
            logger.debug(f"Timezone '{df['time'].dt.tz}' detected. Localizing to None (UTC-naive).")
            df['time'] = df['time'].dt.tz_localize(None)
            
        numeric_cols = ["open", "high", "low", "close"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Drop invalid rows
        initial_rows = len(df)
        df.dropna(subset=['time'] + numeric_cols, inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows with invalid data.")
        
        df = df.sort_values('time').reset_index(drop=True)

    except Exception as e:
        return f"ERROR: Failed to load '{filename}': {e}"
    
    # --- 2. Validation ---
    max_lookforward = config_dict["MAX_LOOKFORWARD"]
    if len(df) <= max_lookforward:
        return f"ERROR: Not enough data ({len(df)} rows) for lookforward of {max_lookforward}."

    if os.path.exists(output_file):
        logger.warning(f"Output file {os.path.basename(output_file)} exists. Overwriting.")
        os.remove(output_file)

    # --- 3. Parallel Processing Setup ---
    # Create chunks indices
    tasks = [(i, i + c.BRONZE_INPUT_CHUNK_SIZE + max_lookforward) for i in range(0, len(df), c.BRONZE_INPUT_CHUNK_SIZE)]
    if not tasks: return "INFO: No processable chunks found."

    profitable_trades_accumulator = []
    total_trades_found = 0
    writer = None

    # --- 4. Execution ---
    try:
        # Initialize workers with read-only copy of data and config
        pool_init_args = (df, config_dict, spread_cost, max_lookforward)
        
        # Use updated CPU usage from config
        with Pool(processes=c.MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
            
            results_iterator = pool.imap(process_chunk_task, tasks)
            
            # Progress bar
            progress_bar = tqdm(results_iterator, total=len(tasks), desc=f"Scanning {filename}", unit="chunk")
            
            for results_list in progress_bar:
                if results_list:
                    profitable_trades_accumulator.extend(results_list)
                
                # Memory Management: Flush to disk if buffer fills up
                if len(profitable_trades_accumulator) >= c.BRONZE_OUTPUT_CHUNK_SIZE:
                    chunk_df = _create_df_from_results(profitable_trades_accumulator)
                    if not chunk_df.empty:
                        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(output_file, table.schema)
                        writer.write_table(table)
                        total_trades_found += len(chunk_df)
                    profitable_trades_accumulator.clear()
        
        # --- 5. Final Flush ---
        if profitable_trades_accumulator:
            final_chunk_df = _create_df_from_results(profitable_trades_accumulator)
            if not final_chunk_df.empty:
                table = pa.Table.from_pandas(final_chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema)
                writer.write_table(table)
                total_trades_found += len(final_chunk_df)

    except Exception as e:
        logger.error(f"Processing interrupted: {e}")
        return f"ERROR: Exception during processing: {e}"
    finally:
        if writer:
            writer.close()

    if total_trades_found == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return f"No winning trades found. Checked {len(df)} candles against config settings."
    
    return f"SUCCESS: Generated {total_trades_found:,} possibilities. Saved to {os.path.basename(output_file)}."


def main() -> None:
    """Main execution function."""
    # Setup Logging using Config levels
    setup_logging(p.LOGS_DIR, c.CONSOLE_LOG_LEVEL, c.FILE_LOG_LEVEL, "bronze_layer")

    start_time = time.time()
    logger.info("--- Bronze Layer: The Possibility Engine (Updated V17) ---")
    logger.info(f"Configuration loaded. CPU Cores allowed: {c.MAX_CPU_USAGE}")
    
    # JIT Warmup (Compiles the Numba function with dummy data so it's ready for the real work)
    logger.info("Warming up simulation engine...")
    dummy_prices = np.random.rand(10)
    dummy_time = np.arange(10, dtype=np.int64)
    dummy_sl = np.array([0.01])
    dummy_tp = np.array([0.02])
    find_winning_trades_numba(dummy_prices, dummy_prices, dummy_prices, dummy_time, dummy_sl, dummy_tp, 1, 0.0001, 5)
    logger.info("Engine ready.")

    # File Selection Logic
    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # User passed a specific file
        target_path = os.path.join(p.RAW_DATA_DIR, target_file_arg)
        if os.path.exists(target_path):
            files_to_process = [target_file_arg]
            logger.info(f"Targeted Mode: {target_file_arg}")
        else:
            logger.error(f"Target file not found: {target_path}")
    else:
        # Standard Mode: Scan for new files
        new_files = scan_new_files(p.RAW_DATA_DIR, p.BRONZE_DATA_DIR, ".csv")
        files_to_process = select_files_interactively(new_files)

    if not files_to_process:
        logger.info("No files selected. Exiting.")
        return

    # Processing Loop
    logger.info(f"Processing {len(files_to_process)} file(s)...")
    for filename in files_to_process:
        input_path = os.path.join(p.RAW_DATA_DIR, filename)
        output_path = os.path.join(p.BRONZE_DATA_DIR, filename.replace('.csv', '.parquet'))
        
        # 1. Get Dynamic Config for this specific file/timeframe
        config_dict, spread_cost = get_config_from_filename(filename)
        
        if config_dict:
            # 2. Run Pipeline
            result_msg = process_file_pipelined(input_path, output_path, config_dict, spread_cost)
            logger.info(f"RESULT [{filename}]: {result_msg}")
        else:
            logger.error(f"Skipping {filename}: Configuration mismatch (Instrument/Timeframe not in config.py).")

    end_time = time.time()
    logger.info(f"Bronze layer generation complete. Total Runtime: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()