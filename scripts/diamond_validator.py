# diamond_validator.py (V2.1 - Inclusive Validation & Enriched Logging)

"""
Diamond Layer - Validator: The Gauntlet & Analyser (Final Version)

... (Docstrings remain the same) ...

V2.1 Update:
- Now includes the master instrument in its validation run to ensure the final,
  data-rich reports contain the complete in-sample and out-of-sample picture.
- Enriches the final trade logs by including the full strategy blueprint
  definition with every trade record for complete analytical clarity.
"""

import logging
import os
import re
import sys
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
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

# --- WORKER-SPECIFIC GLOBALS ---
worker_silver_features_df: pd.DataFrame
worker_silver_features_np: np.ndarray
worker_time_to_idx_lookup: pd.Series
worker_col_to_idx: Dict[str, int]
worker_pip_size: float
worker_spread_cost: float

def init_worker(silver_df: pd.DataFrame, pip_size: float, spread_cost: float):
    """Initializer for worker processes, making a specific market's data available."""
    global worker_silver_features_df, worker_silver_features_np, worker_time_to_idx_lookup
    global worker_col_to_idx, worker_pip_size, worker_spread_cost
    
    worker_silver_features_df = silver_df
    lookup_cols = ['time', 'open', 'high', 'low', 'close']
    worker_col_to_idx = {col: i for i, col in enumerate(lookup_cols)}
    worker_silver_features_np = worker_silver_features_df[lookup_cols].to_numpy()
    worker_time_to_idx_lookup = pd.Series(worker_silver_features_df.index, index=worker_silver_features_df['time'])
    worker_pip_size, worker_spread_cost = pip_size, spread_cost

# --- CORE SIMULATION & METRICS ---

def run_single_simulation(
    strategy_details: pd.Series, base_dirs: Dict[str, str], master_instrument: str, validation_market: str
) -> List[Dict]:
    """Worker task: Simulates one strategy on the pre-loaded market data."""
    trigger_key = strategy_details['trigger_key']
    trigger_path = os.path.join(base_dirs['triggers'], master_instrument, validation_market, f"{trigger_key}.parquet")

    try:
        trigger_times = pd.read_parquet(trigger_path)['time']
    except FileNotFoundError:
        return []

    trade_log = []
    trade_type = 1 if strategy_details['trade_type'] == 'buy' else -1
    trigger_indices = worker_time_to_idx_lookup.reindex(trigger_times).dropna().astype(int).values

    for entry_idx in trigger_indices:
        entry_row = worker_silver_features_df.iloc[entry_idx]
        entry_price = entry_row['close']
        
        bp_type, sl_def, sl_bin, tp_def, tp_bin = (
            strategy_details['type'], strategy_details['sl_def'], strategy_details['sl_bin'],
            strategy_details['tp_def'], strategy_details['tp_bin']
        )
        sl_price, tp_price = np.nan, np.nan
        
        if 'ratio' in sl_def: sl_price = entry_price * (1 - trade_type * sl_bin)
        else:
            level = entry_row.get(sl_def, np.nan)
            if np.isnan(level): continue
            if 'Pct' in bp_type: sl_price = level - trade_type * (abs(entry_price - level) * (sl_bin / 10.0))
            else: sl_price = level - trade_type * (sl_bin * worker_pip_size * 10)

        if 'ratio' in tp_def: tp_price = entry_price * (1 + trade_type * tp_bin)
        else:
            level = entry_row.get(tp_def, np.nan)
            if np.isnan(level): continue
            if 'Pct' in bp_type: tp_price = level + trade_type * (abs(entry_price - level) * (tp_bin / 10.0))
            else: tp_price = level + trade_type * (tp_bin * worker_pip_size * 10)
            
        if np.isnan(sl_price) or np.isnan(tp_price): continue

        limit = min(entry_idx + 1 + config.DIAMOND_MAX_LOOKFORWARD, len(worker_silver_features_np))
        outcome, exit_price, exit_idx = 'expired', entry_price, limit - 1
        for j in range(entry_idx + 1, limit):
            high, low = worker_silver_features_np[j, worker_col_to_idx['high']], worker_silver_features_np[j, worker_col_to_idx['low']]
            if (trade_type == 1 and high >= tp_price) or (trade_type == -1 and low <= tp_price):
                outcome, exit_price, exit_idx = 'win', tp_price, j
                break
            if (trade_type == 1 and low <= sl_price) or (trade_type == -1 and high >= sl_price):
                outcome, exit_price, exit_idx = 'loss', sl_price, j
                break
        
        pnl_net = ((exit_price - entry_price) * trade_type) - worker_spread_cost - ((config.DIAMOND_COMMISSION_PER_LOT / 100_000) * entry_price)
        
        # ### <<< CHANGE: Enriched the trade log with full blueprint details >>>
        trade_log.append({
            'trigger_key': trigger_key, 'market': validation_market, 'entry_time': entry_row['time'],
            'exit_time': pd.to_datetime(worker_silver_features_np[exit_idx, worker_col_to_idx['time']]),
            'pnl': pnl_net, 'outcome': outcome,
            # Blueprint Definition
            'bp_type': bp_type, 'trade_type': strategy_details['trade_type'],
            'sl_def': sl_def, 'sl_bin': sl_bin, 'tp_def': tp_def, 'tp_bin': tp_bin,
            # Market Regimes & Context
            'trend_regime': entry_row.get('trend_regime_14', 'N/A'), 'vol_regime': entry_row.get('vol_regime_14', 'N/A'),
            'session': entry_row.get('session', 'N/A'), 'RSI_14': entry_row.get('RSI_14', np.nan),
        })
    return trade_log

def calculate_performance_metrics(trade_log_df: pd.DataFrame) -> pd.Series:
    """Calculates performance metrics from a trade log DataFrame."""
    if trade_log_df.empty: return pd.Series(dtype='float64')
    total_trades = len(trade_log_df)
    wins = trade_log_df[trade_log_df['pnl'] > 0]
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].sum())
    return pd.Series({
        'Profit Factor': round(gross_profit / gross_loss if gross_loss > 0 else np.inf, 2),
        'Win Rate %': round(len(wins) * 100 / total_trades if total_trades > 0 else 0, 2),
        'Total Trades': total_trades,
    })

# --- REPORT GENERATION ---

def generate_final_reports(all_trades: List[Dict], base_dirs: Dict[str, str], master_instrument: str):
    """Generates the final suite of analytical reports."""
    if not all_trades:
        logger.warning("No trades were simulated. Cannot generate reports.")
        return

    logger.info("FINALIZE: Generating final analytical reports...")
    all_trades_df = pd.DataFrame(all_trades)

    log_dir = os.path.join(base_dirs['trade_logs'], master_instrument)
    os.makedirs(log_dir, exist_ok=True)
    for (key, market), group in tqdm(all_trades_df.groupby(['trigger_key', 'market']), desc="Saving Trade Logs"):
        market_log_dir = os.path.join(log_dir, market)
        os.makedirs(market_log_dir, exist_ok=True)
        group.to_parquet(os.path.join(market_log_dir, f"{key}.parquet"), index=False)
    logger.info("  - Raw trade logs saved.")

    detailed_report = all_trades_df.groupby(['trigger_key', 'market']).apply(calculate_performance_metrics).reset_index()
    detailed_report.to_parquet(os.path.join(base_dirs['final_reports'], f"{master_instrument}_detailed.parquet"), index=False)
    logger.info("  - Detailed cross-market report saved.")

    summary_report = detailed_report.drop(columns=['market']).groupby('trigger_key').mean().reset_index()
    summary_report.to_parquet(os.path.join(base_dirs['final_reports'], f"{master_instrument}_summary.parquet"), index=False)
    logger.info("  - Summary report saved.")

    regime_cols = ['trend_regime', 'vol_regime', 'session']
    regime_reports = [
        all_trades_df.groupby(['trigger_key', col]).apply(calculate_performance_metrics).reset_index()
        .rename(columns={col: 'regime_value'}).assign(regime_type=col)
        for col in tqdm(regime_cols, desc="Generating Regime Analysis")
    ]
    pd.concat(regime_reports, ignore_index=True).to_parquet(
        os.path.join(base_dirs['final_reports'], f"{master_instrument}_regime_analysis.parquet"), index=False
    )
    logger.info("  - Regime analysis report saved.")

# --- MAIN ORCHESTRATOR ---

def run_validator_for_instrument(master_instrument: str, base_dirs: Dict[str, str]):
    """Orchestrates the validation and analysis for a given master instrument."""
    logger.info(f"SETUP: Loading master strategies for {master_instrument}...")
    master_report_path = os.path.join(base_dirs['master_reports'], f"{master_instrument}.parquet")
    try:
        master_strategies_df = pd.read_parquet(master_report_path)
    except FileNotFoundError:
        logger.error(f"Master strategies report not found: {master_report_path}")
        return

    if master_strategies_df.empty:
        logger.info("No master strategies to validate. Exiting.")
        return

    timeframe = re.search(r'(\d+)', master_instrument).group(1)
    all_instruments = [os.path.splitext(f)[0] for f in os.listdir(base_dirs['silver_features']) if f.endswith('.parquet')]
    
    # ### <<< CHANGE: The validation set NOW INCLUDES the master instrument >>>
    validation_markets = sorted([inst for inst in all_instruments if timeframe in inst])

    if not validation_markets:
        logger.warning("No validation markets found for this timeframe. Cannot run validator.")
        return

    logger.info(f"Found {len(master_strategies_df)} master strategies to validate across {len(validation_markets)} markets (including master).")
    
    all_trade_logs = []
    for market in validation_markets:
        is_master = " (Master Instrument)" if market == master_instrument else ""
        logger.info(f"--- Running Validation on Market: {market}{is_master} ---")
        silver_path = os.path.join(base_dirs['silver_features'], f"{market}.parquet")
        try:
            silver_df = pd.read_parquet(silver_path)
            silver_df['time'] = pd.to_datetime(silver_df['time']).dt.tz_localize(None)
        except FileNotFoundError:
            logger.warning(f"Silver features not found for {market}. Skipping.")
            continue
        
        instrument_code = re.sub(r'\d+', '', market).upper()
        pip_size = 0.01 if "JPY" in instrument_code or "XAU" in instrument_code else 0.0001
        spread_pips = config.DIAMOND_SPREAD_PIPS.get(instrument_code, config.DIAMOND_SPREAD_PIPS["DEFAULT"])
        spread_cost = spread_pips * pip_size

        tasks = [row for _, row in master_strategies_df.iterrows()]
        worker_func = partial(run_single_simulation, base_dirs=base_dirs, master_instrument=master_instrument, validation_market=market)
        
        pool_init_args = (silver_df, pip_size, spread_cost)
        with Pool(processes=config.MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
            for trade_list in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc=f"Simulating on {market}"):
                if trade_list: all_trade_logs.extend(trade_list)

    generate_final_reports(all_trade_logs, base_dirs, master_instrument)

def _select_instrument_interactively(master_reports_dir: str) -> List[str]:
    """Scans for master reports and prompts user for selection."""
    logger.info("Interactive Mode: Scanning for instruments with master strategies...")
    try:
        all_instruments = sorted([os.path.splitext(f)[0] for f in os.listdir(master_reports_dir) if f.endswith('.parquet')])
        if not all_instruments:
            logger.info("No master strategy reports found to validate.")
            return []
            
        print("\n--- Select Instrument(s) to Validate ---")
        for i, f in enumerate(all_instruments): print(f"  [{i+1}] {f}")
        print("  [a] Validate All")
        user_input = input("\nEnter selection (e.g., 1,3 or a): > ").strip().lower()
        if not user_input: return []
        if user_input == 'a': return all_instruments

        selected = [all_instruments[int(i.strip())-1] for i in user_input.split(',') if (int(i.strip())-1) < len(all_instruments)]
        return sorted(list(set(selected)))
    except (ValueError, IndexError):
        logger.error("Invalid input.")
        return []
    except FileNotFoundError:
        logger.error(f"Master reports directory not found at: {master_reports_dir}")
        return []

def main():
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    LOGS_DIR = os.path.join(PROJECT_ROOT, config.LOG_DIR)
    setup_logging(LOGS_DIR, config.CONSOLE_LOG_LEVEL, config.FILE_LOG_LEVEL)

    start_time = time.time()
    base_dirs = {
        'silver_features': os.path.join(PROJECT_ROOT, 'silver_data', 'features'),
        'triggers': os.path.join(PROJECT_ROOT, 'diamond_data', 'triggers'),
        'master_reports': os.path.join(PROJECT_ROOT, 'diamond_data', 'master_reports'),
        'final_reports': os.path.join(PROJECT_ROOT, 'diamond_data', 'final_reports'),
        'trade_logs': os.path.join(PROJECT_ROOT, 'diamond_data', 'trade_logs'),
    }
    for d in ['final_reports', 'trade_logs']: os.makedirs(base_dirs[d], exist_ok=True)

    logger.info("--- Diamond Layer - Validator: The Gauntlet & Analyser ---")
    
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        instruments_to_process = [target_arg]
    else:
        instruments_to_process = _select_instrument_interactively(base_dirs['master_reports'])
        
    if not instruments_to_process:
        logger.info("No instruments selected. Exiting.")
    else:
        for instrument in instruments_to_process:
            logger.info(f"--- Processing Master Strategies from: {instrument} ---")
            try:
                run_validator_for_instrument(instrument, base_dirs)
            except Exception:
                logger.critical(f"A fatal error occurred while processing {instrument}.", exc_info=True)

    logger.info(f"Diamond validation finished. Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()