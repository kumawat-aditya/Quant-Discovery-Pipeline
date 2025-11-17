# diamond_backtester.py (V2.0 - Corrected Logic, Config, Logging & Restored Metrics)

"""
Diamond Layer - Backtester: The Mastery Engine

This script serves as the first and most critical quality gate in the Diamond
Layer. Its sole purpose is to perform a high-fidelity, parallelized backtest
of every discovered strategy on its HOME INSTRUMENT ONLY.

It filters the vast universe of potential strategies down to a small, elite
subset of "Master Strategies" that demonstrate robust performance on the data
they were trained on.

Key Functions:
- High-Fidelity Simulation: It iterates through pre-computed trigger times and
  simulates trade outcomes on a candle-by-candle basis, accounting for costs.
- Comprehensive Metrics: It calculates a wide array of professional performance
  metrics, including Profit Factor, Sharpe Ratio, SQN, and Max Drawdown.
- Quality Filtering: It applies a strict set of user-defined criteria to
  separate high-potential strategies from underperformers.
- Feedback Loop: It generates a blacklist of failing strategies, providing
  critical feedback to the Platinum Layer to refine future discovery runs.
"""

import logging
import os
import re
import sys
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

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
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_silver_features_df, worker_silver_features_np, worker_time_to_idx_lookup
    global worker_col_to_idx, worker_pip_size, worker_spread_cost
    
    worker_silver_features_df = silver_df
    lookup_cols = ['time', 'open', 'high', 'low', 'close']
    worker_col_to_idx = {col: i for i, col in enumerate(lookup_cols)}
    worker_silver_features_np = worker_silver_features_df[lookup_cols].to_numpy()
    worker_time_to_idx_lookup = pd.Series(worker_silver_features_df.index, index=worker_silver_features_df['time'])
    worker_pip_size, worker_spread_cost = pip_size, spread_cost

# --- CORE SIMULATION & METRICS LOGIC ---

def _calculate_level_price(entry_row: pd.Series, level_def: str) -> float:
    """Calculates the absolute price of a market level (e.g., 'SMA_50')."""
    if level_def not in entry_row or pd.isna(entry_row[level_def]):
        return np.nan
    return entry_row[level_def]

def run_simulation_for_strategy(strategy_details: pd.Series, base_dirs: Dict[str, str], master_instrument: str):
    """The main worker task. Performs a high-fidelity simulation for one strategy."""
    trigger_key = strategy_details['trigger_key']
    trigger_path = os.path.join(base_dirs['triggers'], master_instrument, master_instrument, f"{trigger_key}.parquet")
    
    try:
        trigger_times = pd.read_parquet(trigger_path)['time']
    except FileNotFoundError:
        return None

    trade_log = []
    trigger_indices = worker_time_to_idx_lookup.reindex(trigger_times).dropna().astype(int).values

    for entry_idx in trigger_indices:
        entry_row = worker_silver_features_df.iloc[entry_idx]
        entry_price = entry_row['close']
        entry_time = entry_row['time']
        
        bp_type, sl_def, sl_bin, tp_def, tp_bin = (
            strategy_details['type'], strategy_details['sl_def'], strategy_details['sl_bin'],
            strategy_details['tp_def'], strategy_details['tp_bin']
        )
        
        trade_type = 1 if strategy_details['trade_type'] == 'buy' else -1
        
        sl_price, tp_price = np.nan, np.nan
        
        if 'ratio' in sl_def:
            sl_price = entry_price * (1 - trade_type * sl_bin)
        else:
            level_price = _calculate_level_price(entry_row, sl_def)
            if np.isnan(level_price): continue
            if 'Pct' in bp_type:
                sl_price = level_price - trade_type * (abs(entry_price - level_price) * (sl_bin / 10.0))
            else: # BPS
                sl_price = level_price - trade_type * (sl_bin * worker_pip_size * 10)

        if 'ratio' in tp_def:
            tp_price = entry_price * (1 + trade_type * tp_bin)
        else:
            level_price = _calculate_level_price(entry_row, tp_def)
            if np.isnan(level_price): continue
            if 'Pct' in bp_type:
                tp_price = level_price + trade_type * (abs(entry_price - level_price) * (tp_bin / 10.0))
            else: # BPS
                tp_price = level_price + trade_type * (tp_bin * worker_pip_size * 10)

        if np.isnan(sl_price) or np.isnan(tp_price): continue

        limit = min(entry_idx + 1 + config.SIMULATION_MAX_LOOKFORWARD, len(worker_silver_features_np))
        outcome, exit_time, exit_price = 'expired', worker_silver_features_np[limit - 1, worker_col_to_idx['time']], worker_silver_features_np[limit - 1, worker_col_to_idx['close']]

        for j in range(entry_idx + 1, limit):
            current_high = worker_silver_features_np[j, worker_col_to_idx['high']]
            current_low = worker_silver_features_np[j, worker_col_to_idx['low']]
            
            if (trade_type == 1 and current_high >= tp_price) or (trade_type == -1 and current_low <= tp_price):
                outcome, exit_price = 'win', tp_price
                exit_time = worker_silver_features_np[j, worker_col_to_idx['time']]
                break
            if (trade_type == 1 and current_low <= sl_price) or (trade_type == -1 and current_high >= sl_price):
                outcome, exit_price = 'loss', sl_price
                exit_time = worker_silver_features_np[j, worker_col_to_idx['time']]
                break

        pnl_points = (exit_price - entry_price) * trade_type
        commission_cost = (config.SIMULATION_COMMISSION_PER_LOT / 100_000) * entry_price
        pnl_net = pnl_points - worker_spread_cost - commission_cost

        trade_log.append({'entry_time': entry_time, 'exit_time': pd.to_datetime(exit_time), 'pnl': pnl_net, 'outcome': outcome})

    if not trade_log: return None
    return calculate_performance_metrics(pd.DataFrame(trade_log), strategy_details)


def calculate_performance_metrics(trade_log_df: pd.DataFrame, strat_details: pd.Series) -> Dict:
    """Calculates a comprehensive suite of performance metrics from a trade log."""
    if trade_log_df.empty:
        return None
        
    metrics = strat_details.to_dict()
    trade_log_df['pnl_cumsum'] = trade_log_df['pnl'].cumsum()
    
    total_trades = len(trade_log_df)
    wins = trade_log_df[trade_log_df['pnl'] > 0]
    losses = trade_log_df[trade_log_df['pnl'] <= 0]
    
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # ### <<< CHANGE: Re-added Avg Win/Loss calculation >>>
    avg_win = wins['pnl'].mean()
    avg_loss = losses['pnl'].mean()
    
    peak = trade_log_df['pnl_cumsum'].cummax()
    max_drawdown = (peak - trade_log_df['pnl_cumsum']).max()
    
    avg_pnl = trade_log_df['pnl'].mean()
    std_pnl = trade_log_df['pnl'].std()
    sqn = (np.sqrt(total_trades) * avg_pnl) / std_pnl if std_pnl > 0 else 0
    
    # ### <<< CHANGE: Re-added Avg Win/Loss to the output metrics >>>
    metrics.update({
        'Total Trades': total_trades,
        'Profit Factor': round(profit_factor, 2),
        'Win Rate %': round(len(wins) * 100 / total_trades if total_trades > 0 else 0, 2),
        'Max Drawdown Pts': round(max_drawdown, 5),
        'SQN': round(sqn, 2),
        'Avg Win Pts': round(avg_win, 5) if pd.notna(avg_win) else 0,
        'Avg Loss Pts': round(avg_loss, 5) if pd.notna(avg_loss) else 0
    })
    
    return metrics


# --- MAIN ORCHESTRATOR ---
def run_backtester_for_instrument(master_instrument: str, base_dirs: Dict[str, str]):
    """Orchestrates the entire backtesting process for a single instrument."""
    logger.info(f"SETUP: Loading data for {master_instrument}...")
    
    diamond_strategies_path = os.path.join(base_dirs['diamond_strategies'], f"{master_instrument}.parquet")
    try:
        strategies_df = pd.read_parquet(diamond_strategies_path)
        if 'trade_type' not in strategies_df.columns:
            logger.error("FATAL: 'trade_type' column is missing from the diamond strategies file. Cannot run accurate backtest.")
            return
    except FileNotFoundError:
        logger.error(f"Diamond strategies file not found: {diamond_strategies_path}")
        return

    silver_features_path = os.path.join(base_dirs['silver_features'], f"{master_instrument}.parquet")
    try:
        silver_df = pd.read_parquet(silver_features_path)
        silver_df['time'] = pd.to_datetime(silver_df['time']).dt.tz_localize(None)
    except FileNotFoundError:
        logger.error(f"Silver features file not found: {silver_features_path}")
        return

    instrument_code = re.sub(r'\d+', '', master_instrument).upper()
    pip_size = 0.01 if "JPY" in instrument_code or "XAU" in instrument_code else 0.0001
    spread_pips = config.SIMULATION_SPREAD_PIPS.get(instrument_code, config.SIMULATION_SPREAD_PIPS["DEFAULT"])
    spread_cost = spread_pips * pip_size

    logger.info(f"Found {len(strategies_df)} strategies to backtest.")
    tasks = [row for _, row in strategies_df.iterrows()]
    
    logger.info("--- Running Parallel Backtests ---")
    worker_func = partial(run_simulation_for_strategy, base_dirs=base_dirs, master_instrument=master_instrument)
    pool_init_args = (silver_df, pip_size, spread_cost)
    all_results = []
    with Pool(processes=config.MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        for result in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Backtesting Strategies"):
            if result: all_results.append(result)

    if not all_results:
        logger.warning("No valid backtest results were generated. Exiting.")
        return
        
    results_df = pd.DataFrame(all_results)
    
    logger.info("FINALIZE: Filtering for master strategies and generating reports...")
    master_strategies_df = results_df[
        (results_df['Profit Factor'] >= config.DIAMOND_MIN_PROFIT_FACTOR) &
        (results_df['Max Drawdown Pts'] <= (config.DIAMOND_MAX_DRAWDOWN_PCT / 100.0)) &
        (results_df['Total Trades'] >= config.DIAMOND_MIN_TOTAL_TRADES) &
        (results_df['SQN'] >= config.DIAMOND_MIN_SQN)
    ]
    failed_strategies_df = results_df.drop(master_strategies_df.index)

    master_report_path = os.path.join(base_dirs['master_reports'], f"{master_instrument}.parquet")
    master_strategies_df.to_parquet(master_report_path, index=False)
    logger.info(f"  - Found {len(master_strategies_df)} Master Strategies. Report saved.")

    blacklist_path = os.path.join(base_dirs['platinum_blacklists'], f"{master_instrument}.parquet")
    if not failed_strategies_df.empty:
        blacklist_df = failed_strategies_df[['key', 'market_rule']]
        try:
            existing_blacklist = pd.read_parquet(blacklist_path)
            combined_blacklist = pd.concat([existing_blacklist, blacklist_df]).drop_duplicates(subset=['key', 'market_rule'])
        except FileNotFoundError:
            combined_blacklist = blacklist_df
        combined_blacklist.to_parquet(blacklist_path, index=False)
        logger.info(f"  - Blacklisted {len(failed_strategies_df)} strategies for feedback.")

def _select_instrument_interactively(diamond_dir: str) -> List[str]:
    """Scans for instruments and prompts user for selection."""
    logger.info("Interactive Mode: Scanning for instruments with prepared strategies...")
    try:
        all_instruments = sorted([os.path.splitext(f)[0] for f in os.listdir(diamond_dir) if f.endswith('.parquet')])
        if not all_instruments:
            logger.info("No prepared strategy files found in Diamond layer.")
            return []
            
        print("\n--- Select Instrument(s) to Backtest ---")
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
    except FileNotFoundError:
        logger.error(f"Diamond strategies directory not found at: {diamond_dir}")
        return []

def main():
    """Main execution function."""
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    LOGS_DIR = os.path.join(PROJECT_ROOT, config.LOG_DIR)
    setup_logging(LOGS_DIR, config.CONSOLE_LOG_LEVEL, config.FILE_LOG_LEVEL)
    
    start_time = time.time()
    base_dirs = {
        'silver_features': os.path.join(PROJECT_ROOT, 'silver_data', 'features'),
        'diamond_strategies': os.path.join(PROJECT_ROOT, 'diamond_data', 'strategies'),
        'triggers': os.path.join(PROJECT_ROOT, 'diamond_data', 'triggers'),
        'master_reports': os.path.join(PROJECT_ROOT, 'diamond_data', 'master_reports'),
        'platinum_blacklists': os.path.join(PROJECT_ROOT, 'platinum_data', 'blacklists'),
    }
    os.makedirs(base_dirs['master_reports'], exist_ok=True)
    os.makedirs(base_dirs['platinum_blacklists'], exist_ok=True)

    logger.info("--- Diamond Layer - Backtester: The Mastery Engine ---")
    
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        logger.info(f"Targeted Mode: Processing '{target_arg}'")
        instruments_to_process = [target_arg]
    else:
        instruments_to_process = _select_instrument_interactively(base_dirs['diamond_strategies'])
        
    if not instruments_to_process:
        logger.info("No instruments selected. Exiting.")
    else:
        logger.info(f"Queued {len(instruments_to_process)} instrument(s): {', '.join(instruments_to_process)}")
        for instrument in instruments_to_process:
            logger.info(f"--- Processing Instrument: {instrument} ---")
            try:
                run_backtester_for_instrument(instrument, base_dirs)
            except Exception:
                logger.critical(f"A fatal error occurred while processing {instrument}.", exc_info=True)

    end_time = time.time()
    logger.info(f"Diamond backtesting finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()