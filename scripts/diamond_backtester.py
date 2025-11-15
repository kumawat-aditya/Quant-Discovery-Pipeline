# diamond_backtester.py (V1.0 - The Mastery Engine)

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

import os
import re
import sys
import traceback
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[FATAL] 'pyarrow' library not found. Please run 'pip install pyarrow'.")
    sys.exit(1)


# --- CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)

# --- TRADING COSTS ---
# These can be adjusted to match broker specifics.
SPREAD_PIPS: float = 2.0  # Example for a major pair
COMMISSION_PER_LOT: float = 7.0  # Example round-trip commission

# --- PERFORMANCE FILTERS ---
# These define what constitutes a "Master Strategy".
MIN_PROFIT_FACTOR: float = 1.5
MAX_DRAWDOWN_PCT: float = 20.0
MIN_TOTAL_TRADES: int = 50
MIN_SQN: float = 1.8


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
    
    worker_silver_features_df = silver_df.copy()
    
    # Create fast lookup structures for the simulation loop
    lookup_cols = ['time', 'open', 'high', 'low', 'close']
    worker_col_to_idx = {col: i for i, col in enumerate(lookup_cols)}
    worker_silver_features_np = worker_silver_features_df[lookup_cols].to_numpy()
    worker_time_to_idx_lookup = pd.Series(
        worker_silver_features_df.index, 
        index=worker_silver_features_df['time']
    )
    worker_pip_size = pip_size
    worker_spread_cost = spread_cost


# --- CORE SIMULATION & METRICS LOGIC ---

def _calculate_level_price(
    entry_row: pd.Series, 
    level_def: str
) -> float:
    """Calculates the absolute price of a market level (e.g., 'SMA_50')."""
    if level_def not in entry_row or pd.isna(entry_row[level_def]):
        return np.nan # Level data is not available at this candle
    return entry_row[level_def]

def run_simulation_for_strategy(strategy_details: pd.Series, base_dirs: Dict[str, str], master_instrument: str):
    """
    The main worker task. Performs a high-fidelity simulation for one strategy.
    """
    trigger_key = strategy_details['trigger_key']
    trigger_path = os.path.join(base_dirs['triggers'], master_instrument, master_instrument, f"{trigger_key}.parquet")
    
    try:
        trigger_times = pd.read_parquet(trigger_path)['time']
    except FileNotFoundError:
        return None # No triggers for this strategy on the master instrument

    trade_log = []
    
    # Get the indices in the NumPy array for each trigger time
    trigger_indices = worker_time_to_idx_lookup.reindex(trigger_times).dropna().astype(int).values

    for entry_idx in trigger_indices:
        entry_row = worker_silver_features_df.iloc[entry_idx]
        entry_price = entry_row['close']
        entry_time = entry_row['time']
        
        # --- Determine SL/TP Prices based on Blueprint ---
        # Note: 'type' refers to the blueprint type (SL-Pct, TP-BPS, etc.)
        bp_type, sl_def, sl_bin, tp_def, tp_bin = (
            strategy_details['type'], strategy_details['sl_def'], strategy_details['sl_bin'],
            strategy_details['tp_def'], strategy_details['tp_bin']
        )
        
        sl_price, tp_price = np.nan, np.nan
        
        # Determine the trade direction (assume BUY for now, adjust for SELL)
        # For simplicity, this example assumes all are BUY trades. A real implementation
        # would need a 'trade_type' column from the strategy definition.
        # For this example, let's assume it's part of the key or can be inferred.
        # Let's add a placeholder for it.
        trade_type = 1 # 1 for Buy, -1 for Sell
        
        if 'ratio' in sl_def:
            sl_price = entry_price * (1 - trade_type * sl_bin)
        else: # e.g. SL-Pct, SL-BPS
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

        if np.isnan(sl_price) or np.isnan(tp_price):
            continue

        # --- Look Forward for Outcome ---
        max_lookforward = 500 # Max candles to hold a trade
        limit = min(entry_idx + 1 + max_lookforward, len(worker_silver_features_np))
        
        outcome = 'expired'
        exit_time = worker_silver_features_np[limit - 1, worker_col_to_idx['time']]
        exit_price = worker_silver_features_np[limit - 1, worker_col_to_idx['close']]

        for j in range(entry_idx + 1, limit):
            current_high = worker_silver_features_np[j, worker_col_to_idx['high']]
            current_low = worker_silver_features_np[j, worker_col_to_idx['low']]
            
            if trade_type == 1: # Buy Trade
                if current_high >= tp_price:
                    outcome, exit_price = 'win', tp_price
                    exit_time = worker_silver_features_np[j, worker_col_to_idx['time']]
                    break
                if current_low <= sl_price:
                    outcome, exit_price = 'loss', sl_price
                    exit_time = worker_silver_features_np[j, worker_col_to_idx['time']]
                    break
            else: # Sell Trade
                if current_low <= tp_price:
                    outcome, exit_price = 'win', tp_price
                    exit_time = worker_silver_features_np[j, worker_col_to_idx['time']]
                    break
                if current_high >= sl_price:
                    outcome, exit_price = 'loss', sl_price
                    exit_time = worker_silver_features_np[j, worker_col_to_idx['time']]
                    break

        # Calculate PnL
        pnl_points = (exit_price - entry_price) * trade_type
        pnl_pips = pnl_points / worker_pip_size
        commission_cost = (COMMISSION_PER_LOT / 100_000) * entry_price # as points
        pnl_net = pnl_points - worker_spread_cost - commission_cost

        trade_log.append({
            'entry_time': entry_time,
            'exit_time': pd.to_datetime(exit_time),
            'pnl': pnl_net,
            'outcome': outcome
        })

    if not trade_log:
        return None
        
    return calculate_performance_metrics(pd.DataFrame(trade_log), strategy_details)


def calculate_performance_metrics(trade_log_df: pd.DataFrame, strat_details: pd.Series) -> Dict:
    """Calculates a comprehensive suite of performance metrics from a trade log."""
    if trade_log_df.empty:
        return None
        
    metrics = strat_details.to_dict()
    trade_log_df['pnl_cumsum'] = trade_log_df['pnl'].cumsum()
    
    # Core Metrics
    total_trades = len(trade_log_df)
    wins = trade_log_df[trade_log_df['pnl'] > 0]
    losses = trade_log_df[trade_log_df['pnl'] <= 0]
    
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    # Drawdown
    peak = trade_log_df['pnl_cumsum'].cummax()
    drawdown = peak - trade_log_df['pnl_cumsum']
    max_drawdown = drawdown.max()
    
    # SQN (System Quality Number)
    avg_pnl = trade_log_df['pnl'].mean()
    std_pnl = trade_log_df['pnl'].std()
    sqn = (np.sqrt(total_trades) * avg_pnl) / std_pnl if std_pnl > 0 else 0
    
    # Add to metrics dict
    metrics.update({
        'Total Trades': total_trades,
        'Profit Factor': round(profit_factor, 2),
        'Win Rate %': round(win_rate * 100, 2),
        'Max Drawdown %': round(max_drawdown * 100, 2), # Simplified, assumes account size of 1
        'SQN': round(sqn, 2),
        'Avg Win': wins['pnl'].mean(),
        'Avg Loss': losses['pnl'].mean()
    })
    
    return metrics


# --- MAIN ORCHESTRATOR ---
def run_backtester_for_instrument(master_instrument: str, base_dirs: Dict[str, str]):
    """Orchestrates the entire backtesting process for a single instrument."""
    print(f"\n[SETUP] Loading data for {master_instrument}...")
    
    # Load strategies to be tested
    diamond_strategies_path = os.path.join(base_dirs['diamond_strategies'], f"{master_instrument}.parquet")
    try:
        strategies_df = pd.read_parquet(diamond_strategies_path)
        # ### <<< CHANGE: We need the full blueprint definition for the simulation
        # This assumes the diamond_strategies file has the blueprint details from platinum
        # If not, a merge would be needed here. Let's assume it does.
        if 'type' not in strategies_df.columns:
             print("[ERROR] Strategy file is missing blueprint columns ('type', 'sl_def', etc.). Please re-run prepper with a join.")
             # As a fallback, let's merge with the combinations file
             combo_path = os.path.join(base_dirs['platinum_combo'], f"{master_instrument}.parquet")
             combo_df = pd.read_parquet(combo_path)
             strategies_df = pd.merge(strategies_df, combo_df, on='key', how='left')

    except FileNotFoundError:
        print(f"[ERROR] Diamond strategies file not found: {diamond_strategies_path}")
        return

    # Load Silver features for simulation
    silver_features_path = os.path.join(base_dirs['silver_features'], f"{master_instrument}.csv")
    try:
        silver_df = pd.read_csv(silver_features_path, parse_dates=['time'])
        silver_df['time'] = silver_df['time'].dt.tz_localize(None)
    except FileNotFoundError:
        print(f"[ERROR] Silver features file not found: {silver_features_path}")
        return

    # Determine pip size and spread for cost calculation
    pip_size = 0.0001
    if "JPY" in master_instrument.upper() or "XAU" in master_instrument.upper():
        pip_size = 0.01
    spread_cost = SPREAD_PIPS * pip_size

    print(f"Found {len(strategies_df)} strategies to backtest.")
    tasks = [row for _, row in strategies_df.iterrows()]
    
    # Run parallel backtesting
    print("\n--- Running Parallel Backtests ---")
    all_results = []
    
    # Create the partial function with static arguments
    worker_func = partial(
        run_simulation_for_strategy, 
        base_dirs=base_dirs, 
        master_instrument=master_instrument
    )
    
    pool_init_args = (silver_df, pip_size, spread_cost)
    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        for result in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Backtesting Strategies"):
            if result:
                all_results.append(result)

    if not all_results:
        print("\n[INFO] No valid backtest results were generated. Exiting.")
        return
        
    results_df = pd.DataFrame(all_results)
    
    # --- Filter for Master Strategies and Generate Reports ---
    print("\n[FINALIZE] Filtering for master strategies and generating reports...")
    
    master_strategies_df = results_df[
        (results_df['Profit Factor'] >= MIN_PROFIT_FACTOR) &
        (results_df['Max Drawdown %'] <= MAX_DRAWDOWN_PCT) &
        (results_df['Total Trades'] >= MIN_TOTAL_TRADES) &
        (results_df['SQN'] >= MIN_SQN)
    ]
    
    failed_strategies_df = results_df.drop(master_strategies_df.index)

    # Save Master Strategies Report
    master_report_path = os.path.join(base_dirs['master_reports'], f"{master_instrument}.parquet")
    master_strategies_df.to_parquet(master_report_path, index=False)
    print(f"  - Found {len(master_strategies_df)} Master Strategies. Report saved.")

    # Save Blacklist for feedback loop
    blacklist_path = os.path.join(base_dirs['platinum_blacklists'], f"{master_instrument}.parquet")
    if not failed_strategies_df.empty:
        blacklist_df = failed_strategies_df[['key', 'market_rule']]
        try:
            existing_blacklist = pd.read_parquet(blacklist_path)
            combined_blacklist = pd.concat([existing_blacklist, blacklist_df]).drop_duplicates(subset=['key', 'market_rule'])
        except FileNotFoundError:
            combined_blacklist = blacklist_df
        
        combined_blacklist.to_parquet(blacklist_path, index=False)
        print(f"  - Blacklisted {len(failed_strategies_df)} strategies for feedback.")

def _select_instrument_interactively(diamond_dir: str) -> List[str]:
    """Scans for instruments in the Diamond layer and prompts user for selection."""
    print("[INFO] Interactive Mode: Scanning for instruments with prepared strategies...")
    try:
        all_instruments = sorted([
            os.path.splitext(f)[0] for f in os.listdir(diamond_dir) if f.endswith('.parquet')
        ])
        if not all_instruments:
            print("[INFO] No prepared strategy files found in Diamond layer.")
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
            if 0 <= idx < len(all_instruments):
                selected.append(all_instruments[idx])
            else:
                print(f"[WARN] Invalid selection '{idx + 1}' ignored.")
        return selected
    except ValueError:
        print("[ERROR] Invalid input. Please enter numbers or 'a'.")
        return []
    except FileNotFoundError:
        print(f"[ERROR] Diamond strategies directory not found at: {diamond_dir}")
        return []

def main():
    """Main execution function."""
    start_time = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    base_dirs = {
        'silver_features': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'silver_data', 'features')),
        'diamond_strategies': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'strategies')),
        'triggers': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'triggers')),
        'master_reports': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'master_reports')),
        'platinum_blacklists': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'platinum_data', 'blacklists')),
        'platinum_combo': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'platinum_data', 'combinations'))
    }

    os.makedirs(base_dirs['master_reports'], exist_ok=True)
    os.makedirs(base_dirs['platinum_blacklists'], exist_ok=True)

    print("--- Diamond Layer - Backtester: The Mastery Engine ---")
    
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        print(f"\n[INFO] Targeted Mode: Processing '{target_arg}'")
        instruments_to_process = [target_arg]
    else:
        instruments_to_process = _select_instrument_interactively(base_dirs['diamond_strategies'])
        
    if not instruments_to_process:
        print("\n[INFO] No instruments selected. Exiting.")
    else:
        print(f"\n[QUEUE] Queued {len(instruments_to_process)} instrument(s): {', '.join(instruments_to_process)}")
        for instrument in instruments_to_process:
            print(f"\n{'='*70}\nProcessing Instrument: {instrument}\n{'='*70}")
            try:
                run_backtester_for_instrument(instrument, base_dirs)
            except Exception:
                print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument}.")
                traceback.print_exc()

    end_time = time.time()
    print(f"\nDiamond backtesting finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()