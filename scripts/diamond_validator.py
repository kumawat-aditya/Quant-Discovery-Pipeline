# diamond_validator.py (V1.0 - The Gauntlet & Analyser)

"""
Diamond Layer - Validator: The Gauntlet & Analyser

This is the final, most rigorous script in the strategy discovery pipeline. It
takes the elite "Master Strategies" identified by the backtester and subjects
them to two critical processes:

1.  The Gauntlet (Out-of-Sample Validation):
    It stress-tests each master strategy by running the same high-fidelity
    simulation across a range of different, unseen markets of the same
    timeframe. This is the ultimate test of a strategy's robustness and ability
    to generalize.

2.  The Analyser (Deep Reporting):
    It generates a suite of final, UI-ready analytical reports. Its most
    critical output is the "Regime Analysis," which breaks down a strategy's
    performance by market condition (e.g., trending vs. ranging, high vs. low
    volatility, specific trading sessions). This provides the "why" behind a
    strategy's PnL and reveals its true operational edges.
"""

import os
import re
import sys
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple
import time
from functools import partial

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

# --- TRADING COSTS (should be consistent with backtester) ---
SPREAD_PIPS: float = 2.0
COMMISSION_PER_LOT: float = 7.0


# --- CORE SIMULATION & METRICS (Adapted from Backtester) ---

# Note: The core simulation logic is nearly identical to the backtester.
# In a larger project, this could be refactored into a shared library.

def _calculate_level_price(entry_row: pd.Series, level_def: str) -> float:
    """Calculates the absolute price of a market level."""
    if level_def not in entry_row or pd.isna(entry_row[level_def]):
        return np.nan
    return entry_row[level_def]

def calculate_performance_metrics(trade_log_df: pd.DataFrame) -> pd.Series:
    """Calculates performance metrics from a trade log DataFrame."""
    if trade_log_df.empty:
        return pd.Series(dtype='float64')
        
    total_trades = len(trade_log_df)
    wins = trade_log_df[trade_log_df['pnl'] > 0]
    
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(trade_log_df[trade_log_df['pnl'] <= 0]['pnl'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    return pd.Series({
        'Profit Factor': round(profit_factor, 2),
        'Win Rate %': round(win_rate * 100, 2),
        'Total Trades': total_trades,
    })


# --- PARALLEL WORKER FUNCTION ---

def run_validation_on_market(
    task: Tuple[pd.Series, str], 
    base_dirs: Dict[str, str], 
    master_instrument: str
) -> List[Dict]:
    """
    Worker task: Simulates one strategy on one validation market and returns
    a deeply logged record of every trade.
    """
    strategy_details, validation_market = task
    
    # Load necessary data for this specific task
    silver_path = os.path.join(base_dirs['silver_features'], f"{validation_market}.csv")
    trigger_key = strategy_details['trigger_key']
    trigger_path = os.path.join(base_dirs['triggers'], master_instrument, validation_market, f"{trigger_key}.parquet")

    try:
        silver_df = pd.read_csv(silver_path, parse_dates=['time'])
        silver_df['time'] = silver_df['time'].dt.tz_localize(None)
        trigger_times = pd.read_parquet(trigger_path)['time']
    except FileNotFoundError:
        return [] # No triggers on this market or data is missing

    # Create fast lookup structures for this specific market's data
    silver_np = silver_df[['time', 'open', 'high', 'low', 'close']].to_numpy()
    time_to_idx = pd.Series(silver_df.index, index=silver_df['time'])
    col_to_idx = {col: i for i, col in enumerate(['time', 'open', 'high', 'low', 'close'])}

    # Determine costs for this market
    pip_size = 0.01 if "JPY" in validation_market.upper() or "XAU" in validation_market.upper() else 0.0001
    spread_cost = SPREAD_PIPS * pip_size
    
    trade_log = []
    trade_type = 1 if strategy_details.get('trade_type', 'buy') == 'buy' else -1
    trigger_indices = time_to_idx.reindex(trigger_times).dropna().astype(int).values

    for entry_idx in trigger_indices:
        entry_row = silver_df.iloc[entry_idx]
        entry_price = entry_row['close']
        
        # --- Calculate SL/TP Prices (same logic as backtester) ---
        bp_type, sl_def, sl_bin, tp_def, tp_bin = (
            strategy_details['type'], strategy_details['sl_def'], strategy_details['sl_bin'],
            strategy_details['tp_def'], strategy_details['tp_bin']
        )
        sl_price, tp_price = np.nan, np.nan
        
        if 'ratio' in sl_def: sl_price = entry_price * (1 - trade_type * sl_bin)
        else:
            level = _calculate_level_price(entry_row, sl_def)
            if np.isnan(level): continue
            if 'Pct' in bp_type: sl_price = level - trade_type * (abs(entry_price - level) * (sl_bin / 10.0))
            else: sl_price = level - trade_type * (sl_bin * pip_size * 10)

        if 'ratio' in tp_def: tp_price = entry_price * (1 + trade_type * tp_bin)
        else:
            level = _calculate_level_price(entry_row, tp_def)
            if np.isnan(level): continue
            if 'Pct' in bp_type: tp_price = level + trade_type * (abs(entry_price - level) * (tp_bin / 10.0))
            else: tp_price = level + trade_type * (tp_bin * pip_size * 10)
            
        if np.isnan(sl_price) or np.isnan(tp_price): continue

        # --- High-Fidelity Simulation Loop ---
        outcome, exit_price, exit_idx = 'expired', entry_price, entry_idx
        limit = min(entry_idx + 1 + 500, len(silver_np))
        for j in range(entry_idx + 1, limit):
            high, low = silver_np[j, col_to_idx['high']], silver_np[j, col_to_idx['low']]
            if (trade_type == 1 and high >= tp_price) or (trade_type == -1 and low <= tp_price):
                outcome, exit_price, exit_idx = 'win', tp_price, j
                break
            if (trade_type == 1 and low <= sl_price) or (trade_type == -1 and high >= sl_price):
                outcome, exit_price, exit_idx = 'loss', sl_price, j
                break
        
        # --- Deep Contextual Logging ---
        pnl_net = ((exit_price - entry_price) * trade_type) - spread_cost - ((COMMISSION_PER_LOT / 100_000) * entry_price)
        
        trade_log.append({
            'trigger_key': trigger_key,
            'market': validation_market,
            'entry_time': entry_row['time'],
            'exit_time': pd.to_datetime(silver_np[exit_idx, col_to_idx['time']]),
            'pnl': pnl_net,
            'outcome': outcome,
            'trend_regime': entry_row.get('trend_regime_14', 'N/A'),
            'vol_regime': entry_row.get('vol_regime_14', 'N/A'),
            'session': entry_row.get('session', 'N/A'),
            'RSI_14': entry_row.get('RSI_14', np.nan),
            'ADX_14': entry_row.get('ADX_14', np.nan),
            'ATR_14': entry_row.get('ATR_14', np.nan),
            'support': entry_row.get('support', np.nan),
            'resistance': entry_row.get('resistance', np.nan),
        })
        
    return trade_log

# --- REPORT GENERATION ---

def generate_final_reports(all_trades: List[Dict], base_dirs: Dict[str, str], master_instrument: str):
    """
    Takes the raw list of all trades from all workers and generates the final
    suite of analytical reports.
    """
    if not all_trades:
        print("[WARN] No trades were simulated. Cannot generate reports.")
        return

    print("\n[FINALIZE] Generating final analytical reports...")
    all_trades_df = pd.DataFrame(all_trades)

    # 1. Save Raw Trade Logs
    log_dir = os.path.join(base_dirs['trade_logs'], master_instrument)
    os.makedirs(log_dir, exist_ok=True)
    for (key, market), group in tqdm(all_trades_df.groupby(['trigger_key', 'market']), desc="Saving Trade Logs"):
        market_log_dir = os.path.join(log_dir, market)
        os.makedirs(market_log_dir, exist_ok=True)
        group.to_parquet(os.path.join(market_log_dir, f"{key}.parquet"), index=False)
    print("  - Raw trade logs saved.")

    # 2. Detailed Report (Performance per strategy, per market)
    detailed_report = all_trades_df.groupby(['trigger_key', 'market']).apply(calculate_performance_metrics).reset_index()
    detailed_report.to_parquet(os.path.join(base_dirs['final_reports'], f"{master_instrument}_detailed.parquet"), index=False)
    print("  - Detailed cross-market report saved.")

    # 3. Summary Report (Average performance per strategy across all markets)
    summary_report = detailed_report.drop(columns=['market']).groupby('trigger_key').mean().reset_index()
    summary_report.to_parquet(os.path.join(base_dirs['final_reports'], f"{master_instrument}_summary.parquet"), index=False)
    print("  - Summary report saved.")

    # 4. Regime Analysis Report (The "Truth" File)
    regime_reports = []
    regime_cols = ['trend_regime', 'vol_regime', 'session']
    for col in tqdm(regime_cols, desc="Generating Regime Analysis"):
        regime_df = all_trades_df.groupby(['trigger_key', col]).apply(calculate_performance_metrics).reset_index()
        regime_df = regime_df.rename(columns={col: 'regime_value'})
        regime_df['regime_type'] = col
        regime_reports.append(regime_df)
        
    final_regime_report = pd.concat(regime_reports, ignore_index=True)
    final_regime_report.to_parquet(os.path.join(base_dirs['final_reports'], f"{master_instrument}_regime_analysis.parquet"), index=False)
    print("  - Regime analysis report saved.")


# --- MAIN ORCHESTRATOR ---

def run_validator_for_instrument(master_instrument: str, base_dirs: Dict[str, str]):
    """Orchestrates the validation and analysis for a given master instrument."""
    print(f"\n[SETUP] Loading master strategies for {master_instrument}...")
    
    master_report_path = os.path.join(base_dirs['master_reports'], f"{master_instrument}.parquet")
    try:
        master_strategies_df = pd.read_parquet(master_report_path)
    except FileNotFoundError:
        print(f"[ERROR] Master strategies report not found: {master_report_path}")
        return

    if master_strategies_df.empty:
        print("[INFO] No master strategies to validate. Exiting.")
        return

    # Find all other instruments of the same timeframe to use for validation
    timeframe = re.search(r'(\d+)', master_instrument).group(1)
    all_instruments = [os.path.splitext(f)[0] for f in os.listdir(base_dirs['silver_features']) if f.endswith('.csv')]
    validation_markets = sorted([inst for inst in all_instruments if timeframe in inst and inst != master_instrument])

    if not validation_markets:
        print("[WARN] No other validation markets found for this timeframe. Cannot run validator.")
        return

    print(f"Found {len(master_strategies_df)} master strategies to validate across {len(validation_markets)} markets.")
    
    # Create the full list of tasks: (strategy_row, market_name)
    tasks = [
        (row, market)
        for _, row in master_strategies_df.iterrows()
        for market in validation_markets
    ]
    
    print("\n--- Running Parallel Out-of-Sample Validations ---")
    all_trade_logs = []
    
    worker_func = partial(
        run_validation_on_market, 
        base_dirs=base_dirs, 
        master_instrument=master_instrument
    )
    
    with Pool(processes=MAX_CPU_USAGE) as pool:
        for trade_list in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Validating Strategies"):
            if trade_list:
                all_trade_logs.extend(trade_list)

    generate_final_reports(all_trade_logs, base_dirs, master_instrument)

def _select_instrument_interactively(master_reports_dir: str) -> List[str]:
    """Scans for master reports and prompts user for selection."""
    print("[INFO] Interactive Mode: Scanning for instruments with master strategies...")
    try:
        all_instruments = sorted([
            os.path.splitext(f)[0] for f in os.listdir(master_reports_dir) if f.endswith('.parquet')
        ])
        if not all_instruments:
            print("[INFO] No master strategy reports found to validate.")
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
        print("[ERROR] Invalid input.")
        return []
    except FileNotFoundError:
        print(f"[ERROR] Master reports directory not found at: {master_reports_dir}")
        return []

def main():
    start_time = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    base_dirs = {
        'silver_features': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'silver_data', 'features')),
        'triggers': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'triggers')),
        'master_reports': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'master_reports')),
        'final_reports': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'final_reports')),
        'trade_logs': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'diamond_data', 'trade_logs')),
    }
    
    for d in ['final_reports', 'trade_logs']: os.makedirs(base_dirs[d], exist_ok=True)

    print("--- Diamond Layer - Validator: The Gauntlet & Analyser ---")
    
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        instruments_to_process = [target_arg]
    else:
        instruments_to_process = _select_instrument_interactively(base_dirs['master_reports'])
        
    if not instruments_to_process:
        print("\n[INFO] No instruments selected. Exiting.")
    else:
        for instrument in instruments_to_process:
            print(f"\n{'='*70}\nProcessing Master Strategies from: {instrument}\n{'='*70}")
            try:
                run_validator_for_instrument(instrument, base_dirs)
            except Exception:
                print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument}.")
                traceback.print_exc()

    print(f"\nDiamond validation finished. Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()