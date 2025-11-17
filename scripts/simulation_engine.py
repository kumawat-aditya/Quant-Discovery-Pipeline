"""
Shared High-Fidelity Simulation Engine for the Diamond Layer.

This module contains the core, high-performance backtesting function used by both
the `diamond_backtester` and `diamond_validator`. By centralizing this logic,
we adhere to the DRY (Don't Repeat Yourself) principle, making the system more
robust, easier to maintain, and simpler to upgrade with new features (e.g.,
trailing stops, different cost models).

The engine is designed to be called from a worker process that has been
initialized with the necessary market data via the `init_worker` function
in the calling script.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import os

# Import the main config file to access simulation parameters
import config

# --- Worker Globals (Expected to be populated by the calling script's init_worker) ---
# These lines are for type hinting and clarity; they are not executed in the main process.
worker_silver_features_df: pd.DataFrame
worker_silver_features_np: np.ndarray
worker_time_to_idx_lookup: pd.Series
worker_col_to_idx: Dict[str, int]
worker_pip_size: float
worker_spread_cost: float


def _calculate_level_price(entry_row: pd.Series, level_def: str) -> float:
    """Helper to safely calculate the absolute price of a market level."""
    if level_def not in entry_row or pd.isna(entry_row[level_def]):
        return np.nan
    return entry_row[level_def]


def run_simulation(
    strategy_details: pd.Series,
    base_dirs: Dict[str, str],
    master_instrument: str,
    simulation_market: str,
    deep_log: bool = False
) -> List[Dict]:
    """
    Performs a high-fidelity simulation for a single strategy on a single market.

    Args:
        strategy_details: A pandas Series containing the full strategy definition.
        base_dirs: A dictionary of base directory paths.
        master_instrument: The instrument on which the strategy was discovered.
        simulation_market: The instrument on which to run this specific simulation.
        deep_log: If True, returns a rich, detailed log for each trade.
                  If False, returns a simple log sufficient for metric calculation.

    Returns:
        A list of dictionaries, where each dictionary represents a single trade.
    """
    trigger_key = strategy_details['trigger_key']
    trigger_path = os.path.join(base_dirs['triggers'], master_instrument, simulation_market, f"{trigger_key}.parquet")

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
            level = _calculate_level_price(entry_row, sl_def)
            if np.isnan(level): continue
            if 'Pct' in bp_type: sl_price = level - trade_type * (abs(entry_price - level) * (sl_bin / 10.0))
            else: sl_price = level - trade_type * (sl_bin * worker_pip_size * 10)

        if 'ratio' in tp_def: tp_price = entry_price * (1 + trade_type * tp_bin)
        else:
            level = _calculate_level_price(entry_row, tp_def)
            if np.isnan(level): continue
            if 'Pct' in bp_type: tp_price = level + trade_type * (abs(entry_price - level) * (tp_bin / 10.0))
            else: tp_price = level + trade_type * (tp_bin * worker_pip_size * 10)
            
        if np.isnan(sl_price) or np.isnan(tp_price): continue

        limit = min(entry_idx + 1 + config.SIMULATION_MAX_LOOKFORWARD, len(worker_silver_features_np))
        outcome, exit_price, exit_idx = 'expired', entry_price, limit - 1
        for j in range(entry_idx + 1, limit):
            high, low = worker_silver_features_np[j, worker_col_to_idx['high']], worker_silver_features_np[j, worker_col_to_idx['low']]
            if (trade_type == 1 and high >= tp_price) or (trade_type == -1 and low <= tp_price):
                outcome, exit_price, exit_idx = 'win', tp_price, j
                break
            if (trade_type == 1 and low <= sl_price) or (trade_type == -1 and high >= sl_price):
                outcome, exit_price, exit_idx = 'loss', sl_price, j
                break
        
        pnl_net = ((exit_price - entry_price) * trade_type) - worker_spread_cost - ((config.SIMULATION_COMMISSION_PER_LOT / 100_000) * entry_price)
        
        # Conditionally create the trade record based on the requested log depth
        if deep_log:
            trade_record = {
                'trigger_key': trigger_key, 'market': simulation_market, 'entry_time': entry_row['time'],
                'exit_time': pd.to_datetime(worker_silver_features_np[exit_idx, worker_col_to_idx['time']]),
                'pnl': pnl_net, 'outcome': outcome,
                'bp_type': bp_type, 'trade_type': strategy_details['trade_type'],
                'sl_def': sl_def, 'sl_bin': sl_bin, 'tp_def': tp_def, 'tp_bin': tp_bin,
                'trend_regime': entry_row.get('trend_regime_14', 'N/A'), 'vol_regime': entry_row.get('vol_regime_14', 'N/A'),
                'session': entry_row.get('session', 'N/A'), 'RSI_14': entry_row.get('RSI_14', np.nan),
            }
        else: # Simple log for backtester metrics
            trade_record = {
                'pnl': pnl_net,
                'outcome': outcome
            }
        trade_log.append(trade_record)
        
    return trade_log