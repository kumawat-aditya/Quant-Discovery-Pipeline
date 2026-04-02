# File: mt5_validator/trading_engine.py

"""
Handles trade execution and position management via the MT5 terminal.

This module provides a TradingEngine class that takes a strategy signal,
calculates precise SL/TP levels, constructs a valid trade request,
and sends the order for execution. It also includes basic risk management
checks.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

from legacy.mt5_validator.mt5_connector import MT5Connector
from legacy.mt5_validator.live_config import LOT_SIZE, MAX_OPEN_TRADES_PER_STRATEGY

# Set up a logger for this module
logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Manages the process of executing and monitoring trades.
    """

    def __init__(self, connector: MT5Connector):
        """
        Initializes the TradingEngine with an active MT5Connector.

        Args:
            connector (MT5Connector): The connector instance for communicating
                                      with the MT5 terminal.
        """
        self.connector = connector
        if not self.connector.is_connected:
            raise ConnectionError("TradingEngine requires an active MT5 connection.")
        logger.info("TradingEngine initialized.")

    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Fetches and caches critical information for a symbol."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return {}
        return {
            "point": info.point,
            "ask": info.ask,
            "bid": info.bid,
            "digits": info.digits
        }

    def _calculate_sl_tp_prices(self, strategy: Dict[str, Any], entry_price: float,
                                silver_features_row: pd.Series) -> Dict[str, float]:
        """
        Calculates the absolute Stop-Loss and Take-Profit prices for a trade.
        This logic is a direct replication of the research simulation_engine.

        Args:
            strategy (Dict[str, Any]): The full strategy dictionary.
            entry_price (float): The price at which the trade will be entered.
            silver_features_row (pd.Series): The row of Silver-level features for
                                             the entry candle.

        Returns:
            Dict[str, float]: A dictionary containing the 'sl_price' and 'tp_price'.
        """
        trade_type = 1 if strategy['trade_type'] == 'buy' else -1
        bp_type, sl_def, sl_bin, tp_def, tp_bin = (
            strategy['type'], strategy['sl_def'], strategy['sl_bin'],
            strategy['tp_def'], strategy['tp_bin']
        )
        
        sl_price, tp_price = 0.0, 0.0
        pip_size = 0.01 if "JPY" in silver_features_row.name.upper() or "XAU" in silver_features_row.name.upper() else 0.0001
        
        # --- SL Calculation (Mirrors simulation_engine) ---
        if 'ratio' in sl_def:
            sl_price = entry_price * (1 - trade_type * sl_bin)
        else:
            level = silver_features_row.get(sl_def)
            if pd.isna(level): return {'sl_price': 0.0, 'tp_price': 0.0}
            if 'Pct' in bp_type:
                sl_price = level - trade_type * (abs(entry_price - level) * (sl_bin / 10.0))
            else: # BPS type
                sl_price = level - trade_type * (sl_bin * pip_size * 10)

        # --- TP Calculation (Mirrors simulation_engine) ---
        if 'ratio' in tp_def:
            tp_price = entry_price * (1 + trade_type * tp_bin)
        else:
            level = silver_features_row.get(tp_def)
            if pd.isna(level): return {'sl_price': 0.0, 'tp_price': 0.0}
            if 'Pct' in bp_type:
                tp_price = level + trade_type * (abs(entry_price - level) * (tp_bin / 10.0))
            else: # BPS type
                tp_price = level + trade_type * (tp_bin * pip_size * 10)
                
        return {'sl_price': sl_price, 'tp_price': tp_price}

    def _construct_trade_request(self, symbol: str, strategy: Dict[str, Any],
                                 sl_price: float, tp_price: float) -> Dict[str, Any]:
        """Builds the dictionary for the mt5.order_send() function."""
        symbol_info = self._get_symbol_info(symbol)
        trade_type = strategy['trade_type']
        
        order_type = mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL
        price = symbol_info['ask'] if trade_type == 'buy' else symbol_info['bid']
        
        # Round SL/TP to the correct number of decimal places for the symbol
        digits = symbol_info.get('digits', 5)
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": LOT_SIZE,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 10, # Slippage in points
            "magic": 234000, # A magic number to identify trades from this EA
            "comment": strategy['trigger_key'], # CRITICAL for performance tracking
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return request

    def execute_trade(self, strategy: Dict[str, Any], symbol: str, silver_features_row: pd.Series):
        """
        The main public method to execute a trade based on a strategy signal.

        Args:
            strategy (Dict[str, Any]): The full strategy dictionary object.
            symbol (str): The symbol to trade.
            silver_features_row (pd.Series): The live Silver-level feature data.
        """
        trigger_key = strategy['trigger_key']
        
        # --- Pre-Trade Risk Checks ---
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions is not None:
            strategy_positions = [p for p in open_positions if p.comment == trigger_key]
            if len(strategy_positions) >= MAX_OPEN_TRADES_PER_STRATEGY:
                logger.debug(f"Skipping trade for {trigger_key} on {symbol}. Max positions already open.")
                return

        logger.info(f"TRADE SIGNAL DETECTED for strategy {trigger_key} on {symbol}.")

        # --- Calculate Trade Parameters ---
        symbol_info = self._get_symbol_info(symbol)
        entry_price = symbol_info['ask'] if strategy['trade_type'] == 'buy' else symbol_info['bid']
        
        sl_tp = self._calculate_sl_tp_prices(strategy, entry_price, silver_features_row)
        sl_price, tp_price = sl_tp['sl_price'], sl_tp['tp_price']

        if sl_price == 0.0 or tp_price == 0.0:
            logger.error(f"Failed to calculate valid SL/TP for {trigger_key} on {symbol}. Aborting trade.")
            return

        # --- Construct and Send Order ---
        request = self._construct_trade_request(symbol, strategy, sl_price, tp_price)
        
        logger.debug(f"Sending trade request: {request}")
        
        result = mt5.order_send(request)

        # --- Post-Trade Logging ---
        if result is None:
            logger.error(f"Order send failed. No result object. Error: {mt5.last_error()}")
            return

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order send failed for {trigger_key} on {symbol}. Retcode: {result.retcode}")
            # Optionally, log the full result object for debugging
            logger.debug(f"Full order result: {result}")
        else:
            logger.info(f"SUCCESS: Trade executed for {trigger_key} on {symbol}. Order Ticket: {result.order}")