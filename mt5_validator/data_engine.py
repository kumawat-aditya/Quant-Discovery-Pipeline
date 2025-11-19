# File: mt5_validator/data_engine.py

"""
Manages market data fetching and caching from MetaTrader 5.

This module provides a DataEngine class that maintains an in-memory,
up-to-date cache of OHLC data for specified symbols. It is designed to
minimize requests to the MT5 terminal by only fetching new candle data
as it becomes available.
"""

import MetaTrader5 as mt5
import pandas as pd
import logging
from typing import Dict, List, Optional

# Import the configuration for easy access to parameters
from live_config import TIMEFRAME, HISTORY_BARS_COUNT, SYMBOLS_TO_TRACK

# Set up a logger for this module
logger = logging.getLogger(__name__)


class DataEngine:
    """
    Handles fetching and caching of historical and live market data.
    """

    def __init__(self, symbols: List[str], timeframe, history_bars: int):
        """
        Initializes the DataEngine.

        Args:
            symbols (List[str]): A list of market symbols to track.
            timeframe: The MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1).
            history_bars (int): The number of historical bars to maintain in the cache.
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.history_bars = history_bars
        self._data_cache: Dict[str, pd.DataFrame] = {}
        logger.info(
            f"DataEngine initialized for symbols: {self.symbols} on timeframe {self.timeframe} "
            f"with a cache size of {self.history_bars} bars."
        )

    def prefill_cache(self) -> bool:
        """
        Fills the initial data cache for all symbols with historical data.
        This should be called once at the start of the application.

        Returns:
            bool: True if all symbols were successfully prefilled, False otherwise.
        """
        logger.info("Prefilling historical data cache for all symbols...")
        all_success = True
        for symbol in self.symbols:
            try:
                # mt5.copy_rates_from_pos fetches data from the current bar backwards
                rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.history_bars)
                if rates is None or len(rates) == 0:
                    logger.error(f"Failed to fetch initial historical data for {symbol}: {mt5.last_error()}")
                    all_success = False
                    continue

                # Convert the NumPy array of tuples to a Pandas DataFrame
                df = pd.DataFrame(rates)
                # The 'time' column is a Unix timestamp; convert it to a readable datetime
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                self._data_cache[symbol] = df
                logger.info(f"Successfully cached {len(df)} bars for {symbol}.")

            except Exception as e:
                logger.error(f"An exception occurred while prefilling data for {symbol}: {e}")
                all_success = False
        
        if all_success:
            logger.info("Historical data cache has been successfully prefilled.")
        else:
            logger.warning("Could not prefill historical data for one or more symbols.")
            
        return all_success

    def update_data(self) -> None:
        """
        Updates the cache for all symbols with the latest bar(s).
        This method is designed to be called in the main application loop.
        """
        for symbol in self.symbols:
            if symbol not in self._data_cache or self._data_cache[symbol].empty:
                logger.warning(f"No cached data for {symbol}. Cannot update. Try prefilling first.")
                continue

            last_known_time = self._data_cache[symbol].index[-1]
            
            # Fetch bars that have started since our last known bar's time
            new_rates = mt5.copy_rates_from(symbol, self.timeframe, last_known_time, 100) # Fetch up to 100 bars
            
            if new_rates is None or len(new_rates) <= 1:
                # No new bars or only the last known bar was returned
                continue
                
            new_df = pd.DataFrame(new_rates)
            new_df['time'] = pd.to_datetime(new_df['time'], unit='s')
            new_df.set_index('time', inplace=True)
            
            # The first row of new_df is the same as our last_known_time, so we drop it
            new_bars_df = new_df.iloc[1:]

            if not new_bars_df.empty:
                # Append the new bars to our existing cache
                updated_df = pd.concat([self._data_cache[symbol], new_bars_df])
                
                # Ensure the cache does not grow beyond the desired size
                self._data_cache[symbol] = updated_df.iloc[-self.history_bars:]
                logger.debug(f"Updated {symbol} with {len(new_bars_df)} new bar(s).")

    def get_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Retrieves the current data cache for a specific symbol.

        Args:
            symbol (str): The market symbol for which to get data.

        Returns:
            Optional[pd.DataFrame]: The DataFrame with the latest OHLC data,
                                     or None if the symbol is not tracked.
        """
        # We reset the index to match the format our research scripts expect (time as a column)
        df = self._data_cache.get(symbol)
        return df.reset_index() if df is not None else None


# --- Example Usage (for testing this file directly) ---
# if __name__ == "__main__":
#     import time
#     from mt5_connector import MT5Connector
#     from live_config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
#     logger.info("Testing DataEngine...")

#     try:
#         with MT5Connector(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, path=MT5_PATH) as connector:
            
#             # 1. Initialize the DataEngine
#             engine = DataEngine(symbols=SYMBOLS_TO_TRACK, timeframe=TIMEFRAME, history_bars=HISTORY_BARS_COUNT)
            
#             # 2. Prefill the cache with historical data
#             if not engine.prefill_cache():
#                 raise RuntimeError("Failed to prefill data cache. Aborting test.")

#             # 3. Get and display the initial state of the data
#             initial_df = engine.get_dataframe("EURUSD")
#             if initial_df is not None:
#                 print("\n--- Initial EURUSD Data ---")
#                 print(initial_df.tail())
#                 print(f"Initial cache size: {len(initial_df)} bars.")

#             # 4. Simulate waiting for new bars
#             logger.info("\nSimulating main loop: waiting 10 seconds before updating...")
#             time.sleep(10)
            
#             # 5. Update the data
#             engine.update_data()
#             logger.info("Update call finished.")

#             # 6. Get and display the updated data
#             updated_df = engine.get_dataframe("EURUSD")
#             if updated_df is not None:
#                 print("\n--- Updated EURUSD Data ---")
#                 print(updated_df.tail())
#                 print(f"Updated cache size: {len(updated_df)} bars.")

#                 if len(updated_df) > len(initial_df):
#                     logger.info("SUCCESS: New bar(s) were added to the cache.")
#                 else:
#                     logger.info("INFO: No new bars were formed during the wait period.")

#     except (ConnectionError, RuntimeError) as e:
#         logger.error(f"Test failed: {e}")
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during the test: {e}")
        
#     logger.info("DataEngine test finished.")