# File: mt5_validator/mt5_validator.py

"""
The Main Controller for the MT5 Live Strategy Validator.

This script serves as the master orchestrator for the live testing application.
It initializes all engine components, runs the main event loop, and coordinates
the process of fetching data, generating features, checking for strategy signals,
and executing trades.
"""

import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

# Import all our custom engine components and configuration
from live_config import MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, DIAMOND_DATA_PATH, TIMEFRAME, HISTORY_BARS_COUNT, SYMBOLS_TO_TRACK, POLLING_INTERVAL_SECONDS
from mt5_connector import MT5Connector
from data_engine import DataEngine
from feature_engine import FeatureEngine
from strategy_loader import StrategyLoader
from trading_engine import TradingEngine

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mt5_validator/logs/validator.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)


class MT5Validator:
    """
    The main application class that orchestrates the entire validation process.
    """

    def __init__(self):
        """Initializes all the core components of the application."""
        logger.info("Initializing MT5 Live Strategy Validator...")
        
        # Initialize components that DO NOT require an active connection yet
        self.connector: MT5Connector = MT5Connector(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, path=MT5_PATH)
        self.strategy_loader: StrategyLoader = StrategyLoader(diamond_path=DIAMOND_DATA_PATH)
        self.data_engine: DataEngine = DataEngine(symbols=SYMBOLS_TO_TRACK, timeframe=TIMEFRAME, history_bars=HISTORY_BARS_COUNT)
        self.feature_engine: FeatureEngine = FeatureEngine()
        
        # Placeholder for the trading engine (requires connection first)
        self.trading_engine: Optional[TradingEngine] = None
        
        self.strategies_by_instrument: Dict[str, List[Dict[str, Any]]] = {}
        self.is_running = False

    def startup(self):
        """
        Connects to MT5, loads strategies, and prefills data caches.
        This is the main setup routine before the loop starts.
        """
        logger.info("--- Validator Starting Up ---")
        
        # 1. Connect to MetaTrader 5
        self.connector.connect()
        if not self.connector.is_connected:
            logger.error("Could not connect to MT5. The application cannot start.")
            return False

        # 1b. Initialize TradingEngine NOW that connection is active
        # This fixes the crash
        try:
            self.trading_engine = TradingEngine(self.connector)
        except ConnectionError as e:
            logger.error(f"Failed to initialize TradingEngine: {e}")
            self.connector.disconnect()
            return False

        # 2. Load all strategies for the symbols we are tracking
        for symbol in SYMBOLS_TO_TRACK:
            # A simple way to infer instrument name from symbol for now
            # Adjust "H1" if your timeframe in live_config is different
            # This assumes your research file is named like "EURUSD15.parquet"
            # timeframe_suffix = "60" if self.data_engine.timeframe == 16385 else "UNKNOWN" # 16385 is H1 int value usually
            
            # Better mapping logic based on standard MT5 constants:
            # M1=1, M5=5, M15=15, M30=30, H1=16385 (usually, but safer to rely on logic)
            # For simplicity, assuming you used '60' for H1 in research:
            instrument_name = f"{symbol}15" 
            
            logger.info(f"Attempting to load strategies for {instrument_name}...")
            strategies = self.strategy_loader.load_master_strategies(instrument_name)
            if strategies:
                self.strategies_by_instrument[symbol] = strategies
                logger.info(f"Loaded {len(strategies)} strategies for {symbol}.")
            else:
                logger.warning(f"No master strategies found for {instrument_name}. It will not be traded.")

        if not self.strategies_by_instrument:
            logger.error("No strategies were loaded for any tracked symbols. Aborting startup.")
            self.connector.disconnect()
            return False

        # 3. Prefill the data engine's cache with historical data
        if not self.data_engine.prefill_cache():
            logger.error("Failed to prefill historical data. The application cannot start.")
            self.connector.disconnect()
            return False
            
        self.is_running = True
        logger.info("--- Validator Startup Complete. Entering Main Loop. ---")
        return True

    def main_loop(self):
        """
        The main event loop of the application.
        """
        while self.is_running:
            try:
                # 1. Fetch any new candle data from MT5
                self.data_engine.update_data()
                
                # 2. Iterate through each symbol we are tracking
                for symbol in SYMBOLS_TO_TRACK:
                    if symbol not in self.strategies_by_instrument:
                        continue # Skip if no strategies are loaded for this symbol

                    # 3. Get the latest market data
                    raw_df = self.data_engine.get_dataframe(symbol)
                    if raw_df is None or len(raw_df) < self.data_engine.history_bars:
                        logger.warning(f"Not enough data for {symbol} to generate features. Skipping cycle.")
                        continue
                        
                    # 4. Generate the features required for signal evaluation
                    gold_df, silver_df = self.feature_engine.generate_gold_features(raw_df.copy())
                    
                    # Get the latest complete bar's features for signal checking and SL/TP calculation
                    # Use -2 to get the last *closed* candle (index -1 is the forming candle)
                    latest_gold_features = gold_df.iloc[-2] 
                    latest_silver_features = silver_df.iloc[-2]
                    
                    # 5. Check every loaded strategy against the latest features
                    for strategy in self.strategies_by_instrument[symbol]:
                        try:
                            # The core signal check using Pandas' query method
                            # We transpose (.T) to make columns accessible by name in query()
                            rule_matches = not latest_gold_features.to_frame().T.query(strategy['market_rule']).empty
                            
                            if rule_matches:
                                # If the rule matches, pass the signal to the trading engine
                                self.trading_engine.execute_trade(strategy, symbol, latest_silver_features)
                                
                        except Exception as e:
                            logger.error(f"Error evaluating rule for strategy {strategy.get('trigger_key', 'unknown')}: {e}")
                            
                # 6. Wait for the next cycle
                time.sleep(POLLING_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                self.is_running = False
            except Exception as e:
                logger.error(f"A critical error occurred in the main loop: {e}", exc_info=True)
                # Optional: logic to restart loop could go here
                self.is_running = False

    def shutdown(self):
        """
        Gracefully shuts down the application.
        """
        logger.info("--- Validator Shutting Down ---")
        if self.connector:
            self.connector.disconnect()
        logger.info("--- Shutdown Complete. ---")


def main():
    """Main execution function."""
    validator = MT5Validator()
    if validator.startup():
        validator.main_loop()
    else:
        # If startup failed, ensure we clean up just in case
        validator.shutdown()


if __name__ == "__main__":
    main()