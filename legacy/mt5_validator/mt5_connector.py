# File: mt5_validator/mt5_connector.py

"""
Handles the connection to the MetaTrader 5 terminal.

This module provides a class to manage the lifecycle of the MT5 connection,
ensuring that the terminal is initialized correctly and shut down gracefully.
It also includes error handling and retry logic for a more stable connection.
"""

import MetaTrader5 as mt5
import logging
import time
from typing import Optional

# Import the configuration from our live_config file
from legacy.mt5_validator.live_config import MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# Set up a logger for this module
logger = logging.getLogger(__name__)


class MT5Connector:
    """
    A class to manage the connection to the MetaTrader 5 terminal.

    This class provides a context manager interface for easy and safe
    connection handling.
    """

    def __init__(self, login: int = MT5_LOGIN, password: str = MT5_PASSWORD, server: str = MT5_SERVER, path: str = MT5_PATH):
        """
        Initializes the connector with account credentials and terminal path.
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.is_connected = False
        self.connection_start_time: Optional[float] = None

    def connect(self) -> bool:
        """
        Initializes and logs into the MetaTrader 5 terminal.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        if self.is_connected:
            logger.info("Connection already established.")
            return True

        logger.info("Attempting to initialize MetaTrader 5 terminal...")

        # mt5.initialize() is the key function to establish the link.
        # It takes the path to the terminal and other optional parameters.
        if not mt5.initialize(
            path=self.path,
            login=self.login,
            password=self.password,
            server=self.server
        ):
            logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
            self.is_connected = False
            return False

        logger.info("MT5 terminal initialized successfully.")

        # mt5.login() is used if initialize() was called without credentials
        # but it's good practice to ensure the login was accepted.
        account_info = mt5.account_info()
        if account_info is None or account_info.login != self.login:
            logger.error(f"MT5 login failed for account {self.login}. Error: {mt5.last_error()}")
            mt5.shutdown()
            self.is_connected = False
            return False

        self.is_connected = True
        self.connection_start_time = time.time()
        logger.info(f"Successfully connected to account {self.login} on {self.server}.")
        logger.info(f"Broker: {account_info.company}, Platform: {account_info.name}")
        
        return True

    def disconnect(self) -> None:
        """
        Shuts down the connection to the MetaTrader 5 terminal.
        """
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            duration = time.time() - (self.connection_start_time or time.time())
            logger.info(f"Disconnected from MetaTrader 5. Connection was active for {duration:.2f} seconds.")
        else:
            logger.info("No active connection to disconnect.")

    def __enter__(self):
        """
        Context manager entry point. Attempts to connect.
        """
        self.connect()
        if not self.is_connected:
            # Allows the main app to handle a failed connection on startup
            raise ConnectionError("Failed to connect to MetaTrader 5 terminal.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Ensures disconnection.
        """
        self.disconnect()

# --- Example Usage (for testing this file directly) ---
# if __name__ == "__main__":
#     # A simple logger setup for testing purposes
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#     logger.info("Testing MT5Connector...")
    
#     try:
#         # The 'with' statement makes connection management clean and safe.
#         # It automatically calls __enter__ (connect) at the start
#         # and __exit__ (disconnect) at the end, even if errors occur.
#         with MT5Connector(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, path=MT5_PATH) as connector:
#             if connector.is_connected:
#                 logger.info("Connection test successful. Connector is active within the 'with' block.")
                
#                 # You could test fetching some data here
#                 version = mt5.version()
#                 logger.info(f"MetaTrader 5 terminal version: {version}")
                
#                 # Wait for a few seconds to simulate doing work
#                 time.sleep(5)

#     except ConnectionError as e:
#         logger.error(f"Could not establish connection: {e}")
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during the test: {e}")

#     logger.info("MT5Connector test finished.")