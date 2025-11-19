# File: mt5_validator/live_config.py

"""
Configuration for the MT5 Live Strategy Validator.

This file contains all the necessary settings to connect to a MetaTrader 5
terminal, define the trading parameters, and manage risk for the live 
or forward-testing environment.
"""

import MetaTrader5 as mt5
import os

# --- 0. CRITICAL PLATFORM & PATH CONFIGURATION ---

# !! IMPORTANT !!
# The MetaTrader 5 Python library is a WINDOWS-ONLY library. This script
# MUST be run on a Windows machine (or a Windows Virtual Machine on a Mac/Linux)
# that also has the MT5 desktop terminal installed and running.

# The full path to the MT5 terminal executable within its Windows environment.
# Example for a standard Windows installation:
MT5_PATH: str = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# Since this script is in a subfolder, we need to define the path back to the
# main project root to find the research data.
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAMOND_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'diamond_data')

# --- 1. MT5 Connection Settings ---
# Your MT5 account credentials. It is highly recommended to use environment
# variables or a secure secret management system in a real production environment.
MT5_LOGIN: int = 99066452  # Replace with your account number
MT5_PASSWORD: str = "ZgMf_6Ld"
MT5_SERVER: str = "MetaQuotes-Demo"


# --- 2. Trading & Market Parameters ---
# Defines which markets to monitor and the basic trading rules.

# A list of symbols the validator will track and trade.
# IMPORTANT: The symbol names must match exactly what your broker uses
# (e.g., 'EURUSD', 'EURUSD.pro', 'GBPUSD').
SYMBOLS_TO_TRACK: list[str] = [
    "EURUSD",
]

# The timeframe to operate on.
TIMEFRAME = mt5.TIMEFRAME_M15

# The default position size (in lots) for each trade.
LOT_SIZE: float = 0.01

# The number of historical bars to fetch for indicator calculation.
# MUST be larger than the longest indicator period (e.g., 200-period SMA).
HISTORY_BARS_COUNT: int = 500


# --- 3. Risk Management Settings ---
# Critical safety parameters to prevent runaway trading.

# The maximum number of concurrent open trades allowed for a SINGLE strategy.
MAX_OPEN_TRADES_PER_STRATEGY: int = 1


# --- 4. Operational Settings ---
# Controls the timing and execution loop of the application.

# The interval in seconds for the main loop to run.
POLLING_INTERVAL_SECONDS: int = 60

# def main():
#     print("Program started...")
#     print(f"showing project path {PROJECT_ROOT_PATH}")
#     print(f"showing diamond path {DIAMOND_DATA_PATH}")
#     print(f"showing platinum path {PLATINUM_DATA_PATH}")
#     print(f"showing timefram: {TIMEFRAME}")

# def run_bot():
#     print("Bot is running...")

# if __name__ == "__main__":
#     main()
