"""
Reusable Logging Setup Utility for the Strategy Finder Pipeline.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str, console_level, file_level, script_name) -> None:
    """
    Configures the root logger to output to both console and a rotating file.

    Args:
        log_dir (str): The directory where log files will be stored.
        console_level: The logging level for console output.
        file_level: The logging level for file output.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)

        # Get the name of the script that is setting up the logging
        log_file_path = os.path.join(log_dir, f"{script_name}.log")

        # Define the log format
        log_format = logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG) # Set the lowest level for the logger itself

        # Prevent duplicate handlers if this function is called multiple times
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # --- Console Handler ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)

        # --- File Handler ---
        # RotatingFileHandler keeps the log files from growing indefinitely.
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=5*1024*1024, backupCount=5 # 5 MB per file, 5 backups
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)

    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to configure professional logging: {e}")
        logging.warning("Falling back to basic console logging.")