import os
from pathlib import Path
import shutil
import sys
import logging
import pandas as pd

# --- CONFIGURATION IMPORT ---
# Calculate paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming structure: project_root/src/data_processing/bronze/bronze_data_generator.py
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
config_dir = os.path.join(project_root, "config")
sys.path.append(config_dir)

try:
    import config as c
    import paths as p # type: ignore
    from logger_setup import setup_logging # type: ignore
except ImportError as e:
    # Fallback logging if setup hasn't run
    logging.basicConfig(level=logging.INFO)
    logging.critical(f"Failed to import project modules. Ensure config.py are accesable: {e}")
    sys.exit(1)

# Initialize logger for this module
logger = logging.getLogger(__name__)

def load_and_clean_raw_ohlc_csv(file_path: Path) -> pd.DataFrame:
    # Load using columns defined in config
    df = pd.read_csv(file_path, sep=None, engine="python", header=None)
    
    # Validate column count
    required_cols = len(c.RAW_DATA_COLUMNS)
    if df.shape[1] < required_cols:
        # Fallback: Try to use as many as available if it matches minimal set
        if df.shape[1] >= 5: 
                df.columns = c.RAW_DATA_COLUMNS[:df.shape[1]]
        else:
            return f"ERROR: File has {df.shape[1]} columns, expected at least 5 (OHLCV)."
    else:
        df.columns = c.RAW_DATA_COLUMNS[:df.shape[1]]

    # Standardize Time
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    # If timezone information exists, remove it.
    if df['time'].dt.tz is not None:
        logger.debug(f"Timezone '{df['time'].dt.tz}' detected. Localizing to None (UTC-naive).")
        df['time'] = df['time'].dt.tz_localize(None)
        
    numeric_cols = ["open", "high", "low", "close"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop invalid rows
    initial_rows = len(df)
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    if len(df) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(df)} rows with invalid data.")
    
    df = df.sort_values('time').reset_index(drop=True)

    return df