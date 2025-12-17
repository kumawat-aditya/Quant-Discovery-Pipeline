# Bronze Layer: The Possibility Engine

This script, `bronze_data_generator.py`, serves as the foundational data generation layer for the entire quantitative trading strategy discovery pipeline. Its primary purpose is to systematically scan historical price data and generate a comprehensive dataset of _every conceivable winning trade_ based on a predefined grid of Stop-Loss (SL) and Take-Profit (TP) ratios.

This generated "universe of possibilities" acts as the bedrock for all subsequent analysis. The script operates by performing a brute-force simulation on every single candlestick, testing thousands of SL/TP combinations, and recording only those that would have resulted in a profitable outcome.

## Core Purpose

The goal of the Bronze layer is not to find strategies, but to create the raw material from which strategies will be mined. By pre-calculating every potential winning trade, we transform the problem from "what is a good trade?" to "what were the market conditions when all these good trades were possible?". This foundational dataset is essential for the feature enrichment and pattern recognition that occurs in the Silver and Platinum layers.

## How It Works

1.  **Data Ingestion**: The script reads a raw OHLC (Open, High, Low, Close) price data file in `.csv` format from the `raw_data/` directory.
2.  **Configuration Parsing**: It parses the input filename to determine the instrument and timeframe (e.g., `EURUSD60.csv` -> EURUSD, 60m). This information is used to load the correct simulation parameters (SL/TP grid, lookforward period, spread) from `config.py`.
3.  **Brute-Force Simulation**: For _every single candlestick_ in the input file, the script performs the following actions:
    - It simulates thousands of potential **BUY** trades, each with a unique SL/TP ratio combination from the configured grid.
    - It simulates thousands of potential **SELL** trades in the same manner.
    - For each simulated trade, it looks forward in time (up to the `MAX_LOOKFORWARD` limit) to determine the outcome.
4.  **Outcome Recording**:
    - If a simulated trade's Take-Profit price is hit _before_ its Stop-Loss price, the trade is considered a "winning possibility." Its details (entry time, trade type, entry price, SL/TP levels, exit time, etc.) are recorded.
    - If the Stop-Loss is hit first, or if the trade remains open for the entire `MAX_LOOKFORWARD` period without hitting either level, it is discarded and not saved.
5.  **Efficient Output**: The collected winning trades are written in large, efficient chunks to a **Parquet** file in the `bronze_data/` directory.

## Key Architectural Features

This script is engineered for high performance and stability to handle the computationally intensive task of simulating billions of trades.

- **Parquet Output**: Data is saved in the highly efficient, columnar Parquet format, which is significantly faster for downstream scripts (like the Silver layer) to read and process compared to CSV.
- **Numba JIT Compilation**: The core simulation logic (`find_winning_trades_numba`) is heavily accelerated with Numba's Just-In-Time compiler. This translates the Python simulation loop into machine code that runs at near-C language speeds, providing a massive performance boost.
- **Parallel Processing (Producer-Consumer Model)**: The script utilizes a `multiprocessing.Pool` to maximize CPU usage. The main data is split into chunks, and multiple "producer" worker processes simulate trades in parallel. A single main process acts as a "consumer," collecting the results and writing them to disk.
- **Memory Safety**: By using `pool.imap()` and a chunking/flushing mechanism, the script processes results as they are completed and writes them to disk periodically. This ensures that the system's RAM is not overwhelmed, preventing crashes even when processing very large datasets.
- **Cross-Platform Stability**: It uses a robust worker initializer (`init_worker`) to share large, read-only data (the main price DataFrame) with worker processes. This avoids data serialization issues and is a stable pattern that works reliably across different operating systems, including Windows.

## Inputs and Outputs

- **Input**: A single `.csv` file from `raw_data/`

  - **Example Filename**: `EURUSD60.csv`
  - **Required Columns**: `time`, `open`, `high`, `low`, `close`

- **Output**: A single `.parquet` file in `bronze_data/`
  - **Example Filename**: `EURUSD60.parquet`
  - **Schema (Columns)**:
    - `entry_time`: Timestamp of the trade entry.
    - `trade_type`: 'buy' or 'sell'.
    - `entry_price`: The closing price of the entry candle.
    - `sl_price`: The calculated Stop-Loss price.
    - `tp_price`: The calculated Take-Profit price.
    - `sl_ratio`: The Stop-Loss ratio used (e.g., 0.005 for 0.5%).
    - `tp_ratio`: The Take-Profit ratio used (e.g., 0.01 for 1%).
    - `exit_time`: Timestamp of the candle where the TP was hit.
    - `outcome`: 'win' (always, by design).

## Configuration

The behavior of the Bronze layer is controlled by parameters in the central `config.py` file.

- `MAX_CPU_USAGE`: Sets the number of CPU cores to use for parallel simulation.
- `BRONZE_INPUT_CHUNK_SIZE`: Defines how many candles are in each work package sent to a CPU core.
- `BRONZE_OUTPUT_CHUNK_SIZE`: Controls how many results are held in memory before being written to the Parquet file.
- `RAW_DATA_COLUMNS`: Specifies the expected column names in the input CSV.
- `TIMEFRAME_PRESETS`: This is the most critical setting. It defines the simulation grid (`SL_RATIOS`, `TP_RATIOS`) and the maximum trade duration (`MAX_LOOKFORWARD`) for different chart timeframes.
- `SIMULATION_SPREAD_PIPS`: Defines the assumed spread cost for each instrument, which is factored into the Take-Profit price check.

## How to Run the Script

You can run the script in three ways:

1.  **Via the Master Orchestrator (Recommended)**:
    The `orchestrator.py` script will run this layer automatically as the first step in the full pipeline.

    ```bash
    python orchestrator.py
    ```

2.  **Standalone (Interactive Mode)**:
    Run the script directly without arguments. It will scan for new `.csv` files in `raw_data` (that don't have a corresponding `.parquet` file in `bronze_data`) and present an interactive menu for you to choose which file(s) to process.

    ```bash
    python scripts/bronze_data_generator.py
    ```

3.  **Standalone (Targeted Mode)**:
    Run the script with a specific filename as a command-line argument to process only that file.
    ```bash
    python scripts/bronze_data_generator.py EURUSD60.csv
    ```
